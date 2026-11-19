"""
Barge-in: someone talks while the robot speaks -> it stops and listens (like a person, like ChatGPT's voice mode).

The robot's own voice reaches the mic about as loud as a person 1-2 m away (-35 vs -35..-40 dBFS, 2026-11-01), and
the small speaker + the body distort it: a waveform echo canceller took out only 6-8 dB. What works (recording
2026-11-01, Selim talking over the 4th of 4 sentences): predict how loud the echo SHOULD be in each of 9 frequency
bands from what is being played (the speaker is -42 dB below 250 Hz, -6 dB at 1.8-2.6 kHz), with the room's tail
fading BARGE_DECAY_DB per 32 ms; a band at least BARGE_THRESHOLD_DB over that prediction is someone else's sound.
3+ hot bands in 5 of the last 8 chunks = barge-in. On the recording (fitted on two sentences, tested on the third):
no false alarm at 12 dB / one at 9, Selim caught 0.1 s (9 dB) to 1.5 s (12 dB) after he started.

    echo = EchoRef()                       # tts.Speaker adds every piece it plays (and when it plays)
    det = BargeIn(echo, tune_file=config.BARGE_TUNE_FILE)
    if det.feed(chunk, t):                 # robot/mic.py, every 32 ms chunk while the robot speaks
        ...                                # (high-passed, un-gained, 16 kHz; t = the chunk's start)
    det.false_alarm()                      # the conversation: what it "heard" was only its own voice

How loud the echo is per band depends on the speaker, its mounting and the level: a damaged or new speaker cut every
reply at once (2026-11-02, level 0.2: the echo of a distorting speaker is far over the old prediction). So the
detector tunes itself: `code/hwtest/bargecal.py` measures the echo with nobody talking (once per speaker) and writes
BARGE_TUNE_FILE; every false alarm the conversation finds (its own words heard back after a cut) raises the bands
that fired, and is saved there too.
"""
import collections
import json
import threading
import time
from pathlib import Path

import numpy as np

from . import config

RATE = 16000
FRAME = 512                                 # robot/mic.py CHUNK: 32 ms
EDGES = (100, 250, 500, 800, 1200, 1800, 2600, 3600, 5000, 7500)
_FREQS = np.fft.rfftfreq(FRAME, 1 / RATE)
_BANDS = [(_FREQS >= a) & (_FREQS < b) for a, b in zip(EDGES, EDGES[1:])]
_WIN = np.hanning(FRAME)


def band_power(x):
    """Power in each of the 9 bands of one 512-sample frame (Hann window)."""
    x = np.asarray(x, np.float64)
    if len(x) < FRAME:
        x = np.concatenate([x, np.zeros(FRAME - len(x))])
    s = np.abs(np.fft.rfft(x[:FRAME] * _WIN)) ** 2
    return np.array([s[m].sum() for m in _BANDS])


def db(p):
    return 10 * np.log10(np.asarray(p) + 1e-16)


def to_16k_mono(frames):
    """48 kHz 16-bit stereo bytes (what the speaker plays) -> 16 kHz mono float (left channel, 3-sample mean:
    enough for band powers)."""
    x = np.frombuffer(frames[:len(frames) // 4 * 4], "<i2")[::2].astype(np.float32) / 32768
    n = len(x) // 3
    return x[:n * 3].reshape(n, 3).mean(1)


class EchoRef:
    """What the speaker plays, and when: pieces of 16 kHz mono with their start time (time.monotonic). Kept for
    `keep_s`. The start of a piece is predicted when it is sent (right after the one before, or a period from now
    after silence) and corrected when speakerd reports its end."""

    def __init__(self, keep_s=10.0, clock=None, start_delay_s=0.03):
        import time
        self.clock = clock or time.monotonic
        self.keep_s, self.start_delay_s = keep_s, start_delay_s
        self.pieces = collections.deque()           # [start, samples, id]
        self.lock = threading.Lock()
        self.end = 0.0                              # when the last piece sent will have played

    def add(self, frames, pid=None):
        x = to_16k_mono(frames)
        with self.lock:
            now = self.clock()
            start = max(now + self.start_delay_s, self.end)
            self.end = start + len(x) / RATE
            self.pieces.append([start, x, pid])
            while self.pieces and self.pieces[0][0] + len(self.pieces[0][1]) / RATE < now - self.keep_s:
                self.pieces.popleft()

    def ended(self, pid, end):
        """speakerd: piece pid's last sample leaves the speaker at `end` - shift it (and those after it)."""
        with self.lock:
            for i, p in enumerate(self.pieces):
                if p[2] == pid:
                    shift = end - (p[0] + len(p[1]) / RATE)
                    if abs(shift) > 0.002:
                        for q in list(self.pieces)[i:]:
                            q[0] += shift
                        self.end += shift
                    return

    def cut(self, at):
        """Stopped (barge-in): nothing plays after `at`."""
        with self.lock:
            for p in self.pieces:
                n = max(0, int(round((at - p[0]) * RATE)))
                if n < len(p[1]):
                    p[1] = p[1][:n]
            self.end = min(self.end, at)

    def segment(self, t0, n=FRAME):
        """The n samples that played from t0 (zeros where nothing played)."""
        out = np.zeros(n, np.float32)
        with self.lock:
            for start, x, _ in self.pieces:
                i0 = int(round((t0 - start) * RATE))
                if i0 >= len(x) or i0 + n <= 0:
                    continue
                a, b = max(0, i0), min(len(x), i0 + n)
                out[a - i0:b - i0] += x[a:b]
        return out

    def playing(self, t, tail_s=1.0):
        """Did anything play between t - tail_s and t?"""
        with self.lock:
            return any(s <= t and s + len(x) / RATE >= t - tail_s for s, x, _ in self.pieces)


def load_tuning(path):
    """BARGE_TUNE_FILE -> {"coupling": [9 dB], "noise": [9 dB], ...} or {} (none yet / unreadable / wrong shape)."""
    try:
        with open(Path(path).expanduser(), encoding="utf-8") as f:
            t = json.load(f)
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(t, dict):
        return {}
    out = {k: [float(v) for v in t[k]] for k in ("coupling", "noise")
           if isinstance(t.get(k), list) and len(t[k]) == len(_BANDS)
           and all(isinstance(v, (int, float)) and np.isfinite(v) for v in t[k])}
    if isinstance(t.get("lag_s"), (int, float)) and 0.0 <= t["lag_s"] <= 0.5:
        out["lag_s"] = float(t["lag_s"])
    for k in ("when", "how", "false_alarms"):
        if k in t:
            out[k] = t[k]
    return out


def save_tuning(path, tuning):
    """Write BARGE_TUNE_FILE atomically. Never raises (tuning lost is no reason to stop the ears)."""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(tuning, indent=1), encoding="utf-8")
        tmp.replace(p)
        return True
    except (OSError, TypeError, ValueError):
        return False


class BargeIn:
    def __init__(self, echo, lag_s=None, threshold_db=None, decay_db=None, coupling=None, noise=None,
                 need=5, of=8, hot_bands=3, learn=0.02, tune_file=None):
        self.echo = echo
        self.tune_file = tune_file
        tuned = load_tuning(tune_file) if tune_file else {}
        self.tuned = tuned.get("how")               # "calibrated" / "learnt" / None = config's values (logs)
        self.false_alarms = int(tuned.get("false_alarms", 0) or 0)
        self.lag_s = (tuned.get("lag_s", config.BARGE_LAG_S)) if lag_s is None else lag_s
        self.threshold = config.BARGE_THRESHOLD_DB if threshold_db is None else threshold_db
        self.decay = config.BARGE_DECAY_DB if decay_db is None else decay_db
        self.coupling = np.array(tuned.get("coupling", config.BARGE_COUPLING_DB) if coupling is None else coupling,
                                 np.float64)
        self.noise = np.array(tuned.get("noise", config.BARGE_NOISE_DB) if noise is None else noise, np.float64)
        self.need, self.hot_bands, self.learn = need, hot_bands, learn
        self.recent = collections.deque(maxlen=of)
        self.excess = collections.deque(maxlen=of)  # dB over the prediction of the same chunks (None: nothing played)
        self.fired = []                             # at the last barge-in: the hot chunks' excess (false_alarm())
        self.tail = np.zeros(len(_BANDS))           # the echo's power still in the room (decaying), per band
        self.last_excess = None                     # dB over the prediction, per band (logs / tuning)
        self._lock = threading.Lock()

    def hot_s(self):
        """Seconds since the first chunk of the current window that heard someone over the echo (0 if none)."""
        r = list(self.recent)
        if True not in r:
            return 0.0
        return (len(r) - r.index(True)) * FRAME / RATE

    def reset(self):
        self.recent.clear()
        self.excess.clear()
        self.tail[:] = 0.0

    def false_alarm(self, margin_db=1.0, step_max_db=None):
        """The last barge-in was only the robot's own echo (the conversation heard its own words back): every band
        that fired is predicted louder from now on - by how far it was over the threshold (+ margin, at most
        BARGE_FALSE_STEP_DB at a time, never over BARGE_COUPLING_MAX_DB) - and saved. Returns the dB added."""
        step_max_db = config.BARGE_FALSE_STEP_DB if step_max_db is None else step_max_db
        with self._lock:
            fired, self.fired = self.fired, []
            if not fired:
                return np.zeros(len(_BANDS))
            over = np.max(np.array(fired), axis=0) - self.threshold
            add = np.where(over > 0, np.clip(over + margin_db, 0.0, step_max_db), 0.0)
            new = np.minimum(self.coupling + add, config.BARGE_COUPLING_MAX_DB)
            add = np.maximum(new - self.coupling, 0.0)
            self.coupling = np.maximum(self.coupling, new)
            self.false_alarms += 1
            self.tuned = self.tuned or "learnt"
        self.save()
        return add

    def save(self):
        """The tuning as it is now -> tune_file (no file: nothing)."""
        if not self.tune_file:
            return False
        with self._lock:
            tuning = dict(coupling=[round(float(v), 2) for v in self.coupling],
                          noise=[round(float(v), 2) for v in self.noise], lag_s=self.lag_s,
                          how=self.tuned or "learnt", false_alarms=self.false_alarms,
                          when=time.strftime("%Y-%m-%d %H:%M:%S"))
        return save_tuning(self.tune_file, tuning)

    def learn_noise(self, chunk):
        """A quiet chunk while the robot is silent (nobody speaking): the room's floor, band by band (a TV or a fan
        raises it - no false barge-ins from them)."""
        self.noise += 0.02 * (db(band_power(chunk)) - self.noise)

    def feed(self, chunk, t):
        """One mic chunk (16 kHz, high-passed, un-gained) that started at t. True = someone talks over the robot."""
        ref = self.echo.segment(t - self.lag_s, len(chunk))
        self.tail = np.maximum(band_power(ref), self.tail * 10 ** (-self.decay / 10))
        self.tail[self.tail < 1e-9] = 0.0           # faded into the room's noise: nothing of it left
        mic = db(band_power(chunk))
        if not self.tail.any():
            self.noise += 0.01 * (np.minimum(mic, self.noise + 6) - self.noise)   # the room alone: its floor
            self.recent.append(False)
            self.excess.append(None)
            return False
        with self._lock:
            pred = np.maximum(db(self.tail) + self.coupling, self.noise)
            excess = mic - pred
            self.last_excess = excess
            hot = int((excess > self.threshold).sum()) >= self.hot_bands
            self.recent.append(hot)
            self.excess.append(excess if hot else None)
            if not hot:                             # the robot alone: learn how loud its echo is, band by band
                strong = db(self.tail) + self.coupling > self.noise + 10
                step = np.clip(mic - db(self.tail) - self.coupling, -3, 3)
                self.coupling[strong] += self.learn * step[strong]
            fire = sum(self.recent) >= self.need
            if fire:                                # kept until the conversation knows whether it was its own voice
                self.fired = [e for e in self.excess if e is not None]
        return fire


def fit_coupling(mic_db, tail_db, noise_db, default=None, pct=97.0, min_chunks=20, margin_db=1.0):
    """The speaker -> mic coupling per band from a recording with nobody talking (code/hwtest/bargecal.py):
    mic_db, tail_db = (chunks, 9) band powers of the mic and of the decaying echo reference (lag applied).
    Per band: the pct-th percentile of mic - reference over the chunks where the echo is clearly over the room
    (+ margin) - so the robot's own voice is almost never `threshold` over it. Bands with too few such chunks keep
    `default`. Returns (coupling, chunks used per band)."""
    default = np.array(config.BARGE_COUPLING_DB if default is None else default, np.float64)
    mic_db, tail_db = np.asarray(mic_db, np.float64), np.asarray(tail_db, np.float64)
    out, used = default.copy(), np.zeros(len(default), int)
    for b in range(len(default)):
        ok = (mic_db[:, b] > np.asarray(noise_db)[b] + 6) & (tail_db[:, b] > tail_db[:, b].max() - 50)
        used[b] = int(ok.sum())
        if used[b] >= min_chunks:
            out[b] = min(float(np.percentile(mic_db[ok, b] - tail_db[ok, b], pct)) + margin_db,
                         config.BARGE_COUPLING_MAX_DB)
    return out, used


def save_snippet(ring, echo, lag_s=None, out_dir=None, keep=None, when=None):
    """A barge-in, kept for tuning: stereo 16 kHz WAV - left = the mic (high-passed, un-gained), right = what the
    robot played, shifted by the echo's lag (both normalised together). Never raises."""
    import wave
    try:
        lag_s = config.BARGE_LAG_S if lag_s is None else lag_s
        d = Path(out_dir or config.BARGE_SAVE_DIR).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        mic = np.concatenate([c for c, _ in ring]).astype(np.float32)
        ref = echo.segment(ring[0][1] - lag_s, len(mic))
        peak = max(float(np.abs(mic).max()), float(np.abs(ref).max()), 1e-6)
        stereo = np.stack([mic, ref], 1) / peak * 0.9
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(when)) + ".wav"
        with wave.open(str(d / name), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((stereo * 32767).astype("<i2").tobytes())
        prune(d, keep or config.BARGE_SAVE_N)
    except Exception:  # noqa: BLE001 - tuning data must never stop the ears
        pass


def prune(folder, keep, pattern="*.wav"):
    """Keep the newest `keep` files (by time written - names were only HH:MM:SS until 2026-11-02, so sorting them
    by name deleted every file of a new day at once, today's 15:37 before yesterday's 20:19)."""
    files = []
    for p in Path(folder).glob(pattern):
        try:
            files.append((p.stat().st_mtime, p.name, p))
        except OSError:
            continue
    for _, _, old in sorted(files)[:-keep] if keep > 0 else sorted(files):
        try:
            old.unlink()
        except OSError:
            pass
