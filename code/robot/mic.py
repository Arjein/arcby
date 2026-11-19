"""
Ears: the I2S microphone as a stream of utterances. Every 32 ms chunk goes through the speech detector
(robot/vad.py); only stretches with speech come out - never silence, never while the robot itself speaks, and
nothing is written to disk (no audio is kept, Selim 2026-10-31).

    ears = Ears(Vad(), muted=lambda: voice.busy, on_utterance=handle)   # handle(pcm, t0, t1[, onset]): 16 kHz
    ears.start(stop_event)                                              # float32; onset = first speech chunk

The callback gets `onset=` (time of the first speech chunk; t0 is the start of the pre-roll, ~0.2-0.3 s earlier)
only if it accepts that keyword. Times come from a sample counter (not the time a chunk was read).

Segmenter (pure, testable): speech starts after MIC_START_S above MIC_VAD_ON (with MIC_PREROLL_S of what came
before, so the first syllable is not lost) and ends after MIC_END_S below MIC_VAD_OFF; at least MIC_MIN_S of actual speech, at most MIC_MAX_S long.
"""
import collections
import inspect
import logging
import subprocess
import threading
import time
import wave

import numpy as np

from . import config
from .vad import CHUNK, RATE

log = logging.getLogger(__name__)
CHUNK_S = CHUNK / RATE                      # 32 ms


class Utterance(tuple):
    """(pcm, t0, t1) - unpacks like the old 3-tuple - plus .onset: the time of the first speech chunk."""

    def __new__(cls, pcm, t0, t1, onset):
        u = super().__new__(cls, (pcm, t0, t1))
        u.onset = onset
        return u


class Segmenter:
    def __init__(self, on=config.MIC_VAD_ON, off=config.MIC_VAD_OFF, start_s=config.MIC_START_S,
                 end_s=config.MIC_END_S, preroll_s=config.MIC_PREROLL_S, min_s=config.MIC_MIN_S,
                 max_s=config.MIC_MAX_S):
        self.on, self.off = on, off
        self.start_n = max(1, round(start_s / CHUNK_S))
        self.end_n = max(1, round(end_s / CHUNK_S))
        self.min_n, self.max_n = round(min_s / CHUNK_S), round(max_s / CHUNK_S)
        self.pre = collections.deque(maxlen=max(1, round(preroll_s / CHUNK_S)))
        self.reset()

    def reset(self):
        self.buf, self.t0, self.onset, self.loud, self.quiet, self.speech = None, None, None, [], 0, 0
        self.pre.clear()

    def force_start(self, chunks):
        """Start an utterance now with these (chunk, t) - the speech that talked over the robot (barge-in): its
        first words came before the ears were listening."""
        if not chunks:
            return
        self.buf = [c for c, _ in chunks]
        self.t0 = self.onset = chunks[0][1]
        self.speech = len(chunks)
        self.quiet = 0
        self.loud = []
        self.pre.clear()

    @property
    def idle(self):
        """No utterance going on and none starting."""
        return self.buf is None and not self.loud

    def feed(self, chunk, p, t):
        """One chunk (CHUNK samples), its speech probability and its start time. Returns an Utterance
        ((pcm, t0, t1) + .onset) when an utterance ended, else None."""
        if self.buf is None:
            if p >= self.on:
                self.loud.append((chunk, t))
                if len(self.loud) >= self.start_n:  # speech for real: start, with the pre-roll in front
                    self.buf = [c for c, _ in self.pre] + [c for c, _ in self.loud]
                    self.t0 = (self.pre[0][1] if self.pre else self.loud[0][1])
                    self.onset = self.loud[0][1]
                    self.speech = len(self.loud)
                    self.pre.clear()
                    self.loud = []
                return None
            for c in self.loud:                     # a click, not speech: it becomes pre-roll
                self.pre.append(c)
            self.loud = []
            self.pre.append((chunk, t))
            return None
        self.buf.append(chunk)
        self.speech += p >= self.on
        self.quiet = self.quiet + 1 if p < self.off else 0
        if self.quiet >= self.end_n or len(self.buf) >= self.max_n:
            ended = self.quiet >= self.end_n
            n = len(self.buf) - (self.quiet if ended else 0) + min(self.quiet, 5)
            pcm = np.concatenate(self.buf[:n])
            out = Utterance(pcm, self.t0, t + CHUNK_S, self.onset)
            # what came after the cut is the next one's pre-roll: a second phrase 0.6-0.9 s later (or the rest
            # of a 12 s cut) keeps its first syllable
            tail = self.buf[-self.pre.maxlen:]
            self.pre.clear()
            for k, c in enumerate(tail):
                self.pre.append((c, t - (len(tail) - 1 - k) * CHUNK_S))
            self.buf, self.quiet = None, 0
            if self.speech >= self.min_n:           # enough actual speech (not pre-roll / tail)
                return out
        return None


class HighPass:
    """Streaming FIR high-pass (windowed sinc, `taps` long, exactly 0 at DC) - numpy only, ~5 ms of CPU per second."""

    def __init__(self, cutoff_hz=config.MIC_HIGHPASS_HZ, rate=RATE, taps=401):
        n = np.arange(taps) - (taps - 1) / 2
        lp = np.sinc(2 * cutoff_hz / rate * n) * np.hamming(taps)
        lp /= lp.sum()
        self.h = -lp
        self.h[(taps - 1) // 2] += 1.0                 # delta - low-pass = high-pass
        self.h = self.h.astype(np.float32)
        self.reset()

    def reset(self):
        self.tail = None

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        if self.tail is None:                       # start from the first value: no jump from the mic's offset
            self.tail = np.full(len(self.h) - 1, x[0] if len(x) else 0.0, np.float32)
        buf = np.concatenate([self.tail, x])
        self.tail = buf[-(len(self.h) - 1):]
        return np.convolve(buf, self.h, mode="valid").astype(np.float32)


class AutoGain:
    """Slow automatic gain: follows the loud parts (an envelope that falls by half in ~4 s) and brings them to
    `target` RMS - never less than x1, never more than max_gain (silence must not be blown up to speech).
    learn(rms) only with speech (Ears: the VAD said so): a click or a servo moving the head (~-50 dBFS) used to
    pull the gain from x120 to x6 for ~20 s, and quiet speech went unheard."""

    def __init__(self, target=config.MIC_GAIN_TARGET, max_gain=config.MIC_GAIN_MAX, start_rms=0.003, fall=0.995,
                 limit=0.9, too_loud_s=config.MIC_GAIN_TOO_LOUD_S):
        self.target, self.max_gain, self.fall, self.limit = target, max_gain, fall, limit
        self.loud = start_rms
        self.too_loud_n = max(1, round(too_loud_s / (CHUNK / RATE)))
        self._limited = 0                   # chunks in a row the limiter had to turn down by > 2x

    @property
    def gain(self):
        return min(self.max_gain, max(1.0, self.target / max(self.loud, 1e-6)))

    def __call__(self, x):
        """x at the current gain, turned down for this chunk if its peak would go over `limit` - never clipped:
        a clipped square wave is not speech to the VAD, so the gain never learnt and loud speech stayed unheard
        (replay 2026-11-01: -30 dBFS speech 11 of 28 heard clipped, 26 of 28 limited)."""
        x = np.asarray(x, np.float32)
        peak = float(np.abs(x).max()) if len(x) else 0.0
        g = self.gain
        if peak * g > self.limit:
            g = self.limit / peak
        self._last_cut = g < self.gain / 2
        return (x * g).astype(np.float32)

    def learn(self, x, speech):
        """After each chunk: the envelope falls; a speech chunk (x before the gain) can raise it - and so can a
        long stretch (too_loud_s) the limiter had to turn down a lot (someone loud whom the VAD did not take for
        speech yet). Short noises (a click, the servos) still leave it alone."""
        rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
        self._limited = self._limited + 1 if getattr(self, "_last_cut", False) else 0
        learn = speech or self._limited >= self.too_loud_n
        self.loud = max(self.loud * self.fall, rms if learn else 0.0, 1e-6)


def normalise(pcm, peak=config.MIC_UTTERANCE_PEAK):
    """One utterance scaled as a whole so its loudest sample is `peak` (no clipping, the level of the words kept)."""
    top = float(np.abs(pcm).max()) if len(pcm) else 0.0
    return (pcm * (peak / top)).astype(np.float32) if top > 0 else pcm


class Decimator:
    """Streaming 48 kHz -> 16 kHz: a 63-tap low-pass (7 kHz) then every 3rd sample, state kept across chunks.
    (The mean of 3 samples let 8-16 kHz fold into the speech band at up to 0.67.)"""

    def __init__(self, factor=3, taps=63, cutoff_hz=7000, rate=48000):
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * cutoff_hz / rate * n) * np.hamming(taps)
        self.h = (h / h.sum()).astype(np.float32)
        self.factor = factor
        self.tail = None
        self.phase = 0                      # index in the next chunk of the next sample to keep

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        if self.tail is None:               # start from the first value: no jump from the mic's offset
            self.tail = np.full(len(self.h) - 1, x[0] if len(x) else 0.0, np.float32)
        buf = np.concatenate([self.tail, x])
        self.tail = buf[-(len(self.h) - 1):]
        y = np.convolve(buf, self.h, mode="valid")[self.phase::self.factor]
        self.phase = (self.phase - len(x)) % self.factor
        return y.astype(np.float32)


def to_16k_mono(raw, channel=0, down=None):
    """arecord S32_LE stereo 48 kHz bytes -> float32 mono 16 kHz (-1..1): one channel, low-passed and decimated
    (`down`: a Decimator kept across chunks; a fresh one if None). Nothing is subtracted per chunk: removing each
    32 ms chunk's mean turned the mic's slow drift into a 31 Hz click train louder than quiet speech (-56 dBFS
    from silence, 2026-11-01) - the HighPass after this removes DC and drift smoothly."""
    d = np.frombuffer(raw, "<i4").reshape(-1, 2)[:, channel].astype(np.float32) / 2 ** 31
    return (down or Decimator())(d)


def _accepts(fn, name):
    """Does fn take the keyword `name` (or **kwargs)?"""
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(q.name == name or q.kind == q.VAR_KEYWORD for q in params)


class Ears:
    def __init__(self, vad, card=config.MIC_CARD, muted=lambda: False, on_utterance=None, segmenter=None,
                 clock=time.monotonic, popen=subprocess.Popen):
        self.vad, self.card, self.muted, self.on_utterance = vad, card, muted, on_utterance
        self.seg = segmenter or Segmenter()
        self.down = Decimator()                     # 48 -> 16 kHz without folding 8-16 kHz into speech
        self.highpass = HighPass()                  # the rumble out, before the VAD and the speech-to-text...
        self.agc = AutoGain()                       # ...and the quiet speech up
        self._clock, self._popen = clock, popen
        self._wants_onset = _accepts(on_utterance, "onset")
        self.proc = None
        self.thread = None
        self.level = 0.0                    # recent RMS (for logs / tuning)
        self.heard = 0                      # utterances so far
        self.chunks = 0                     # 32 ms pieces read so far (31 a second when it keeps up)
        self.vad_max = 0.0                  # highest speech probability since status() last read it
        self.restarts = 0                   # times arecord ended and was started again
        self.debug = None                   # deque of (chunk after high-pass, after gain, VAD p) - ARCBY_EARS_DEBUG
        self._idle = 0                      # chunks since the VAD was last reset with nothing going on
        self._anchor = None                 # (clock, chunk count) the sample counter is timed from
        self._read = 0                      # chunks read from this arecord
        self._unmute_at = 0.0               # muted until then (the robot's voice still in the air / in buffers)
        self.listener = None                # live STT (robot/conversation.py): speech_started(onset, pcm) as a
                                            # sentence starts, speech_chunk(pcm) while it goes on, speech_ended(kept)
        self._live_gain = 1.0               # the gain at the sentence's start: one level for the whole stream
        self.barge = None                   # robot/bargein.py BargeIn: someone talks while the robot speaks...
        self.on_barge = None                # ...-> on_barge() (True = the robot stopped): listen at once
        self._ring = collections.deque(maxlen=max(1, round(config.BARGE_RING_S / CHUNK_S)))
        self.on_barge_saved = None          # on_barge_saved([(chunk, t), ...]): the mic before a barge-in, kept
        self._barge_until = 0.0             # after a barge-in: not muted until then (its voice fading out)
        self.barged = 0                     # barge-ins so far

    def _open(self):
        return self._popen(["arecord", "-q", "-D", f"plughw:{self.card},0", "-c", "2", "-r", "48000",
                            "-f", "S32_LE", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def start(self, stop):
        self.proc = self._open()
        self.thread = threading.Thread(target=self._run, args=(stop,), name="ears", daemon=True)
        self.thread.start()
        return self

    def _chunk_time(self):
        """Start time of the chunk just read, from the sample count since the stream started (read times jump when
        the thread catches up); re-anchored if the stream lost samples (an overrun: the clock runs ahead)."""
        now = self._clock()
        self._read += 1
        if self._anchor is None or now - (self._anchor[0] + (self._read - self._anchor[1]) * CHUNK_S) > 0.5:
            self._anchor = (now - CHUNK_S, self._read - 1)
        return self._anchor[0] + (self._read - 1 - self._anchor[1]) * CHUNK_S

    def _run(self, stop):
        need = CHUNK * 3 * 2 * 4               # 512 samples at 16 kHz = 1536 stereo int32 frames at 48 kHz
        backoff = 1.0
        try:
            while not stop.is_set():
                raw = self.proc.stdout.read(need) if self.proc is not None else b""
                if not raw or len(raw) < need:
                    # arecord ended (card busy / gone): start it again, waiting longer each time - deaf, not dead
                    log.warning("ears: the microphone stream ended - again in %.0f s", backoff)
                    self.close()
                    if stop.wait(backoff):
                        return
                    backoff = min(backoff * 2, 60.0)
                    self.restarts += 1
                    self._reset_stream()
                    try:
                        self.proc = self._open()
                    except OSError as e:
                        log.warning("ears: arecord did not start (%s)", e)
                        self.proc = None
                    continue
                if self._read > 300:            # ~10 s of sound since the last restart: healthy again
                    backoff = 1.0
                self._chunk(raw)
        finally:
            self.close()

    def _reset_stream(self):
        self.seg.reset()
        self.vad.reset()
        self.down = Decimator()
        self.highpass.reset()
        self._anchor, self._read = None, 0

    def _chunk(self, raw):
        t = self._chunk_time()
        chunk = self.highpass(to_16k_mono(raw, down=self.down))
        self.level = 0.9 * self.level + 0.1 * float(np.sqrt((chunk ** 2).mean()))
        now = self._clock()
        if self.muted():                    # the robot is speaking: its own voice is not an utterance -
            self._unmute_at = now + config.MIC_UNMUTE_HOLD_S    # and must not teach the gain how loud people are
        if now < self._unmute_at and now >= self._barge_until:  # ...nor its last syllable still in the buffers
            if self.seg.buf is not None:
                self._tell("speech_ended", False)
            if not self.seg.idle or self.seg.pre:
                self.seg.reset()            # nothing heard during it becomes pre-roll either
            self.vad.reset()
            self._idle = 0
            if self.barge is not None and self.on_barge is not None:
                self._ring.append((chunk, t))
                if not self.barge.feed(chunk, t):
                    return
                stopped = self.on_barge()
                if not stopped:
                    log.info("ears: a voice over the robot's own, but nobody looks at it - not stopped")
                    self.barge.reset()      # heard again from scratch (not every 32 ms)
                else:
                    # someone talks over the robot: it stops (on_barge), the ears listen now - the interrupting
                    # speech heard so far becomes the pre-roll of the utterance. "listen": it had finished already
                    # (a quick answer in the hold after its voice) - heard the same way, but no barge-in to count
                    really = stopped is True
                    if really:
                        self.barged += 1
                    self._barge_until = now + config.BARGE_LISTEN_S
                    self._unmute_at = now
                    started = self.barge.hot_s() + config.BARGE_PREROLL_S
                    self.barge.reset()
                    ring = list(self._ring)
                    self._ring.clear()
                    # from where their voice started: earlier is only the robot's own words (the STT heard the
                    # robot's "İyiyim Selim" instead of Selim's "hava nasıl", 2026-11-01)
                    keep = ring[-max(1, round(started / CHUNK_S)):]
                    self.seg.force_start(keep)       # the utterance starts with the speech that interrupted
                    self.vad.reset()
                    if self.listener is not None:
                        self._live_gain = self.agc.gain
                        self._tell("speech_started", self.seg.onset, self._live(np.concatenate(self.seg.buf)))
                    if really and self.on_barge_saved is not None:
                        self.on_barge_saved(ring)    # kept for tuning (robot/bargein.py save_snippet)
            return
        self._ring.clear()
        gained = self.agc(chunk)
        p = self.vad.prob(gained)
        if self.debug is not None:
            self.debug.append((chunk, gained, p))
        self.chunks += 1
        # Silero's state drifts on long, loud (x120) room noise until it never says speech again: after a minute
        # 0 of 107 speech chunks found, 102 after a reset (2026-11-01). Reset every MIC_VAD_RESET_S while nothing
        # is being heard - whatever the noise scores (a TV at p 0.4 used to hold the reset off for ever).
        self._idle = self._idle + 1 if self.seg.idle else 0
        if self._idle >= config.MIC_VAD_RESET_S / CHUNK_S:
            self.vad.reset()
            self._idle = 0
        self.vad_max = max(self.vad_max, p)
        if self.barge is not None and p < config.MIC_VAD_OFF and self.seg.idle:
            self.barge.learn_noise(chunk)   # the room's own floor, for telling a voice over the robot's echo
        self.agc.learn(chunk, p >= config.MIC_VAD_ON)
        # the utterance itself is cut from the UN-gained sound: the gain waits at up to x120 between sentences,
        # and the first syllables were clipped ("napıyorsun" -> "Dua etsin", 2026-11-01)
        was = self.seg.buf is not None
        got = self.seg.feed(chunk, p, t)
        if self.listener is not None:
            if not was and self.seg.buf is not None:
                self._live_gain = self.agc.gain
                self._tell("speech_started", self.seg.onset, self._live(np.concatenate(self.seg.buf)))
            elif was and self.seg.buf is not None:
                self._tell("speech_chunk", self._live(chunk))
            elif was:
                self._tell("speech_ended", got is not None)
        if got is not None:
            self.vad.reset()                # ...and after each utterance
            self._idle = 0
            self.heard += 1
            if self.on_utterance is not None:
                pcm, t0, t1 = got
                if self._wants_onset:
                    self.on_utterance(normalise(pcm), t0, t1, onset=got.onset)
                else:
                    self.on_utterance(normalise(pcm), t0, t1)

    def _live(self, x):
        """The stream's level: the gain from the sentence's start, turned down where a peak would pass 0.9."""
        peak = float(np.abs(x).max()) if len(x) else 0.0
        g = min(self._live_gain, 0.9 / peak) if peak > 0 else self._live_gain
        return (x * g).astype(np.float32)

    def _tell(self, what, *a):
        try:
            getattr(self.listener, what)(*a)
        except Exception as e:  # noqa: BLE001 - the live STT must never stop the ears
            log.warning("ears: live listener %s failed: %s", what, e)

    def dump(self, path):
        """Debug only (ARCBY_EARS_DEBUG=1 - a test recording Selim asked for): the kept seconds -> WAV (high-passed,
        16 kHz) + a .txt of the gain-applied level and VAD p per chunk. Never part of the dataset."""
        items = list(self.debug or ())
        if not items:
            return 0
        x = np.concatenate([c for c, _, _ in items])
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(x * 64, -1, 1) * 32767).astype("<i2").tobytes())
        with open(str(path) + ".txt", "w") as f:
            for c, g, p in items:
                f.write(f"{20 * np.log10(np.sqrt((c ** 2).mean()) + 1e-12):.1f} {20 * np.log10(np.sqrt((g ** 2).mean()) + 1e-12):.1f} {p:.3f}\n")
        return len(x) / 16000

    @property
    def pause_s(self):
        """For the listening nods: None when no sentence is being heard (or the ears are muted), else the seconds
        of quiet inside it so far (0.0 while sound comes in)."""
        if self._clock() < self._unmute_at or self.seg.buf is None:
            return None
        return self.seg.quiet * CHUNK_S

    def status(self, since_chunks=0):
        """For the status log: 'N heard, gain xG, level -60 dBFS, VAD max 0.12' (VAD max reset by reading)."""
        out = (f"{self.heard} heard, gain x{self.agc.gain:.0f}, level {20 * np.log10(self.level + 1e-12):.0f} dBFS, "
               f"VAD max {self.vad_max:.2f}" + (f", {self.restarts} mic restarts" if self.restarts else ""))
        self.vad_max = 0.0
        return out

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
