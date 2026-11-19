"""
Voices: who is speaking, from the sound alone (speaker recognition) - local, nothing leaves the robot.

    emb = load_voice_embedder()             # ONNX speaker model (faces.MODELS, kind "voice"), or None
    vec = emb.embed(pcm16k)                 # unit vector of one utterance (>= ~1 s of speech is reliable)
    voices = VoiceStore(config.PEOPLE_DIR)  # voice prints of the people the robot knows, next to their faces
    pid, sim = voices.identify(vec)         # (person id, similarity) or (None, best similarity)
    voices.learn(pid, vec)                  # when the face says who is speaking (looking at the robot)
    voices.forget(pid)                      # people_cli.py forget (the robot never forgets anyone by itself)

The robot learns a voice only from utterances whose speaker it SAW the whole sentence (exactly one known person
looking at it - robot/conversation.py) and whose sound is not someone else's, never from guesses. Prints are kept in
PEOPLE_DIR/voices.npz, each with the time it was learnt ("<pid>__t": a bad print can be found and removed).
While only one person has prints, a voice is theirs only at VOICE_SOLO_SIM (no second person to compare with).
No audio is stored.
"""
import logging
import threading
import time
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

RATE = 16000
TIMES = "__t"                         # voices.npz key suffix: when each print was learnt


def mel_banks(n_mels=80, n_fft=512, rate=RATE, low=20.0, high=None):
    """Kaldi's triangular mel filters on the FFT bins: (n_mels, n_fft // 2 + 1)."""
    high = high or rate / 2

    def mel(f):
        return 1127.0 * np.log1p(np.asarray(f, np.float64) / 700.0)
    edges = np.linspace(mel(low), mel(high), n_mels + 2)
    bins = mel(np.arange(n_fft // 2 + 1) * rate / n_fft)
    left, centre, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    up = (bins - left) / (centre - left)
    down = (right - bins) / (right - centre)
    return np.maximum(0.0, np.minimum(up, down)).astype(np.float32)


_BANKS = mel_banks()
_POVEY = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85


def fbank(pcm, n_mels=80):
    """Kaldi-style log mel filterbank (25 ms frames every 10 ms, povey window, pre-emphasis 0.97, no dither),
    mean-normalised over the utterance - the input the speaker models were trained on. (frames, n_mels)."""
    feats = fbank_raw(pcm, n_mels)
    return feats - feats.mean(axis=0)


def fbank_raw(pcm, n_mels=80):
    """fbank() without the mean normalisation (samples at int16 scale)."""
    x = np.asarray(pcm, np.float64) * 32768.0
    if len(x) < 400:
        x = np.pad(x, (0, 400 - len(x)))
    n = 1 + (len(x) - 400) // 160
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160][:n].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] *= 1 - 0.97
    power = np.abs(np.fft.rfft(frames * _POVEY, 512)) ** 2
    return np.log(np.maximum(power @ _BANKS.T.astype(np.float64), np.finfo(np.float32).eps)).astype(np.float32)


class VoiceEmbedder:
    def __init__(self, session, name=""):
        self.session, self.name = session, name
        self.input = session.get_inputs()[0].name

    def embed(self, pcm):
        v = self.session.run(None, {self.input: fbank(pcm)[None]})[0].reshape(-1).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def load_voice_embedder(name=config.VOICE_MODEL, models_dir=None):
    """The speaker model, or None (not downloaded / no onnxruntime): the robot then just does not know voices."""
    from .faces import _session, model_path
    path = model_path(name, models_dir)
    if not Path(path).exists():
        log.warning("voices: no %s model (python3 code/get_models.py)", name)
        return None
    try:
        return VoiceEmbedder(_session(path, 1), name)   # one core: ~2 s of speech beside the face models
    except Exception as e:  # noqa: BLE001 - no voices is not a reason to stop the robot
        log.warning("voices: %s could not be loaded (%s)", name, e)
        return None


class VoiceStore:
    def __init__(self, folder, keep=config.VOICE_KEEP, match_sim=config.VOICE_MATCH_SIM,
                 margin=config.VOICE_MARGIN, alike=config.VOICE_ALIKE_SIM, solo_sim=config.VOICE_SOLO_SIM,
                 clock=time.time):
        self.path = Path(folder).expanduser() / "voices.npz"
        self.keep, self.match_sim, self.margin, self.alike = keep, match_sim, margin, alike
        self.solo_sim, self._clock = solo_sim, clock
        self._lock = threading.Lock()
        self.prints = {}
        self.times = {}                     # person -> when each print was learnt (0 = before times were kept)
        try:
            with np.load(self.path) as z:
                for k in z.files:
                    if k.endswith(TIMES):
                        self.times[k[:-len(TIMES)]] = z[k]
                    else:
                        self.prints[k] = z[k]
        except (OSError, ValueError):
            pass
        for pid, p in self.prints.items():
            t = self.times.get(pid)
            if t is None or len(t) != len(p):
                self.times[pid] = np.zeros(len(p))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name("voices.tmp.npz")
        np.savez(tmp, **self.prints, **{pid + TIMES: t for pid, t in self.times.items() if pid in self.prints})
        tmp.replace(self.path)

    def scores(self, vec):
        """{person id: similarity} - the mean of its 3 closest prints (one odd recording does not decide)."""
        with self._lock:
            return {pid: float(np.mean(np.sort(p @ vec)[-3:])) for pid, p in self.prints.items() if len(p)}

    def identify(self, vec, among=None):
        """(person id, sim) when one person's voice clearly fits; else (None, best sim). among = only these."""
        s = {k: v for k, v in self.scores(vec).items() if among is None or k in among}
        if not s:
            return None, 0.0
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        pid, best = ranked[0]
        if len(ranked) == 1:                # nobody to compare with: the margin says nothing - be stricter
            return (pid, best) if best >= max(self.match_sim, self.solo_sim) else (None, best)
        if best >= self.match_sim and best - ranked[1][1] >= self.margin:
            return pid, best
        return None, best

    def learn(self, pid, vec):
        """One more print for pid (the newest `keep`; one too like a kept print adds nothing). True if kept."""
        with self._lock:
            p = self.prints.get(pid)
            if p is not None and len(p) and float((p @ vec).max()) >= self.alike:
                return False
            t = self.times.get(pid, np.zeros(0)) if p is not None else np.zeros(0)
            p = vec[None] if p is None else np.vstack([p, vec[None]])[-self.keep:]
            self.prints[pid] = p.astype(np.float32)
            self.times[pid] = np.append(t, self._clock())[-self.keep:]
            self._save()
        return True

    def consistency(self, pid):
        """How alike pid's prints are: for each print the mean similarity to the others (low = an odd one out:
        another speaker, noise). Empty when fewer than 2 prints."""
        with self._lock:
            p = self.prints.get(pid)
            if p is None or len(p) < 2:
                return np.zeros(0)
            s = p @ p.T
            return (s.sum(1) - np.diag(s)) / (len(p) - 1)

    def count(self, pid):
        with self._lock:
            return len(self.prints.get(pid, ()))

    def forget(self, pid):
        with self._lock:
            self.times.pop(pid, None)
            if self.prints.pop(pid, None) is not None:
                self._save()

    def merge(self, keep, drop):
        """Two ids found to be one person (people_cli merge): their prints together."""
        with self._lock:
            if drop in self.prints:
                dim = self.prints[drop].shape[1]
                a = self.prints.get(keep, np.zeros((0, dim), np.float32))
                b = self.prints.pop(drop)
                ta = self.times.pop(keep, np.zeros(len(a)))
                tb = self.times.pop(drop, np.zeros(len(b)))
                self.prints[keep] = np.vstack([a, b])[-self.keep:]
                self.times[keep] = np.concatenate([ta, tb])[-self.keep:]
                self._save()
