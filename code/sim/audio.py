"""
Sound for the scenario simulator: speech-like voices, the robot's echo as the mic hears it, the mic's own floor, and
the real barge-in recording of 2026-11-01 (local only - family voice data, never in git).

Levels (measured on the robot, docs/ears.md "Barge-in"): the mic floor after the high-pass ~-78 dBFS; a person 1-2 m
away -35..-40 dBFS; the robot's voice plays at ~-18 dBFS and reaches the mic through config.BARGE_COUPLING_DB per band
(~-35 dBFS overall, as loud as a person), BARGE_LAG_S later, with a short room tail.

Everything is made at 48 kHz (what arecord gives the ears) and deterministic for a given seed.
"""
import json
import os
import wave
import zlib
from pathlib import Path

import numpy as np

from robot import config
from robot.bargein import EDGES

RATE = 48000
FLOOR_DBFS = -78.0           # the mic after robot/mic.py HighPass, nobody speaking (recording 2026-11-01)
MIC_DC = -0.026              # the raw I2S mic sits there (removed by the ears' high-pass)
ROBOT_PLAY_DBFS = -18.0      # what the speaker plays (ref*.wav of the recording: -15..-22 dBFS per second)
RECORDING_DIRS = (os.environ.get("ARCBY_SIM_RECORDING", ""), "~/.local/share/arcby/sim/barge-2026-11-01")

# Spectral shape per robot/bargein.py band (100-250-500-800-1200-1800-2600-3600-5000-7500 Hz), dB per Hz relative:
# a person near the mic = long-term speech spectrum (the real talk-over of 2026-11-01 stood 20-30 dB over the robot's
# echo at 500-1200 Hz: much more mid than the TTS); a phone speaker has no lows; the robot's TTS (Charon) is
# bass-heavy - measured on ref0-3.wav: 0 -1..-8 -7..-18 -15..-27 -15..-29 -24..-32 -29..-37 -31..-45 -29..-43.
SPEECH = (-2, 0, -2, -6, -10, -14, -17, -20, -25)
PHONE = (-30, -14, -4, 0, -3, -7, -11, -15, -24)
NOISE = (-6, -3, 0, 0, -2, -4, -6, -8, -12)
TTS = (0, -4, -10, -17, -18, -26, -32, -36, -36)
# who sounds how: pitch (Hz, 0 = no voice: a cough), level at the mic (dBFS over the voiced parts), shape
VOICES = {
    "Selim": dict(pitch=115, level=-35.0, shape=SPEECH),
    "Emre": dict(pitch=125, level=-37.0, shape=SPEECH),
    "Meltem": dict(pitch=210, level=-38.0, shape=SPEECH),
    "Halil": dict(pitch=105, level=-38.0, shape=SPEECH),
    "TikTok": dict(pitch=180, level=-38.0, shape=PHONE),
    "cough": dict(pitch=0, level=-33.0, shape=NOISE),
    "robot": dict(pitch=95, level=ROBOT_PLAY_DBFS, shape=TTS),     # what the speaker plays
}


def dbfs(x):
    x = np.asarray(x, np.float64)
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12) if len(x) else -240.0


def _profile(x, shape):
    """x with the band densities of `shape` (dB per Hz per robot/bargein.py band), smooth inside each band (two
    formant-like bumps), -30 dB below 100 Hz, -40 dB above 7.5 kHz."""
    s = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    p = np.abs(s) ** 2
    g = np.full(len(f), 1e-3)
    g[f < EDGES[0]] = 10 ** (-30 / 20)
    for (a, b), d in zip(zip(EDGES, EDGES[1:]), shape):
        m = (f >= a) & (f < b)
        if m.any():
            g[m] = 10 ** (d / 20) / np.sqrt(p[m].mean() + 1e-20)
    bumps = 1 + 0.6 * np.exp(-((f - 650) / 200) ** 2) + 0.4 * np.exp(-((f - 1600) / 400) ** 2)
    return np.fft.irfft(s * g * bumps, len(x)).astype(np.float32)


def _shape(x, gains_of_f):
    s = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    return np.fft.irfft(s * gains_of_f(f), len(x)).astype(np.float32)


VOWELS = set("aeıioöuüâîûAEIİOÖUÜ")


def timeline(text, rng, seconds=None):
    """Syllables (start, length) and words (centre, word) for a phrase, as people and Turkish TTS say it: ~6
    syllables a second, short gaps between words, a pause at a comma (~0.25 s) and after a sentence (~0.4 s).
    With `seconds` and no words (a cough, a mumble): syllables over that long."""
    syl, words, pos = [], [], 0.03
    tokens = [w for w in (text or "").split() if any(c.isalpha() for c in w)]
    if not tokens:
        while pos < (seconds or 0.6) - 0.08:
            n = rng.uniform(0.11, 0.21)
            syl.append((pos, n))
            pos += n + rng.uniform(0.0, 0.04)
        return syl, [], max(pos + 0.05, seconds or 0.0)
    for tok in tokens:
        start = pos
        for _ in range(max(1, sum(c in VOWELS for c in tok))):
            n = rng.uniform(0.13, 0.19)
            syl.append((pos, n))
            pos += n + rng.uniform(0.0, 0.015)
        words.append(((start + pos) / 2, tok))
        pos += (rng.uniform(0.35, 0.45) if tok[-1] in ".?!" else rng.uniform(0.22, 0.3) if tok[-1] in ",;:"
                else rng.uniform(0.03, 0.09))
    return syl, words, pos + 0.05


def voice(text, who="Selim", seed=0, level=None, seconds=None):
    """A speech-like sound for `text`: a glottal-pulse buzz (pitch with a slow wander) plus breath noise, through
    three formant-like peaks, cut into the phrase's syllables, words and pauses; `level` dBFS over the voiced parts.
    Returns (48 kHz float, [(word centre s, word)])."""
    v = VOICES.get(who, VOICES["Selim"])
    rng = np.random.default_rng(zlib.crc32(f"{who}/{seed}/{text}".encode()))
    syl, words, seconds = timeline(text, rng, seconds)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    if v["pitch"]:
        f0 = v["pitch"] * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
        phase = np.cumsum(f0) / RATE
        src = (2 * (phase % 1.0) - 1) * 0.8 + rng.normal(0, 0.25, n)
    else:
        src = rng.normal(0, 1, n)
    x = _profile(src, v["shape"])
    env = np.zeros(n, np.float32)
    for a, d in syl:
        i, j = int(a * RATE), min(n, int((a + d) * RATE))
        if j > i:
            env[i:j] = rng.uniform(0.55, 1.0) * np.sin(np.linspace(0, np.pi, j - i)) ** 0.7
    x = x * env
    voiced = x[env > 0.3]
    target = v["level"] if level is None else level
    if len(voiced):
        x = x * 10 ** ((target - dbfs(voiced)) / 20)
    return x.astype(np.float32), words


def couple(played, seed=0, spread_db=2.0):
    """What the mic hears of what the speaker plays: config.BARGE_COUPLING_DB per band (+- spread_db, a little
    different from what the detector starts with, as on a real day), a soft speaker distortion and a short room tail
    fading BARGE_DECAY_DB per 32 ms. Delay not included (the world places it BARGE_LAG_S later)."""
    rng = np.random.default_rng(seed + 991)
    coupling = np.array(config.BARGE_COUPLING_DB) + rng.uniform(-spread_db, spread_db, len(config.BARGE_COUPLING_DB))
    edges = np.array(EDGES, float)

    def gains(f):
        g = np.full(len(f), 10 ** (-45 / 20))
        for (a, b), c in zip(zip(edges, edges[1:]), coupling):
            g[(f >= a) & (f < b)] = 10 ** (c / 20)
        g[f >= edges[-1]] = 10 ** (-40 / 20)
        return g
    y = _shape(np.tanh(played * 3) / 3, gains)         # the small speaker squashes peaks a little
    tail_n = int(0.25 * RATE)
    decay = 10 ** (-config.BARGE_DECAY_DB / 20 / (0.032 * RATE))
    h = rng.normal(0, 1, tail_n) * decay ** np.arange(tail_n)
    h *= 0.35 / np.sqrt(np.sum(h ** 2))
    h[0] += 1.0
    out = np.convolve(y, h)[:len(y) + tail_n // 2]
    return out.astype(np.float32)


def mic_floor(seconds, seed=0):
    """The raw mic with nobody speaking: its DC offset, a slow drift and the -78 dBFS floor."""
    rng = np.random.default_rng(seed + 7)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    floor = rng.normal(0, 10 ** (FLOOR_DBFS / 20), n)
    return (MIC_DC + 0.002 * np.sin(2 * np.pi * 0.4 * t) + floor).astype(np.float32)


def frames(x):
    """48 kHz mono float -> what speakerd plays (48 kHz stereo 16-bit bytes; robot/bargein.py EchoRef.add)."""
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    return np.repeat(y, 2).tobytes()


def raw_chunk(x):
    """48 kHz mono float -> arecord's S32_LE stereo bytes (the ears read channel 0)."""
    y = (np.clip(x, -1, 1 - 2 ** -31) * 2 ** 31).astype("<i4")
    return np.stack([y, np.zeros_like(y)], 1).tobytes()


def wav48(path):
    """A 48 kHz stereo 16-bit WAV -> (bytes as played, mono float)."""
    with wave.open(str(path)) as w:
        data = w.readframes(w.getnframes())
    x = np.frombuffer(data, "<i2")[::2].astype(np.float32) / 32768
    return data, x


class Recording:
    """The barge-in recording of 2026-11-01 (robot mic during 4 robot sentences; Selim talks over the 4th from
    44.0 s; a knock at ~18.6 s during the 1st). mic = raw 48 kHz mono float; phrases = [(start s, frames, seconds)]
    with start = speakerd's playing time in mic seconds (log16.json "k"); the echo reaches the mic BARGE_LAG_S later."""

    TALK_OVER_S = 44.0
    KNOCK_S = 18.6

    def __init__(self, folder):
        folder = Path(folder).expanduser()
        self.mic = np.load(folder / "mic48k.npy").astype(np.float32)
        log = json.loads((folder / "log16.json").read_text())
        self.phrases = []
        for p in log["phrases"]:
            data, x = wav48(folder / f"ref{p['i']}.wav")
            self.phrases.append((p["k"] / 16000, data, len(x) / RATE))
        self.seconds = len(self.mic) / RATE

    @classmethod
    def find(cls):
        """The recording, or None (with the folders looked in) when it is not on this machine."""
        for d in RECORDING_DIRS:
            if d and (Path(d).expanduser() / "mic48k.npy").exists():
                return cls(d)
        return None
