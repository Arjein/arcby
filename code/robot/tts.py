"""
Voice: the robot speaks Turkish and English with one Gemini TTS voice (config.TTS_VOICE), without ever
holding up the rest of the robot.

    voice = Voice()                          # GEMINI_API_KEY from config.SECRETS_FILE (or the environment)
    voice.say("Merhaba Selim!")               # queued: returns at once, the voice thread speaks
    voice.say_now("Good evening.")           # blocking (scripts, tests); returns "cache" / "gemini" / "silent"
    voice.prepare(["Merhaba Selim", ...])     # make phrases ahead of time (cached), nothing is played
    with voice.hold():                       # a reply still being written (the mind streams it): busy all along,
        voice.say("Sentence one.", mark={})  # sentence 2 is made while 1 plays; mark gets "made" / "play" times
        voice.say("Sentence two.")

Every phrase is kept as a WAV in TTS_CACHE_DIR/<voice>/ (key = model + voice + style + text): it is paid for
once and plays at once ever after. At most TTS_DAILY_REQUESTS Gemini calls a day (TTS_CACHE_DIR/usage.json,
days in TTS_TIMEZONE). When TTS_MODEL is refused (Google's own daily quota), the same voice comes from
TTS_FALLBACK_MODELS (without the style prompt); a phrase cached by any of them plays from the cache. With no model
left: silent (espeak-ng only if TTS_ESPEAK - off: Selim does not want it).
The sound goes to robot/speakerd.py, a process of its own that keeps the card running for good (no amplifier pops).
Loudness = the sample peak (the I2S card has no volume control): TTS_LEVEL_DAY, TTS_LEVEL_NIGHT at night.
Two threads: "voice" makes the sound of the next phrase (cache / Gemini) while "speaker" plays the one before, in
order - a reply of two sentences waits for one Gemini call, not two. 3.x models stream (gemini_stream): a phrase
starts playing with its first piece (~2 s) while the rest is still being made. Vocal tags (<chuckle>, <sigh>, ...,
AUDIO_TAGS) are passed on when TTS_AUDIO_TAGS; any other bracketed text is never read out.
Only the text to speak goes to Google - never pictures or sound.
"""
import collections
import contextlib
import atexit
import base64
import hashlib
import itertools
import io
import json
import logging
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from . import config

log = logging.getLogger(__name__)

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OUT_RATE = 48000                    # the I2S card's rate (stereo)
# Turkish letters (İ lower-cases to i + U+0307) or common words. No re.I: it makes "İ" match a plain "i"
TURKISH = re.compile("[çğıöşü\u0307]|\\b(ve|bir|ben|sen|merhaba|günaydın|nasılsın|evet|hayır|tamam|iyi)\\b")


_COUNT_LOCK = threading.Lock()


@contextlib.contextmanager
def count_lock(path):
    """Hold while reading + writing a daily request count file: alive.py and say.py (two processes) or two threads
    count into the same file - without it one count could overwrite the other's (review 2026-11-01)."""
    with _COUNT_LOCK:
        lock = Path(str(path) + ".lock")
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            f = open(lock, "a")
        except OSError:                     # no lock file possible: count without it (as before)
            yield
            return
        try:
            try:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            yield
        finally:
            f.close()                       # closing releases the flock


def read_secret(name, path=config.SECRETS_FILE):
    """A KEY=value line from the secrets file (the environment wins). Never logged."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in Path(path).expanduser().read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() == name:
                return v.strip().strip("\"'") or None
    except OSError:
        pass
    return None


# Gemini-TTS markup tags the robot may use (the mind's prompt names them; robot/conversation.py's fillers too)
# Gemini 3.8 TTS inline vocal tags (ai.google.dev speech-generation docs, 2026-11-01): <chuckle>, <sigh>... in the
# transcript, in English even inside Turkish. Older names map onto them; the 2.5 model gets none (it reads them).
AUDIO_TAGS = ("argh", "breath", "heavy breath", "exhales", "cackle", "cheer", "chuckle", "chuckles", "cough", "cry",
              "gasp", "giggle", "groan", "growl", "grunt", "grr", "hiss", "laugh", "laughter", "moan", "pant", "pff",
              "phew", "scream", "shout", "shriek", "sigh", "sighs", "sneeze", "snicker", "snort", "sob",
              "throat-clearing", "tsk", "whimper", "whispers", "whispering", "yawn", "short pause", "long pause")
OLD_TAGS = {"laughing": "laugh", "medium pause": "short pause", "uhm": "breath", "um": "breath"}
_BRACKETS = re.compile(r"[\[<]([^\[\]<>]*)[\]>]")


def audio_tags(text, keep=None):
    """Allowed audio tags stay as <tag> (lower-case; "[sigh]" / "[Laughing]" -> "<sigh>" / "<laugh>"), any other
    bracketed text goes - nothing odd is read aloud. keep=False (TTS_AUDIO_TAGS off) strips them all."""
    keep = config.TTS_AUDIO_TAGS if keep is None else keep

    def one(m):
        tag = " ".join(m.group(1).lower().split())
        tag = OLD_TAGS.get(tag, tag)
        return f" <{tag}> " if keep and tag in AUDIO_TAGS else " "
    text = _BRACKETS.sub(one, text)
    return re.sub(r"\s+([,.!?…])", r"\1", " ".join(text.split()))


def has_words(text):
    """Anything to say besides tags?"""
    return any(c.isalpha() for c in _BRACKETS.sub("", text))


def interactions_model(model):
    """The 3.x TTS models: the Interactions API (style as an annotation, inline <tags>, streaming). 2.5: the old
    generateContent call with the style in the prompt. "cloud:..." = Google Cloud TTS (cloud_pcm)."""
    return model.startswith("gemini-") and not model.startswith("gemini-2.")


CLOUD_TTS = "https://texttospeech.googleapis.com/v1/text:synthesize"
_cloud_token = None


def cloud_pcm(text, model, timeout=config.TTS_TIMEOUT_S, token=None, post=None):
    """Google Cloud Text-to-Speech, Chirp 3 HD (model "cloud:Charon" -> tr-TR-Chirp3-HD-Charon, en-US for English
    text): the whole phrase ~0.8 s (Pi, 2026-11-01). A service-account login (robot/gcloud.py), no tags."""
    global _cloud_token
    if token is None:
        if _cloud_token is None:
            from .gcloud import AccessToken
            _cloud_token = AccessToken()
        token = _cloud_token.get()
        if not token:
            raise Refused(403, f"no Cloud login ({_cloud_token.error})")
    name = model.split(":", 1)[1]
    lang = "en-US" if guess_lang(text) == "en" else "tr-TR"
    body = {"input": {"text": audio_tags(text, keep=False)},
            "voice": {"languageCode": lang, "name": f"{lang}-Chirp3-HD-{name}"},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}}
    req = urllib.request.Request(CLOUD_TTS, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    raw = base64.b64decode((post or _post_json)(req, timeout).get("audioContent", ""))
    if not raw:
        raise RuntimeError("no audio (empty Cloud TTS answer)")
    pcm, rate = read_wav(io.BytesIO(raw)) if raw[:4] == b"RIFF" else \
        (np.frombuffer(raw[:len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768, 24000)
    return pcm, rate


def spoken(text, say_as=None, tags=None):
    """How the text is said: 'Arcby' -> 'Arkbi'; audio tags checked (audio_tags)."""
    for a, b in (config.TTS_SAY_AS if say_as is None else say_as).items():
        text = re.sub(rf"\b{re.escape(a)}\b", b, text)
    return audio_tags(text, tags)


ENGLISH = re.compile(r"\b(the|you|your|what|is|are|how|hello|hi|can|do|does|my|me|it|this|that|please|thanks|"
                     r"thank|yes|no|who|where|when|why|i'm|i|good|morning|evening|night|let|see|one|moment)\b")


def guess_lang(text):
    """'en' only for text that looks English: "Arkbi bana kendini anlat" has no Turkish letters but is Turkish (it
    got an English filler, 2026-11-01). The robot speaks Turkish unless spoken to in English."""
    t = text.lower()
    return "en" if ENGLISH.search(t) and not TURKISH.search(t) else "tr"


class Refused(RuntimeError):
    """Gemini said no: code = HTTP status, retry_s = how long it asked us to wait (None = not said)."""

    def __init__(self, code, message, retry_s=None, per_day=False):
        super().__init__(f"HTTP {code}: {message}")
        self.code, self.retry_s, self.per_day = code, retry_s, per_day


def _post_json(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:            # the key is in a header, so not in this message
        raise _refused(e) from None


def _refused(e):
    """An HTTPError from Gemini -> Refused (status, message, retry delay, per-day quota or not)."""
    try:
        err = json.loads(e.read() or b"{}").get("error", {})
    except ValueError:
        err = {}
    delays = [d["retryDelay"] for d in err.get("details", []) if "retryDelay" in d]
    retry = float(delays[0].rstrip("s")) if delays and re.fullmatch(r"[\d.]+s", delays[0]) else None
    message = err.get("message", "")
    quota_ids = [v.get("quotaId", "") for d in err.get("details", []) for v in d.get("violations", [])]
    per_day = any("PerDay" in q for q in quota_ids) or bool(re.search(r"per ?day|daily", message, re.I))
    return Refused(e.code, message[:200], retry, per_day)


INTERACTIONS = "https://generativelanguage.googleapis.com/v1beta/interactions"
STREAM_RATE = 24000


def _sse(r):
    """Server-sent events from a response: each "data:" line as JSON (anything else skipped)."""
    for line in r:
        line = line.decode(errors="replace").strip()
        if line.startswith("data:"):
            try:
                yield json.loads(line[5:])
            except ValueError:
                continue


def gemini_stream(text, key, model=config.TTS_MODEL, voice=config.TTS_VOICE, style=config.TTS_STYLE,
                  timeout=config.TTS_TIMEOUT_S, urlopen=urllib.request.urlopen):
    """A 3.x TTS model, streamed (Interactions API): yields float32 samples at STREAM_RATE as they are made - the
    first ~2 s after the call, the rest faster than it plays (measured on the Pi 2026-11-01). The style is an
    annotation (never read out), <tags> stay in the text. Raises Refused / RuntimeError before the first chunk
    when nothing comes."""
    content = {"type": "text", "text": text}
    if style:
        content["annotations"] = [{"type": "speech_metadata", "style": style}]
    body = {"model": model, "input": [{"type": "user_input", "content": [content]}], "stream": True,
            "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": STREAM_RATE},
            "generation_config": {"speech_config": [{"voice": voice}]}}
    req = urllib.request.Request(INTERACTIONS + "?alt=sse", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        r = urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise _refused(e) from None
    got, odd = False, b""
    with r:
        for ev in _sse(r):
            if "error" in ev:
                err = ev["error"] if isinstance(ev["error"], dict) else {"message": str(ev["error"])}
                raise Refused(int(err.get("code", 500) or 500), str(err.get("message", ""))[:200])
            d = ev.get("delta") or {}
            if d.get("type") != "audio" or not d.get("data"):
                continue
            raw = odd + base64.b64decode(d["data"])
            odd = raw[len(raw) // 2 * 2:]
            raw = raw[:len(raw) // 2 * 2]
            if raw:
                got = True
                yield np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    if not got:
        raise RuntimeError("no audio (empty stream)")


def gemini_audio(text, key, model=config.TTS_MODEL, voice=config.TTS_VOICE, style=config.TTS_STYLE,
                 timeout=config.TTS_TIMEOUT_S):
    """The whole phrase -> (samples, rate): 3.x models through the stream, 2.5 through gemini_pcm (no tags)."""
    if model.startswith("cloud:"):
        return cloud_pcm(text, model, timeout)
    if interactions_model(model):
        return np.concatenate(list(gemini_stream(text, key, model, voice, style, timeout))), STREAM_RATE
    return gemini_pcm(audio_tags(text, keep=False), key, model, voice, style, timeout)


def gemini_pcm(text, key, model=config.TTS_MODEL, voice=config.TTS_VOICE, style=config.TTS_STYLE,
               timeout=config.TTS_TIMEOUT_S, post=_post_json):
    """One Gemini TTS call -> (mono float32 samples -1..1, sample rate). Raises on any failure."""
    # the words to say in quotes: short phrases ("Bana bak.") came back empty (finishReason OTHER) about half
    # the time without them, 0 of 2 with them (2026-10-31)
    body = {"contents": [{"parts": [{"text": f'{style}: "{text}"' if style else text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    cand = (post(req, timeout).get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    if not parts or "inlineData" not in parts[0]:                # finishReason OTHER: happens now and then
        raise RuntimeError(f"no audio (finishReason {cand.get('finishReason')})")
    part = parts[0]["inlineData"]
    mime = part.get("mimeType", "")
    raw = base64.b64decode(part["data"])
    if raw[:4] == b"RIFF" or "wav" in mime.lower():
        # gemini-3.8-flash-tts answers with a whole WAV file (header + a 6 KB C2PA signature chunk at the end):
        # played as raw samples, that was a loud noise burst after every phrase (2026-11-01)
        pcm, rate = read_wav(io.BytesIO(raw))
    else:
        found = re.search(r"rate=(\d+)", mime)
        pcm = np.frombuffer(raw[:len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768
        rate = int(found.group(1)) if found else 24000
    if not len(pcm):
        raise RuntimeError("empty audio")
    return pcm, rate


def espeak_pcm(text, lang):
    """Offline robotic voice: (samples, rate)."""
    wav = subprocess.run(["espeak-ng", "-v", "tr" if lang == "tr" else "en-gb", "-s", "145", "--stdout", text],
                         capture_output=True, check=True, timeout=15).stdout
    return read_wav(io.BytesIO(wav))


def read_wav(f):
    w = wave.open(f)
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    if w.getnchannels() == 2:
        x = x[::2]
    return x, w.getframerate()


# "RIFF" and "WAVE" as little-endian int16 samples: a WAV file inside the samples (see unwrap_wav)
_RIFF = np.frombuffer(b"RIFF", "<i2").astype(int)
_WAVE = np.frombuffer(b"WAVE", "<i2").astype(int)


def unwrap_wav(pcm):
    """Samples that are really a whole WAV file (cached from the WAV answers before 2026-11-01, rescaled by
    32767/32768 on the way: each value can be 1 off) -> only its audio. Anything else comes back as it is."""
    if len(pcm) < 22:
        return pcm
    v = np.round(pcm[:22].astype(np.float64) * 32768).astype(int)
    if not (np.abs(v[0:2] - _RIFF).max() <= 1 and np.abs(v[4:6] - _WAVE).max() <= 1):
        return pcm
    # the "data" chunk size (bytes 40..43). Rescaling truncated each value toward zero, so either half may be 1
    # too small (the high half 1 -> 0 for a phrase over 1.4 s): the size that fits the file is the right one (the
    # only thing after the audio is a 6 KB signature chunk, never another 64 KB)
    lo, hi = int(v[20]), int(v[21]) & 0xFFFF
    room = len(pcm) - 22
    sizes = [(((lo + dl) & 0xFFFF) | (hi + dh) << 16) // 2 for dl in (-1, 0, 1) for dh in (0, 1)]
    fits = [n for n in sizes if 0 < n <= room]
    body = pcm[22:22 + max(fits)] if fits else pcm[22:]
    return body * (32768 / 32767)


def write_wav(path, pcm, rate):
    tmp = f"{path}.tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    os.replace(tmp, path)


def to_card(pcm, rate, level):
    """Samples -> WAV bytes for the I2S card: 48 kHz stereo, peak = level (at most TTS_LEVEL_MAX: protects the
    speaker), 5 ms fade in / out (no click)."""
    level = min(float(level), config.TTS_LEVEL_MAX)
    x = pcm - pcm.mean()
    x = x / (np.abs(x).max() or 1.0) * level
    n = max(1, int(len(x) * OUT_RATE / rate))
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    fade = min(len(y) // 2, int(0.005 * OUT_RATE))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(OUT_RATE)
        w.writeframes((np.repeat(y, 2) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class CardStream:
    """A phrase that arrives in pieces -> 48 kHz stereo frames for the card, piece by piece: one fixed gain (the
    peak is not known yet: Gemini's speech peaks at 0.8-0.9, TTS_STREAM_PEAK), resampled without a seam between
    pieces, a 5 ms fade in at the start and out at the end (finish()) - no click."""

    def __init__(self, rate, level, peak=None):
        peak = config.TTS_STREAM_PEAK if peak is None else peak
        level = min(float(level), config.TTS_LEVEL_MAX)        # protects the speaker (see to_card)
        self.step, self.gain = rate / OUT_RATE, level / peak
        self.carry = np.zeros(0, np.float32)      # the last input sample (interpolation goes on from it)
        self.t = 0.0                               # next output position, in input samples from carry's start
        self.fade = int(0.005 * OUT_RATE)
        self.started = False
        self.held = np.zeros(0)                    # the last fade-length of output, held back for the fade out

    def push(self, x):
        x = np.concatenate([self.carry, np.clip(np.asarray(x, np.float32) * self.gain, -1, 1)])
        if len(x) < 2:
            self.carry = x
            return b""
        pos = np.arange(self.t, len(x) - 1 + 1e-9, self.step)
        y = np.interp(pos, np.arange(len(x)), x)
        self.t = (pos[-1] + self.step) - (len(x) - 1) if len(pos) else self.t - (len(x) - 1)
        self.carry = x[-1:]
        if not self.started and len(y):
            n = min(self.fade, len(y))
            y[:n] *= np.linspace(0.0, 1.0, self.fade)[:n]
            self.started = True
        y = np.concatenate([self.held, y])
        self.held, y = y[-self.fade:], y[:-self.fade] if len(y) > self.fade else y[:0]
        return self._frames(y)

    def finish(self):
        y = self.held * np.linspace(1.0, 0.0, len(self.held)) if len(self.held) else self.held
        self.held = np.zeros(0)
        return self._frames(y)

    @staticmethod
    def _frames(y):
        return (np.repeat(np.asarray(y, np.float64), 2) * 32767).astype("<i2").tobytes()


class SpeakerError(RuntimeError):
    pass


SPEAKERD = str(Path(__file__).with_name("speakerd.py"))


class Speaker:
    """The client of robot/speakerd.py, the speaker's own process. That process keeps the sound card running for
    good with ALSA playing silence by itself: the voiceHAT amplifier switches on once, never off-on again, so no
    "pffzzrt" between phrases or when alive.py is busy (Selim, 2026-11-01). Nothing in this process feeds it.

    play(wav_bytes) blocks until the last sample has left the speaker (+ ECHO_S of room echo): the ears are muted
    exactly that long. If speakerd dies, waiting plays fail at once; open_speaker() starts a new one."""

    ECHO_S = 0.15                        # after the last sample: the room still rings this long (ears stay muted)
    START_S = 5.0                        # speakerd must say "ready" within this

    def __init__(self, card=0, popen=subprocess.Popen, sleep=time.sleep, log_path=None, clock=time.monotonic,
                 device=None):
        self.sleep, self.clock = sleep, clock
        self.device = device or f"hw:{card},0"
        self.error = None
        self.info = None
        self.alive = True
        self._ids = itertools.count(1)
        self._waiting = {}
        self._lock = threading.Lock()
        self._wlock = threading.Lock()          # one writer at a time on speakerd's stdin (play / stop threads)
        self.echo = None                        # robot/bargein.py EchoRef: what is played and when (barge-in)
        self._ready = threading.Event()
        err = open(log_path, "a") if log_path else subprocess.DEVNULL     # speakerd logs there
        try:
            self.proc = popen([sys.executable, SPEAKERD, "--device", self.device],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        finally:
            if log_path:
                err.close()
        self._reader = threading.Thread(target=self._read, name="speaker", daemon=True)
        self._reader.start()
        if not self._ready.wait(self.START_S) or not self.alive:
            self.close()
            raise SpeakerError(f"speaker {self.device}: {self.error or 'did not start'}")

    def _read(self):
        try:
            for raw in self.proc.stdout:
                words = raw.decode(errors="replace").split()
                if not words:
                    continue
                if words[0] == "ready":
                    self.info = " ".join(words[1:])
                    self._ready.set()
                elif words[0] == "done" and len(words) == 3:
                    with self._lock:
                        w = self._waiting.pop(int(words[1]), None)
                    if w is not None:
                        w["end"] = float(words[2])
                        w["event"].set()
                    if self.echo is not None:
                        self.echo.ended(int(words[1]), float(words[2]))
                elif words[0] == "error":
                    self.error = " ".join(words[1:])
                    log.warning("speaker: %s", self.error)
        except (OSError, ValueError):
            pass
        self._dead()

    def _dead(self):
        self.alive = False
        self._ready.set()
        with self._lock:
            waiting, self._waiting = self._waiting, {}
        for w in waiting.values():                 # nobody waits for a phrase that will never end
            w["event"].set()

    def play(self, wav_bytes):
        with wave.open(io.BytesIO(wav_bytes)) as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (OUT_RATE, 2, 2):
                raise ValueError("Speaker plays 48 kHz 16-bit stereo")
            frames = w.readframes(w.getnframes())
        self._wait(self._send(frames), len(frames))

    def play_stream(self, pieces):
        """48 kHz stereo frames in pieces (a phrase still being made): each goes to speakerd as it comes - they
        play back to back; a late piece = a moment of silence, the card never stops (no pop). Blocks until the
        last one has played."""
        last, total = None, 0
        for frames in pieces:
            if frames:
                last, total = self._send(frames), total + len(frames)
        if last is not None:
            self._wait(last, total)

    def _send(self, frames):
        if not self.alive:
            raise SpeakerError(f"the speaker has stopped ({self.error or 'no reason given'})")
        pid = next(self._ids)
        wait = {"event": threading.Event(), "end": None}
        with self._lock:
            self._waiting[pid] = wait
        if self.echo is not None:
            self.echo.add(frames, pid)
        try:
            with self._wlock:
                self.proc.stdin.write(struct.pack("<4sII", b"PLAY", pid, len(frames)) + frames)
                self.proc.stdin.flush()
        except (OSError, ValueError):
            self._dead()
            raise SpeakerError("the speaker has stopped (pipe closed)") from None
        return wait

    def stop(self):
        """Someone talks over the robot: what plays fades out (~0.2 s), everything sent so far is dropped."""
        if not self.alive:
            return
        try:
            with self._wlock:
                self.proc.stdin.write(struct.pack("<4sII", b"STOP", 0, 0))
                self.proc.stdin.flush()
        except (OSError, ValueError):
            return
        if self.echo is not None:
            self.echo.cut(self.clock() + 0.25)

    def _wait(self, wait, nbytes):
        if not wait["event"].wait(nbytes / (OUT_RATE * 4) + 5):
            raise SpeakerError("speaker: playing took far too long")
        if wait["end"] is None:
            raise SpeakerError(f"the speaker stopped while playing ({self.error or 'no reason given'})")
        left = wait["end"] + self.ECHO_S - self.clock()
        if left > 0:
            self.sleep(left)

    def close(self):
        """Let the queued sound finish, then stop speakerd (terminate only if it does not end by itself)."""
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._reader.join(timeout=1.0)
        if self.proc.stdout:
            try:
                self.proc.stdout.close()
            except OSError:
                pass
        self._dead()


_speaker = None
_speaker_log = None
_speaker_lock = threading.Lock()


def open_speaker(card=0, log_path=None):
    """The one Speaker kept open (started now if not yet: its single amplifier switch at start-up, not before the
    first sentence; started again if its process died). log_path: where speakerd logs (kept for restarts)."""
    global _speaker, _speaker_log
    with _speaker_lock:
        if log_path:
            _speaker_log = log_path
        if _speaker is None or not _speaker.alive:
            if _speaker is not None:
                _speaker.close()                   # reap the dead one
            _speaker = Speaker(card, log_path=_speaker_log)
            _speaker.echo = _echo
        return _speaker


_echo = None


def set_echo_ref(echo):
    """robot/bargein.py EchoRef for every Speaker (also one started again after its process died)."""
    global _echo
    _echo = echo
    if _speaker is not None:
        _speaker.echo = echo


@atexit.register
def close_speaker():
    global _speaker
    with _speaker_lock:
        if _speaker is not None:
            _speaker.close()
            _speaker = None


def stop_speaker():
    """Barge-in: stop the one Speaker (if it is open)."""
    sp = _speaker
    if sp is not None:
        sp.stop()


def play_stream(pieces, card=0):
    open_speaker(card).play_stream(pieces)


def play(wav_bytes, card=0):
    """Play on the I2S card through the one Speaker that stays open."""
    open_speaker(card).play(wav_bytes)


class Voice:
    def __init__(self, key=None, cache_dir=config.TTS_CACHE_DIR, model=config.TTS_MODEL, voice=config.TTS_VOICE,
                 style=config.TTS_STYLE, daily_requests=config.TTS_DAILY_REQUESTS, tz=config.TTS_TIMEZONE,
                 synth=gemini_audio, fallback=espeak_pcm, player=play, clock=time.time,
                 fallback_models=config.TTS_FALLBACK_MODELS, espeak=config.TTS_ESPEAK,
                 level_file=config.TTS_LEVEL_FILE, streamer="auto", stream_player=play_stream,
                 stopper=stop_speaker):
        self.key = read_secret("GEMINI_API_KEY") if key is None else key
        self.model, self.voice, self.style, self.daily = model, voice, style, daily_requests
        # each model with its own style (2.5: a prompt prefix; 3.x: an annotation)
        self.models = [(model, style)] + [(m, config.TTS_STYLES.get(m, style)) for m in fallback_models
                                          if m != model]
        # streamer(text, key, model, voice, style) -> sample pieces: a phrase plays while it is being made
        # (3.x models). "auto" = gemini_stream with the real Gemini synth, none with a test's fake one
        self.streamer = (gemini_stream if synth is gemini_audio else None) if streamer == "auto" else streamer
        self.stream_player = stream_player
        self.stopper = stopper
        self._gen = 0                       # interrupts so far: anything made / queued before the latest is dropped
        self.interrupted_at = None          # time.monotonic() of the last interrupt (logs)
        self._limit_said = -1e9             # when "my limit is full" was last said
        self.espeak = espeak
        self.level_file = level_file
        self.paused = {}                    # model -> refused until (its quota / no credit)
        self.root = Path(cache_dir).expanduser()
        self.dir = self.root / voice
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tz = ZoneInfo(tz)
        self.synth, self.fallback, self.player, self._clock = synth, fallback, player, clock
        self._lock = threading.Lock()
        self.spoken = collections.deque(maxlen=12)  # (time.monotonic(), text) said lately - also the greeter's: the
                                            # conversation knows its own words when it hears them back
        self._todo = queue.Queue()          # phrases to make (the "voice" thread)...
        self._ready = queue.Queue()         # ...and made, to play (the "speaker" thread), in the same order
        self._thread = self._player = None
        self._pending = 0                   # phrases said but not finished (queued, being made, playing)
        self._holds = 0                     # replies still being written (hold()): busy meanwhile
        self._pending_lock = threading.Lock()
        self._prepare_lock = threading.Lock()   # one prepare() at a time (the greeter's and the fillers)
        self._refusals = {}                 # model -> 429s in a row (no success between)
        self.speaking = False
        self.paused_until = 0.0             # Gemini said no credit / quota: do not ask again before this
        self._thread_lock = threading.Lock()
        self._cache_checked = 0.0

    # ---------- making sound ----------

    def cache_path(self, text, model=None, style=None):
        model = self.model if model is None else model
        style = self.style if style is None else style
        h = hashlib.sha1(f"{model}|{self.voice}|{style}|{text}".encode()).hexdigest()[:20]
        return self.dir / f"{h}.wav"

    def _cache_keys(self):
        """Every (model, style) a phrase may be cached under: today's models, and the 3.x ones without a style
        (they were fallbacks without one until 2026-11-01 - 20 phrases were made again, the day's limit hit)."""
        return self.models + [(m, "") for m, st in self.models if st and interactions_model(m)]

    def cached(self, text):
        """Is the phrase in the cache (it plays at once, no request)?"""
        text = spoken(text)
        return any(self.cache_path(text, m, st).exists() for m, st in self._cache_keys())

    def audio(self, text):
        """(samples, rate, where from: 'cache' / 'gemini' / 'espeak' / 'silent')."""
        text = spoken(text)
        if not has_words(text):                 # only tags (or nothing): never a request
            return np.zeros(1, np.float32), 16000, "silent"
        for model, style in self._cache_keys():  # made before by any of the models: free and at once
            path = self.cache_path(text, model, style)
            if path.exists():
                return (*self._cached(path), "cache")
        for model, style in self.models:
            if self._clock() < max(self.paused.get(model, 0.0), self.paused_until):
                continue
            got = self._make(text, model, style)
            if got is not None:
                return (*got, "gemini")
        if self.espeak:
            return (*self.fallback(text, guess_lang(text)), "espeak")
        log.warning("tts: no Gemini voice now - not said: %s", text)
        limit = self._limit_notice()
        if limit is not None:
            return limit
        return np.zeros(1, np.float32), 16000, "silent"

    def _limit_notice(self):
        """No voice left (the day's limit, Google's quota, no credit): say so - from the cache, at most every
        TTS_LIMIT_EVERY_S - so the family knows why it is silent (Selim 2026-11-01)."""
        now = time.monotonic()
        if now - self._limit_said < config.TTS_LIMIT_EVERY_S:
            return None
        said = spoken(config.TTS_LIMIT_PHRASE)
        for model, style in self._cache_keys():
            path = self.cache_path(said, model, style)
            if path.exists():
                self._limit_said = now
                return (*self._cached(path), "limit")
        return None

    def _cached(self, path):
        """A cached phrase. One that still holds a whole WAV file in its samples (the noise burst) is cleaned and
        saved back - no new Gemini request."""
        pcm, rate = read_wav(str(path))
        clean = unwrap_wav(pcm)
        if clean is not pcm:
            log.info("tts cache: removed a WAV header / signature from %s", path.name)
            try:
                write_wav(path, clean, rate)
            except OSError:
                pass
        return clean, rate

    def _pause_for_the_day(self, model):
        """Until Google's next quota day (midnight in California)."""
        la = ZoneInfo("America/Los_Angeles")
        now = datetime.fromtimestamp(self._clock(), la)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
        self.paused[model] = midnight.timestamp()

    def _make(self, text, model, style):
        """One model, up to 3 tries for an empty answer. (samples, rate) or None."""
        for attempt in range(3):
            if not (self.key and self._take_request()):
                return None
            try:
                pcm, rate = self.synth(text, self.key, model, self.voice, style)
                write_wav(self.cache_path(text, model, style), pcm, rate)
                self._refusals.pop(model, None)
                self._trim_cache()
                return pcm, rate
            except Refused as e:
                self._refused(model, e)
                return None
            except Exception as e:  # noqa: BLE001 - no internet, a bad answer
                again = str(e).startswith("no audio") and attempt < 2     # an empty answer: worth another try
                log.warning("gemini tts %s failed (%s)%s", model, str(e)[:200], ", trying again" if again else "")
                if not again:
                    return None
        return None

    def _refused(self, model, e):
        """Gemini said no to model: rest it (or every model) as long as the answer says."""
        self._give_back_request()        # refused = nothing made: does not count toward our daily limit
        if e.code == 429:
            self._refusals[model] = self._refusals.get(model, 0) + 1
        if e.code == 429 and (e.per_day or self._refusals[model] >= config.TTS_DAY_AFTER_429):
            # out of this model's daily quota: asking again is wasted time (~0.4 s before every reply on
            # 2026-11-01: Google's "exceeded your current quota" did not say "per day")
            self._pause_for_the_day(model)
        elif e.code == 429:              # a per-minute limit: wait as asked
            self.paused[model] = self._clock() + (e.retry_s or config.TTS_RATE_PAUSE_S)
        elif e.code == 404 or (e.code == 400 and "not found" in str(e).lower()):
            self._pause_for_the_day(model)       # the model is gone (a retired preview)
        elif e.code in (402, 403):       # no credit / not allowed: the same for every model
            self.paused_until = self._clock() + config.TTS_PAUSE_S
        log.warning("gemini tts %s refused (%s)", model, str(e)[:120])

    def _trim_cache(self, max_bytes=None):
        """Keep the phrase cache under TTS_CACHE_MAX_BYTES: the least recently used phrases go first. Looked at
        no more than once a minute."""
        max_bytes = config.TTS_CACHE_MAX_BYTES if max_bytes is None else max_bytes
        now = time.monotonic()
        if now - self._cache_checked < 60 and max_bytes == config.TTS_CACHE_MAX_BYTES:
            return
        self._cache_checked = now
        try:
            files = [(p.stat(), p) for p in self.root.rglob("*.wav")]
        except OSError:
            return
        total = sum(st.st_size for st, _ in files)
        for st, p in sorted(files, key=lambda f: max(f[0].st_atime, f[0].st_mtime)):
            if total <= max_bytes:
                break
            try:
                p.unlink()
                total -= st.st_size
            except OSError:
                pass

    def level(self):
        """Speech peak now: the dashboard's setting (TTS_LEVEL_FILE) if there is one, else by day / night."""
        try:
            v = float(json.loads(Path(self.level_file).expanduser().read_text())["level"])
            return min(1.0, max(0.0, v))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        hour = datetime.fromtimestamp(self._clock(), self.tz).hour
        start, end = config.TTS_NIGHT_HOURS
        night = hour >= start or hour < end
        return config.TTS_LEVEL_NIGHT if night else config.TTS_LEVEL_DAY

    def say_now(self, text, level=None):
        """Speak and wait until done. Returns where the sound came from."""
        pcm, rate, src = self.audio(text)
        if src == "silent":
            return src
        self.speaking = True
        try:
            self.player(to_card(pcm, rate, self.level() if level is None else level))
        finally:
            self.speaking = False
        log.info("said (%s): %s", src, text)
        return src

    def prepare(self, texts, gap_s=config.TTS_PREPARE_GAP_S, sleep=time.sleep):
        """Make (and cache) phrases without playing them, at most one Gemini call per gap_s (the per-minute
        limit). Returns their sources."""
        out = []
        with self._prepare_lock:
            for t in texts:
                if out and out[-1] == "gemini":
                    sleep(gap_s)
                out.append(self.audio(t)[2])
        return out

    # ---------- without waiting ----------

    def say(self, text, level=None, mark=None):
        """Queue a phrase; the voice threads make and speak it (in order). mark (a dict) gets "made" (sound ready),
        "play" (playing starts; time.monotonic) and "src" - for the conversation's timing line."""
        with self._thread_lock:                    # the greeter and the conversation may both say the first
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
                self._player = threading.Thread(target=self._play_run, name="speaker-out", daemon=True)
                self._thread.start()
                self._player.start()
        self.spoken.append((time.monotonic(), text))
        with self._pending_lock:
            self._pending += 1
        self._todo.put((text, level, mark))

    @contextlib.contextmanager
    def hold(self):
        """busy for the whole block - a reply whose next sentence is still being written: the ears stay muted and
        nobody else's turn starts between its sentences."""
        with self._pending_lock:
            self._holds += 1
        try:
            yield self
        finally:
            with self._pending_lock:
                self._holds = max(0, self._holds - 1)       # an interrupt has set it to 0 already

    def interrupt(self):
        """Someone talks over the robot (robot/bargein.py): stop now - what plays fades out, what is queued or still
        being made is dropped, a hold() no longer keeps it busy. The ears listen at once."""
        with self._pending_lock:
            self._gen += 1
            self._holds = 0
        self.interrupted_at = time.monotonic()
        for q in (self._todo, self._ready):
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if item is None:                # closing: keep the marker for the thread
                    q.put(None)
                    break
                self._done()
        try:
            self.stopper()
        except Exception:  # noqa: BLE001 - the speaker gone: nothing plays anyway
            log.exception("could not stop the speaker")
        log.info("voice: interrupted - someone talks over the robot")

    @property
    def busy(self):
        """Speaking or about to - also while a queued phrase is still being made (Gemini: seconds) or a reply is
        still being written (hold); the ears do not listen to the robot itself, the conversation waits for the end."""
        return self.speaking or self._pending > 0 or self._holds > 0

    def close(self, timeout=30):
        with self._thread_lock:
            if self._thread is not None:
                self._todo.put(None)
                self._thread.join(timeout=timeout)
                self._player.join(timeout=timeout)
                self._thread = self._player = None

    def _done(self):
        with self._pending_lock:
            self._pending -= 1

    def _run(self):
        """Make each phrase's sound (the next one while the speaker still plays the one before)."""
        while True:
            item = self._todo.get()
            if item is None:
                self._ready.put(None)
                return
            text, level, mark = item
            gen = self._gen
            try:
                if self._stream(text, level, mark, gen):
                    continue
                pcm, rate, src = self.audio(text)
                if gen != self._gen:                # interrupted while it was being made
                    self._done()
                    continue
                if mark is not None:
                    mark.update(made=time.monotonic(), src=src)
                if src == "silent":
                    self._done()
                    continue
                wav = to_card(pcm, rate, self.level() if level is None else level)
            except Exception:  # noqa: BLE001 - one failed phrase must not silence the robot
                log.exception("could not make %r", text)
                self._done()
                continue
            self._ready.put((text, wav, src, mark))

    def _stream(self, text, level, mark, gen=None):
        """Not cached yet and a 3.x model there: the phrase streams. It goes to the speaker's queue AT ONCE (in order)
        and a thread of its own fills it as Gemini makes it - so sentence 2 is being made while sentence 1 still
        plays (one after the other, it waited ~2 s more: a pause between the sentences, Selim 2026-11-01). Cached
        once whole. False = not streamed (cached, only tags, no streaming model now): the whole-phrase way."""
        if self.streamer is None:
            return False
        said = spoken(text)
        if not has_words(said) or any(self.cache_path(said, m, st).exists() for m, st in self._cache_keys()):
            return False
        now = self._clock()
        free = [(m, st) for m, st in self.models if now >= max(self.paused.get(m, 0.0), self.paused_until)]
        if not free or not interactions_model(free[0][0]):
            return False                    # the first voice to ask does not stream (Cloud TTS): the whole phrase
        models = [(m, st) for m, st in free if interactions_model(m)]
        if not self.key:
            return False
        q = queue.Queue()
        self._ready.put(("stream", text, q, level, mark, self._gen))
        threading.Thread(target=self._fill_stream, args=(text, said, q, mark, gen, models), name="voice-stream",
                         daemon=True).start()
        return True

    def _fill_stream(self, text, said, q, mark, gen, models):
        """The stream's thread: the first model that answers fills q (the next one if refused / failed before its
        first piece); none -> the whole-phrase way (older models, the limit notice), as one piece."""
        try:
            for model, style in models:
                if gen is not None and gen != self._gen:
                    return
                if not self._take_request():
                    break
                pieces = self.streamer(said, self.key, model, self.voice, style)
                try:
                    first = next(pieces)
                except Refused as e:
                    self._refused(model, e)
                    continue
                except Exception as e:  # noqa: BLE001 - no internet, an empty answer: the next model
                    log.warning("gemini tts %s stream failed (%s)", model, str(e)[:200] or type(e).__name__)
                    continue
                if mark is not None:
                    mark.update(made=time.monotonic(), src="stream")
                q.put(first)
                parts, whole = [first], True
                try:
                    for piece in pieces:
                        if gen is not None and gen != self._gen:
                            whole = False           # cut off: the rest is not wanted (nor cached)
                            pieces.close()
                            break
                        q.put(piece)
                        parts.append(piece)
                except Exception as e:  # noqa: BLE001 - what came is played; not cached
                    whole = False
                    log.warning("gemini tts %s stream broke off (%s)", model, str(e)[:200])
                if whole:
                    self._refusals.pop(model, None)
                    try:
                        write_wav(self.cache_path(said, model, style), np.concatenate(parts), STREAM_RATE)
                        self._trim_cache()
                    except OSError:
                        pass
                return
            if gen is not None and gen != self._gen:
                return
            pcm, rate, src = self.audio(text)           # no stream: the older models, or the limit notice
            if mark is not None:
                mark.update(made=time.monotonic(), src=src)
            if src != "silent":
                n = max(1, int(len(pcm) * STREAM_RATE / rate))
                q.put(np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.float32))
        except Exception:  # noqa: BLE001 - one failed phrase must not silence the robot
            log.exception("could not make %r", text)
        finally:
            q.put(None)

    def _play_stream(self, q, level, gen=None):
        card = CardStream(STREAM_RATE, self.level() if level is None else level)
        while (piece := q.get()) is not None:         # not iter(q.get, None): a piece is an array (== fails)
            if gen is not None and gen != self._gen:
                return                                  # interrupted: speakerd fades out what it has
            yield card.push(piece)
        yield card.finish()

    def _play_run(self):
        while True:
            item = self._ready.get()
            if item is None:
                return
            if item[0] == "stream":
                _, text, q, level, mark, gen = item
                src = "stream"
            else:
                text, wav, src, mark = item
            self.speaking = True
            try:
                if mark is not None:
                    mark["play"] = time.monotonic()
                if src == "stream":
                    self.stream_player(self._play_stream(q, level, gen))
                else:
                    self.player(wav)
                log.info("said (%s): %s", src, text)
            except Exception:  # noqa: BLE001
                log.exception("could not speak %r", text)
            finally:
                self.speaking = False
                self._done()

    # ---------- daily limit ----------

    def usage(self):
        """(today, Gemini requests made today)."""
        today = datetime.fromtimestamp(self._clock(), self.tz).strftime("%Y-%m-%d")
        try:
            u = json.loads((self.root / "usage.json").read_text())
        except (OSError, ValueError):
            u = {}
        return today, (u.get("requests", 0) if u.get("day") == today else 0)

    def _take_request(self):
        with self._lock, count_lock(self.root / "usage.json"):
            today, n = self.usage()
            if n >= self.daily:
                log.warning("gemini tts: %d requests today (limit %d), silent until midnight", n, self.daily)
                return False
            tmp = self.root / "usage.json.tmp"
            tmp.write_text(json.dumps({"day": today, "requests": n + 1}))
            os.replace(tmp, self.root / "usage.json")
            return True

    def _give_back_request(self):
        with self._lock, count_lock(self.root / "usage.json"):
            today, n = self.usage()
            if n > 0:
                tmp = self.root / "usage.json.tmp"
                tmp.write_text(json.dumps({"day": today, "requests": n - 1}))
                os.replace(tmp, self.root / "usage.json")


class MuteVoice:
    """alive.py --mute: the voice's interface, but silent and free - nothing is made (no TTS request) or played.
    What it would say is logged ("would say: ...") and the conversation's "said" events reach the dashboard as usual,
    so the ears + mind can be tested without a speaker (2026-11-02: the speaker was broken). The ears are never muted
    by it (nothing plays), and no barge-in can happen."""

    voice, model, key, daily = "mute", "mute", None, 0
    speaking = False
    busy = False
    interrupted_at = None

    def __init__(self, clock=time.time):
        self._clock = clock
        self.spoken = collections.deque(maxlen=12)
        self.said = 0

    def say(self, text, level=None, mark=None):
        now = time.monotonic()
        self.spoken.append((now, text))
        self.said += 1
        if mark is not None:
            mark.setdefault("made", now)
            mark.setdefault("play", now)
            mark.setdefault("src", "mute")
        if has_words(spoken(text)):
            log.info("would say: %s", text)

    def say_now(self, text, level=None):
        self.say(text, level)
        return "mute"

    def prepare(self, texts, **_):
        return ["mute"] * len(list(texts))

    def cached(self, text):
        return True

    @contextlib.contextmanager
    def hold(self):
        yield self

    def interrupt(self):
        pass

    def level(self):
        return 0.0

    def usage(self):
        return datetime.fromtimestamp(self._clock()).strftime("%Y-%m-%d"), 0

    def close(self, timeout=30):
        pass
