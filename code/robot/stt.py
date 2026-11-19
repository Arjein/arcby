"""
Speech to text, Turkish: an utterance from robot/mic.py (16 kHz float32) -> text. Two backends behind one call:

    stt = load_stt()                 # config.STT_BACKEND
    text = stt.transcribe(pcm)       # "" when nothing understandable was said
    stt.close()

WhisperSTT: faster-whisper in its own venv (~/.venvs/arcby-stt), the model loaded once in a child process; on the
Pi, nothing leaves the robot. GeminiSTT: the clip as WAV to the Gemini API (GEMINI_API_KEY in config.SECRETS_FILE)
- better Turkish, but the voice leaves the robot: only with Selim's OK (config.STT_BACKEND).
"""
import base64
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

from . import config
from datetime import datetime
from zoneinfo import ZoneInfo

from .tts import API, Refused, _post_json, count_lock, read_secret

log = logging.getLogger(__name__)

STT_PY = Path.home() / ".venvs" / "arcby-stt" / "bin" / "python"
SERVER = r"""
import sys
from faster_whisper import WhisperModel
m = WhisperModel(sys.argv[1], device="cpu", compute_type="int8", cpu_threads=int(sys.argv[2]))
print("ready", flush=True)
for line in sys.stdin:
    path, lang, prompt = line.rstrip("\n").split("\t")
    segs, _ = m.transcribe(path, language=lang or None, beam_size=1, vad_filter=False,
                           condition_on_previous_text=False, initial_prompt=prompt or None)
    print(" ".join(s.text.strip() for s in segs).replace("\n", " "), flush=True)
"""


def wav_bytes(pcm, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        x = np.asarray(pcm, np.float32)
        peak = float(np.abs(x).max()) or 1.0
        w.writeframes((x / peak * 0.9 * 32767).astype("<i2").tobytes())   # normalised: the mic is quiet
    return buf.getvalue()


# answers that are not words: "...", "(anlaşılmıyor)", "[müzik]", "*inaudible*"
_NOT_SPEECH = re.compile(r"\([^)]*\)|\[[^\]]*\]|\*[^*]*\*")


def clean(text):
    """The STT answer without descriptions of sounds; "" if no letter is left."""
    text = _NOT_SPEECH.sub(" ", text or "")
    text = " ".join(text.split())
    return text if any(c.isalpha() for c in text) else ""


class WhisperSTT:
    def __init__(self, model=config.STT_WHISPER_MODEL, lang=config.STT_LANG, threads=4, python=STT_PY,
                 popen=subprocess.Popen):
        self.model, self.lang, self.threads, self.python, self._popen = model, lang, threads, python, popen
        self.proc = None
        self._start()
        self._tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.name = f"whisper-{model}"

    def _start(self):
        self.proc = self._popen([str(self.python), "-u", "-c", SERVER, self.model, str(self.threads)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        if self.proc.stdout.readline().strip() != "ready":
            raise RuntimeError(f"whisper {self.model} did not start")

    def transcribe(self, pcm, prompt=""):
        """The words, or "" - a dead server is started again (once per call) instead of staying deaf."""
        Path(self._tmp.name).write_bytes(wav_bytes(pcm))
        for attempt in range(2):
            try:
                if self.proc is None or self.proc.poll() is not None:
                    log.warning("whisper: the server is gone - starting it again")
                    self._start()
                self.proc.stdin.write(f"{self._tmp.name}\t{self.lang}\t{prompt}\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if line:
                    return clean(line.strip())
            except (OSError, ValueError, RuntimeError) as e:
                log.warning("whisper failed (%s)", e)
            self.proc = None if self.proc is None or self.proc.poll() is not None else self.proc
            if self.proc is not None:            # alive but said nothing: kill it so the next try starts fresh
                self.proc.kill()
                self.proc = None
        return ""

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        Path(self._tmp.name).unlink(missing_ok=True)


class DailyLimit:
    """At most `per_day` requests a day (days in TTS_TIMEZONE), counted in a small JSON file."""

    def __init__(self, path, per_day, clock=time.time, tz=config.TTS_TIMEZONE):
        self.path, self.per_day, self._clock, self.tz = Path(path).expanduser(), per_day, clock, ZoneInfo(tz)

    def used(self):
        today = datetime.fromtimestamp(self._clock(), self.tz).strftime("%Y-%m-%d")
        try:
            u = json.loads(self.path.read_text())
        except (OSError, ValueError):
            u = {}
        return today, (u.get("requests", 0) if u.get("day") == today else 0)

    def take(self):
        with count_lock(self.path):         # two threads / processes counting at once
            today, n = self.used()
            if n >= self.per_day:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"day": today, "requests": n + 1}))
            os.replace(tmp, self.path)
            return True


def thinking_config(model, thinking):
    """generationConfig "thinkingConfig" for STT_GEMINI_THINKING: "off" -> the least the model allows (3.x:
    thinkingLevel minimal; 2.5: thinkingBudget 0), None -> none sent (the model's default), a dict as it is."""
    if not thinking:
        return None
    if thinking == "off":
        return {"thinkingBudget": 0} if re.search(r"gemini-2\.", model) else {"thinkingLevel": "minimal"}
    return dict(thinking)


class GeminiSTT:
    PROMPT = ("Transcribe this short speech exactly as spoken. It is usually Turkish (write Turkish with Turkish "
              "letters), sometimes English: write each word in the language it was spoken in - never translate. "
              "Only the words, no explanation, no descriptions of sounds. Coughs, sneezes, throat-clearing, "
              "laughter and hesitation sounds are not words: leave them out. If nothing understandable is said, "
              "answer with nothing.")

    def __init__(self, model=config.STT_GEMINI_MODEL, key=None, post=_post_json, timeout=20, clock=time.time,
                 limit=None, thinking=config.STT_GEMINI_THINKING):
        self.model, self.post, self.timeout, self._clock, self.thinking = model, post, timeout, clock, thinking
        self.key = read_secret("GEMINI_API_KEY") if key is None else key
        self.name = f"gemini-{model}"
        self.limit = limit or DailyLimit(Path(config.TTS_CACHE_DIR).expanduser().parent / "stt_usage.json",
                                         config.STT_DAILY_REQUESTS, clock)
        self.paused_until = 0.0             # refused (no credit / too many): deaf until then
        self.last_error = None
        self.last_seconds = None            # how long the last call took
        self._thinking_ok = True            # False: the model refused the thinkingConfig (HTTP 400) - not sent again
        self._fallback_think = None         # ...then this one instead ("low" for a model without "minimal")

    def transcribe(self, pcm, prompt=""):
        """The words, or "" - also when the daily limit is reached or Gemini refused (last_error says why)."""
        if not self.key or self._clock() < self.paused_until:
            self.last_error = "no key" if not self.key else "paused"
            return ""
        if not self.limit.take():
            self.last_error = f"daily limit {self.limit.per_day} reached"
            return ""
        t = time.monotonic()
        try:
            self.last_error = None
            try:
                return clean(self._call(pcm, prompt))
            except Refused as e:
                if not (e.code == 400 and "think" in str(e).lower() and self._thinking_ok and self.thinking):
                    raise
                # e.g. 3.8 Flash has no "minimal" level: "low" then (without any setting it thinks at its default:
                # 3-5 s instead of ~2.5, 2026-11-01); any other refused setting: sent without it, for good
                was = thinking_config(self.model, self.thinking)
                self._fallback_think = {"thinkingLevel": "low"} \
                    if was == {"thinkingLevel": "minimal"} and "minimal" in str(e).lower() else None
                self._thinking_ok = False
                log.warning("gemini stt: %s refused the thinking setting (%s) - %s from now on", self.model,
                            str(e)[:120], self._fallback_think or "sent without it")
                return clean(self._call(pcm, prompt))
        except Refused as e:
            pause = (e.retry_s or config.TTS_RATE_PAUSE_S) if e.code == 429 else \
                config.TTS_PAUSE_S if e.code in (402, 403) else 0.0
            self.paused_until = self._clock() + pause
            self.last_error = str(e)[:160]
            log.warning("gemini stt refused (%s) - deaf for %.0f s", self.last_error, pause)
        except Exception as e:  # noqa: BLE001 - no internet: the robot just did not understand
            self.last_error = str(e)[:160]
            log.warning("gemini stt failed (%s)", self.last_error)
        finally:
            self.last_seconds = round(time.monotonic() - t, 2)
        return ""

    def _call(self, pcm, prompt):
        body = {"contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav_bytes(pcm)).decode()}},
            {"text": self.PROMPT + (f" These names belong to the household and may be said: {prompt}. Write a "
                                    "name only when it is clearly spoken - never guess one for unclear sound."
                                    if prompt else "")}]}],
            "generationConfig": {"temperature": 0.0}}
        think = thinking_config(self.model, self.thinking) if self._thinking_ok else self._fallback_think
        if think:
            body["generationConfig"]["thinkingConfig"] = think
        req = urllib.request.Request(API.format(model=self.model), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        cand = (self.post(req, self.timeout).get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        return " ".join(p.get("text", "") for p in parts).strip()

    def close(self):
        pass


def load_stt(backend=config.STT_BACKEND, **kw):
    return GeminiSTT(**kw) if backend == "gemini" else WhisperSTT(**kw)


def timed(stt, pcm, prompt=""):
    """(text, seconds)."""
    t = time.perf_counter()
    text = stt.transcribe(pcm, prompt)
    return text, time.perf_counter() - t

