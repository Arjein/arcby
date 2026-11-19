"""
Live speech to text: the sentence goes to Gemini WHILE it is said (Live API, model STT_LIVE_MODEL), so the words are
ready ~0.4 s after it ends instead of 2-5 s for a whole clip (robot/stt.py GeminiSTT, measured 2026-11-01 on the Pi).

    live = LiveSTT()                         # None-safe: no key -> live.ok False
    s = live.open()                          # connects in its own thread (~0.5 s) - feed at once, it queues
    s.feed(pcm)                              # 16 kHz float32 pieces, as they are heard
    text = s.finish(timeout=3.0)             # after the sentence ended: the words, "" = nothing understood,
                                             # None = failed (no answer in time / connection broken)
    s.cancel()                               # not wanted after all (never sent, or the rest dropped)

Our own VAD (robot/mic.py) says where the sentence starts and ends (activityStart / activityEnd): the model's own
activity detection skipped quiet sentences. Privacy: robot/conversation.py opens a session only for a sentence said
to the robot (looked at as it started, or in an open conversation) - nothing else leaves the robot.
"""
import base64
import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from . import config
from .stt import DailyLimit
from .tts import read_secret
from .wsock import Closed, WebSocket

log = logging.getLogger(__name__)

URL = ("wss://generativelanguage.googleapis.com/ws/"
       "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")


def setup_message(model=config.STT_LIVE_MODEL, langs=config.STT_LIVE_LANGS):
    return {"setup": {"model": f"models/{model}",
                      "inputAudioTranscription": {"languageCodes": list(langs)},
                      "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}}}}


def audio_message(pcm):
    data = (np.clip(np.asarray(pcm, np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return {"realtimeInput": {"audio": {"data": base64.b64encode(data).decode(), "mimeType": "audio/pcm;rate=16000"}}}


class LiveSession:
    """One sentence. feed() never blocks (the ears thread calls it); a thread connects, sends and reads."""

    def __init__(self, connect, setup, clock=time.monotonic):
        self._connect, self._setup, self._clock = connect, setup, clock
        self._q = queue.Queue()
        self._done = threading.Event()
        self._parts = []
        self.text, self.error = None, None
        self.cancelled = False
        self._ended = False
        self.connect_s = None               # how long the connection took (logs)
        self._ws = None
        self._thread = threading.Thread(target=self._run, name="live-stt", daemon=True)
        self._thread.start()

    def feed(self, pcm):
        if not self.cancelled:
            self._q.put(audio_message(pcm))

    def end(self):
        """The sentence is over (our VAD): the model gives its final words. Once."""
        if not self._ended:
            self._ended = True
            self._q.put({"realtimeInput": {"activityEnd": {}}})

    def finish(self, timeout=config.STT_LIVE_WAIT_S):
        """The words ("" = nothing understood) or None (failed / too slow: use the clip STT instead)."""
        self.end()
        if not self._done.wait(timeout):
            self.error = self.error or f"no answer within {timeout:.1f} s"
            self.cancel()
            return None
        return self.text

    def cancel(self):
        self.cancelled = True
        self._q.put(None)
        ws = self._ws
        if ws is not None:
            ws.close()

    def _run(self):
        t = self._clock()
        try:
            ws = self._ws = self._connect()
            if self.cancelled:
                ws.close()
                return
            ws.send_text(json.dumps(self._setup))
            first = ws.recv(timeout=config.STT_LIVE_WAIT_S)
            if first is None or "setupComplete" not in (first if isinstance(first, str) else first.decode()):
                raise Closed(None, f"setup not confirmed: {str(first)[:120]}")
            self.connect_s = round(self._clock() - t, 2)
            ws.send_text(json.dumps({"realtimeInput": {"activityStart": {}}}))
            reader = threading.Thread(target=self._read, args=(ws,), name="live-stt-read", daemon=True)
            reader.start()
            while True:
                msg = self._q.get()
                if msg is None:
                    break
                ws.send_text(json.dumps(msg))
            reader.join(timeout=1.0)
        except Exception as e:  # noqa: BLE001 - no internet / refused: the clip STT takes over
            if not self.cancelled:
                self.error = str(e)[:160]
                log.warning("live stt failed (%s)", self.error)
        finally:
            self._done.set()
            if self._ws is not None:
                self._ws.close()

    def _read(self, ws):
        try:
            while not self.cancelled:
                msg = ws.recv(timeout=0.5)
                if msg is None:
                    continue
                sc = json.loads(msg).get("serverContent", {})
                tr = sc.get("inputTranscription")
                if tr and tr.get("text"):
                    self._parts.append(tr["text"])
                if sc.get("generationComplete") or sc.get("turnComplete"):
                    self.text = " ".join(p.strip() for p in self._parts if p.strip()).strip()
                    self._done.set()
                    self._q.put(None)
                    return
        except Closed as e:
            if not self.cancelled and not self._done.is_set():
                self.error = str(e)[:160]
        except Exception as e:  # noqa: BLE001
            self.error = str(e)[:160]
        finally:
            self._q.put(None)


class LiveSTT:
    def __init__(self, model=config.STT_LIVE_MODEL, key=None, connect=None, limit=None, clock=time.time):
        self.model = model
        self.key = read_secret("GEMINI_API_KEY") if key is None else key
        self.name = f"live-{model}"
        self._connect = connect or (lambda: WebSocket.connect(URL, headers={"x-goog-api-key": self.key},
                                                              timeout=config.STT_LIVE_WAIT_S))
        self.limit = limit or DailyLimit(Path(config.TTS_CACHE_DIR).expanduser().parent / "stt_live_usage.json",
                                         config.STT_LIVE_DAILY, clock)

    @property
    def ok(self):
        return bool(self.key)

    def open(self):
        """A session for one sentence, or None (no key / daily limit)."""
        if not self.ok or not self.limit.take():
            return None
        return LiveSession(self._connect, setup_message(self.model))


def load_live_stt():
    if not config.STT_LIVE:
        return None
    live = LiveSTT()
    return live if live.ok else None
