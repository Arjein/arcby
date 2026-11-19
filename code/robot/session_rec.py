"""
Test sessions recorded by the running robot (docs/vision.md S3): while SESSION_FLAG exists, alive.py
saves the big-stream frames (SESSION_FPS a second, in a thread - the camera loop does not wait) and robot/show.py the
hand crops it checks, so a guided session becomes a test set seen exactly as the robot sees (the head following,
the real light). Local only (faces) - on the stick.

    SESSION_FLAG = {"dir": "/mnt/arcby-data/sessions/<name>", "until": <unix time>}   (the guide writes it, removes it)
    <dir>/frames/<n>.jpg + frames.jsonl {n, wall, t}            <dir>/crops/<n>.jpg + crops.jsonl (robot/show.py)
"""
import json
import logging
import queue
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


class SessionRecorder:
    def __init__(self, flag=None, fps=None, clock=time.time):
        self.flag = Path(flag or config.SESSION_FLAG).expanduser()
        self.every = 1.0 / (fps or config.SESSION_FPS)
        self._clock = clock
        self.dir = None
        self.until = 0.0
        self._checked = -1e9
        self._next = 0.0
        self.n = 0
        self._q = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._run, name="session-rec", daemon=True)
        self._thread.start()
        self.dropped = 0

    def active(self, now=None):
        """Is a session on (the flag read at most once a second)?"""
        now = self._clock() if now is None else now
        if now - self._checked >= 1.0:
            self._checked = now
            try:
                d = json.loads(self.flag.read_text())
                folder, until = Path(d["dir"]).expanduser(), float(d["until"])
            except (OSError, ValueError, KeyError, TypeError):
                folder, until = None, 0.0
            if folder != self.dir:
                if folder is not None:
                    (folder / "frames").mkdir(parents=True, exist_ok=True)
                    log.info("session: recording into %s", folder)
                    self.n = 0
                elif self.dir is not None:
                    log.info("session: stopped (%d frames, %d dropped)", self.n, self.dropped)
                self.dir = folder
            self.until = until
        return self.dir is not None and now <= self.until

    def frame(self, f):
        """Each camera frame: saved when a session is on and one is due."""
        now = self._clock()
        if not self.active(now) or now < self._next:
            return
        self._next = now + self.every
        try:
            w, h = f.lores_size if hasattr(f, "lores_size") else f.lores_rgb().shape[1::-1]
            big, _ = f.crop_main((0, 0, w, h), pad=0.0)
        except Exception:  # noqa: BLE001 - no big stream: nothing to keep
            return
        try:
            self._q.put_nowait((self.dir, self.n, now, f.t, big))
            self.n += 1
        except queue.Full:
            self.dropped += 1

    def _run(self):
        from PIL import Image
        while True:
            item = self._q.get()
            if item is None:
                return
            folder, n, wall, t, big = item
            try:
                Image.fromarray(big).save(folder / "frames" / f"{n:05d}.jpg", "JPEG", quality=85)
                with open(folder / "frames.jsonl", "a") as f:
                    f.write(json.dumps({"n": n, "wall": round(wall, 3), "t": round(t, 3)}) + "\n")
            except OSError as e:
                log.warning("session: frame %d not saved (%s)", n, e)

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=5)


def save_crop(folder, n, rgb, info):
    """robot/show.py: one hand crop it checked (+ what it found) into the session."""
    from PIL import Image
    d = Path(folder) / "crops"
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(d / f"{n:05d}.jpg", "JPEG", quality=90)
    with open(Path(folder) / "crops.jsonl", "a") as f:
        f.write(json.dumps(dict(info, n=n)) + "\n")
