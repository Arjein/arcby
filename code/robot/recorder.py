"""
Recorder: everything the robot senses, kept as a tidy dataset to learn from later (own models: eye contact,
people, sounds...). Local only - nothing leaves the robot. No audio (Selim, 2026-10-31).

    rec = Recorder()                          # opens DATA_DIR/<YYYY-MM-DD>/<HHMMSS>/ and writes meta.json
    rec.write("events", {"kind": "said", "text": "Merhaba Selim."})    # any stream; t / wall added if missing
    rec.keyframe(rgb, t, busy=True)           # small-stream picture now and then (rate-limited)
    rec.tick()                                # now and then: flush, size limit
    rec.close()

A session folder holds:
    meta.json             code version (git), every config value, models, command line, start time
    <stream>.jsonl.gz     one JSON object per line, gzip (flushed every DATA_FLUSH_S: readable after a power cut):
                          frames (camera + vision, every processed frame), events, imu, sonar, system
    keyframes/<t>.jpg     the small stream - only when the scene changed (DATA_KEYFRAME_CHANGE) or who is in view,
                          at most 1 per DATA_KEYFRAME_BUSY_S (faces in view) / _IDLE_S, at least 1 per _MAX_S
    faces/                face crops + looks.jsonl (robot/collect.py)
Every record has "t" (time.monotonic(), the frame / sensor clock - lines streams up) and "wall" (Unix time).
When all of DATA_DIR passes DATA_MAX_BYTES (or the disk has less than DATA_MIN_FREE_BYTES free), the oldest sessions
lose first their keyframes, then their streams, and only last their face crops (small and the most valuable); the
current session loses only its oldest keyframes, and only when it alone is over. The check runs in its own thread
(walking gigabytes of files must not hold up the camera). A write that fails (disk full, SD card error) switches
recording off with one warning - the robot itself keeps running.
Read with code/data_cli.py, or gzip.open(path, "rt") and one json.loads per line.
"""
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

log = logging.getLogger(__name__)


def folder_bytes(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def config_snapshot():
    """Every UPPER_CASE config value that JSON can hold."""
    out = {}
    for k, v in vars(config).items():
        if k.isupper():
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = repr(v)
    return out


def git_version(repo=Path(__file__).resolve().parents[2]):
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return f"{head}{' +local changes' if dirty else ''}" if head else "unknown (not a git checkout)"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Recorder:
    def __init__(self, root=config.DATA_DIR, max_bytes=config.DATA_MAX_BYTES, meta=None, clock=time.monotonic,
                 wall=time.time):
        self.root = Path(root).expanduser()
        self.max_bytes = max_bytes
        self._clock, self._wall = clock, wall
        now = datetime.fromtimestamp(wall())
        self.dir = self.root / now.strftime("%Y-%m-%d") / now.strftime("%H%M%S")
        n = 1
        while self.dir.exists():                    # two runs in one second
            n += 1
            self.dir = self.root / now.strftime("%Y-%m-%d") / f"{now:%H%M%S}-{n}"
        (self.dir / "keyframes").mkdir(parents=True)
        self.faces_dir = self.dir / "faces"
        self._lock = threading.Lock()
        self._streams = {}
        self.counts = {}
        self._next_key_t = -1e9
        self._key_thumb, self._key_who, self._key_t = None, None, -1e9   # the last KEPT keyframe
        self.keyframes_skipped = 0
        self._next_flush = self._next_retain = 0.0
        self.keyframes = 0
        self.failed = None                          # why recording stopped (a write failed), else None
        self._retainer = None
        info = dict(started=now.isoformat(timespec="seconds"), host=os.uname().nodename, code=git_version(),
                    command=sys.argv, python=sys.version.split()[0], config=config_snapshot())
        info.update(meta or {})
        (self.dir / "meta.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
        self.retain()

    # ---------- writing ----------

    def _fail(self, what, e):
        """A write failed (disk full, SD card error): recording stops, the robot goes on."""
        if self.failed is None:
            self.failed = f"{what}: {e}"
            log.warning("recorder: %s failed (%s) - recording switched off, the robot keeps running", what, e)

    def write(self, stream, record):
        if self.failed:
            return
        rec = dict(record)
        rec.setdefault("t", round(self._clock(), 4))
        rec.setdefault("wall", round(self._wall(), 3))
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=_plain) + "\n"
        with self._lock:
            try:
                f = self._streams.get(stream)
                if f is None:
                    f = self._streams[stream] = gzip.open(self.dir / f"{stream}.jsonl.gz", "at", compresslevel=6)
                f.write(line)
            except OSError as e:
                self._fail(f"writing {stream}", e)
                return
            self.counts[stream] = self.counts.get(stream, 0) + 1

    def keyframe(self, rgb, t, busy, who=()):
        """Keep the picture if the scene (or who is in view) changed since the last kept one - not more often than
        BUSY_S / IDLE_S, not less often than MAX_S. Returns its file name or None."""
        if self.failed or t < self._next_key_t:
            return None
        self._next_key_t = t + (config.DATA_KEYFRAME_BUSY_S if busy else config.DATA_KEYFRAME_IDLE_S)
        thumb = scene_thumb(rgb)
        who = frozenset(who)
        changed = (self._key_thumb is None or who != self._key_who
                   or float(np.abs(thumb - self._key_thumb).mean()) >= config.DATA_KEYFRAME_CHANGE
                   or t - self._key_t >= config.DATA_KEYFRAME_MAX_S)
        if not changed:
            self.keyframes_skipped += 1
            return None
        self._key_thumb, self._key_who, self._key_t = thumb, who, t
        name = f"{t:.3f}.jpg"
        try:
            Image.fromarray(rgb).save(self.dir / "keyframes" / name, quality=config.DATA_KEYFRAME_QUALITY)
        except OSError as e:
            self._fail("a keyframe", e)
            return None
        self.keyframes += 1
        return name

    def tick(self):
        """Call often: flushes now and then, keeps the size limit now and then."""
        now = self._clock()
        if now >= self._next_flush:
            self.flush()
            self._next_flush = now + config.DATA_FLUSH_S
        if now >= self._next_retain and (self._retainer is None or not self._retainer.is_alive()):
            self._next_retain = now + config.DATA_RETAIN_EVERY_S
            self._retainer = threading.Thread(target=self._retain_quietly, name="recorder-retain", daemon=True)
            self._retainer.start()

    def _retain_quietly(self):
        try:
            self.retain()
        except Exception:  # noqa: BLE001 - the size check must never take the robot down
            log.exception("recorder: size check failed")

    def flush(self):
        with self._lock:
            for f in self._streams.values():
                try:
                    f.flush()                       # gzip sync point: everything so far is readable
                except OSError as e:
                    self._fail("flushing", e)

    def close(self):
        if self._retainer is not None:
            self._retainer.join(timeout=5)
        with self._lock:
            for f in self._streams.values():
                try:
                    f.close()
                except OSError as e:
                    self._fail("closing", e)
            self._streams.clear()

    # ---------- size limit ----------

    def sessions(self):
        """All session folders, oldest first."""
        return sorted((s for d in self.root.iterdir() if d.is_dir() for s in d.iterdir() if s.is_dir()),
                      key=lambda p: (p.parent.name, p.name)) if self.root.exists() else []

    def limit(self, total, free_bytes=None, min_free=config.DATA_MIN_FREE_BYTES):
        """Bytes DATA_DIR may hold now: max_bytes, less if the disk would keep under min_free free."""
        if free_bytes is None:
            try:
                free_bytes = shutil.disk_usage(self.root).free
            except OSError:
                return self.max_bytes
        return max(0, min(self.max_bytes, total + free_bytes - min_free))

    def retain(self, free_bytes=None):
        """While the whole DATA_DIR is over the limit (max_bytes, or less to keep DATA_MIN_FREE_BYTES of the disk
        free): the oldest sessions lose their keyframes, then their streams, then the rest (face crops) - the most
        valuable, smallest part goes last. This session loses only its oldest keyframes, when it alone is over."""
        self._next_retain = self._clock() + config.DATA_RETAIN_EVERY_S
        old = [s for s in self.sessions() if s != self.dir]
        total = folder_bytes(self.root) if self.root.exists() else 0
        limit = self.limit(total, free_bytes)
        stages = (("keyframes", lambda s: [s / "keyframes"]),
                  ("streams", lambda s: list(s.glob("*.jsonl.gz"))),
                  ("session", lambda s: [s]))
        for stage, parts in stages:
            for s in old:
                if total <= limit:
                    return total
                for p in parts(s):
                    if not p.exists():
                        continue
                    b = folder_bytes(p) if p.is_dir() else p.stat().st_size
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink()
                    total -= b
                    log.info("recorder: data limit %.1f GB - deleted %s of %s (%.0f MB)", limit / 1e9,
                             stage, s.relative_to(self.root), b / 1e6)
                if stage == "session" and s.parent.exists() and not any(s.parent.iterdir()):
                    s.parent.rmdir()
        if total > limit:                   # this session alone: its oldest keyframes go (by time in the name)
            keys = sorted(self.dir.glob("keyframes/*.jpg"), key=_key_time)
            dropped = 0
            for p in keys[:-1]:
                if total <= limit:
                    break
                try:
                    b = p.stat().st_size
                    p.unlink()
                except OSError:
                    continue
                total -= b
                dropped += 1
            log.warning("recorder: this session alone is over the data limit (%.1f GB) - %d oldest keyframes deleted",
                        limit / 1e9, dropped)
        return total


def _key_time(path):
    """A keyframe's time from its name (<t>.jpg); other names first."""
    try:
        return float(path.stem)
    except ValueError:
        return -1.0


def scene_thumb(rgb):
    """48 x 27 grey, normalised (exposure changes are not scene changes): for keyframe change tests."""
    g = np.asarray(Image.fromarray(rgb).convert("L").resize((48, 27), Image.BILINEAR), np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def _plain(o):
    """numpy numbers / arrays and tuples -> plain JSON."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)
