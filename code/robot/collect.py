"""
Collect: keep the faces the robot sees, to test and tune recognition on real robot views (dark rooms, seen from
the floor, people turning) instead of clean internet photos. Local only - nothing leaves the Pi.

    collector = Collector(config.COLLECT_DIR)
    vision = Vision(..., collector=collector)       # vision calls collector.save(look, ...) after each look

Per saved look, in COLLECT_DIR/<YYYY-MM-DD>/:
    <HHMMSS>-t<track>-<n>-face.png      the aligned 112 x 112 face (what the embedder saw)
    <HHMMSS>-t<track>-<n>-crop.jpg      the face region of the big stream (to re-align / re-embed later)
and one line in COLLECT_DIR/looks.jsonl: time, track, files, best match + similarity, the track's person,
face size / turn / brightness, head pose and speed, the condition bin. At most one look per track every
COLLECT_EVERY_S, and only a new KIND of look (balanced data, not thousands of the same face at the same desk):
each person's looks of a day are spread over condition bins (bin_of: distance x head turn x light x looking), at
most COLLECT_PER_BIN_DAY per bin and none too alike one already kept there; strangers / undecided tracks at most
COLLECT_STRANGER_PER_TRACK. The day's counts live in state_path, so a restart does not start them again.
Nothing more is saved above max_bytes.
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

log = logging.getLogger(__name__)


def folder_bytes(root):
    return sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())


def bin_of(look):
    """The look's condition bin: (distance, head turn, light, looking at the robot)."""
    w = look.width_px
    dist = "far" if w < 60 else "mid" if w < 100 else "near" if w < 160 else "close"
    turn = "straight" if look.yaw < 0.12 else "turned" if look.yaw < 0.30 else "side"
    light = "dark" if look.luma < 60 else "dim" if look.luma < 120 else "bright"
    looking = "looking" if look.engagement >= config.GAZE_LOOKING_ON else "away"
    return f"{dist}/{turn}/{light}/{looking}"


class Collector:
    def __init__(self, root=config.COLLECT_DIR, every_s=config.COLLECT_EVERY_S, max_bytes=config.COLLECT_MAX_BYTES,
                 clock=time.time, state_path=None, per_bin=config.COLLECT_PER_BIN_DAY,
                 alike_sim=config.COLLECT_ALIKE_SIM, stranger_per_track=config.COLLECT_STRANGER_PER_TRACK):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.every_s, self.max_bytes, self._clock = every_s, max_bytes, clock
        self.per_bin, self.alike_sim, self.stranger_per_track = per_bin, alike_sim, stranger_per_track
        self.used = folder_bytes(self.root)
        self.saved = 0
        self.skipped = 0                    # looks that added nothing new
        self.failed = None                  # a write failed (disk full): nothing more is saved
        self._last = {}                     # track id -> time of its last saved look (look.t)
        self._full_logged = False
        self.state_path = Path(state_path).expanduser() if state_path else None
        self._day, self._counts = None, {}  # "person|bin" -> kept today (persisted)
        self._kept = {}                     # "person|bin" -> today's kept vectors (memory only)
        self._load_state()

    # ---------- balance: only new kinds of look ----------

    def _today(self):
        return datetime.fromtimestamp(self._clock()).strftime("%Y-%m-%d")

    def _load_state(self):
        if self.state_path and self.state_path.exists():
            try:
                st = json.loads(self.state_path.read_text())
                if st.get("day") == self._today():
                    self._day, self._counts = st["day"], dict(st.get("counts", {}))
            except (OSError, ValueError, KeyError):
                pass

    def _save_state(self):
        if self.state_path:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"day": self._day, "counts": self._counts}))
            os.replace(tmp, self.state_path)

    def _new_kind(self, look, vec, person, unknown):
        """Is this look worth keeping? (bin quota, not alike a kept one). Returns its count key or None."""
        today = self._today()
        if today != self._day:
            self._day, self._counts, self._kept = today, {}, {}
        if person is None:                  # stranger / not decided: a few looks per track
            key = f"track{look.track}|all"
            limit = self.stranger_per_track
        else:
            key = f"{person}|{bin_of(look)}"
            limit = self.per_bin
        if self._counts.get(key, 0) >= limit:
            return None
        if vec is not None:
            kept = self._kept.get(key, [])
            if kept and float(np.max(np.stack(kept) @ vec)) >= self.alike_sim:
                return None
        return key

    def save(self, look, match=None, sim=0.0, person=None, name=None, unknown=False, vec=None):
        """Keep this look (rate-limited per track, only new kinds of look, size-capped). Returns True when saved.
        A write error (disk full, SD card error) switches collecting off - the robot keeps running."""
        if self.failed:
            return False
        try:
            return self._save(look, match, sim, person, name, unknown, vec)
        except OSError as e:
            self.failed = str(e)
            log.warning("collect: writing into %s failed (%s) - collecting switched off, the robot keeps running",
                        self.root, e)
            return False

    def _save(self, look, match, sim, person, name, unknown, vec):
        if look.t - self._last.get(look.track, -1e9) < self.every_s:
            return False
        key = self._new_kind(look, vec, person, unknown)
        if key is None:
            self.skipped += 1
            return False
        if self.used >= self.max_bytes:
            if not self._full_logged:
                log.warning("collect: %s is full (%.1f GB), not saving more", self.root, self.used / 1e9)
                self._full_logged = True
            return False
        self._last[look.track] = look.t
        self._counts[key] = self._counts.get(key, 0) + 1
        if vec is not None:
            self._kept.setdefault(key, []).append(np.asarray(vec, np.float32))
        self._save_state()
        if len(self._last) > 500:           # tracks long gone
            self._last = dict(sorted(self._last.items(), key=lambda kv: kv[1])[-100:])
        now = datetime.fromtimestamp(self._clock())
        day = self.root / now.strftime("%Y-%m-%d")
        day.mkdir(exist_ok=True)
        self.saved += 1
        stem = f"{now:%H%M%S}-t{look.track}-{self.saved}"
        face, crop = day / f"{stem}-face.png", day / f"{stem}-crop.jpg"
        Image.fromarray(look.aligned).save(face)
        Image.fromarray(look.crop).save(crop, quality=90)
        line = dict(time=round(now.timestamp(), 2), track=look.track, face=str(face.relative_to(self.root)),
                    crop=str(crop.relative_to(self.root)), match=match, sim=round(float(sim), 4), person=person,
                    name=name, unknown=unknown, width_px=round(float(look.width_px), 1), yaw=round(float(look.yaw), 3),
                    luma=round(float(look.luma), 1), pan=round(float(look.pan), 1), tilt=round(float(look.tilt), 1),
                    head_dps=round(float(look.head_dps), 1), embed_ms=round(float(look.ms)), quality=round(float(look.quality), 3), bin=key.split("|")[1], engagement=round(float(look.engagement), 3),
                    facing=None if look.facing is None else round(float(look.facing), 3))
        with open(self.root / "looks.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self.used += face.stat().st_size + crop.stat().st_size + len(json.dumps(line)) + 1
        return True
