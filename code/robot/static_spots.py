"""
Static spots: places where the face detector keeps "finding" something that never moves and never passes the face
check - a pattern on a cushion, a picture on the wall. In the first recordings two such spots made 685 of 823
tracks. Once a spot is known, a detection there is checked only every STATIC_CHECK_S per spot (a person may stand
there) and makes no face_new / face_lost events.

    spots = StaticSpots(config.STATIC_FILE)
    spots.note_ended(track)            # vision.py: every ended track teaches it (only still, never-a-face ones count)
    spots.is_static(pan, tilt)         # is this place one of them?
    spots.check_due(pan, tilt, t)      # ...and may it be checked again now? (once per STATIC_CHECK_S per spot)

Evidence = (pan, tilt, time, mode) where still non-faces ended, in head angles (where the head would look);
STATIC_MIN_HITS within STATIC_RADIUS_DEG and STATIC_KEEP_S make a spot. Evidence: a non-face track that stayed still
(90 % of its positions within STATIC_MAX_DEV_DEG - a running maximum let detector jitter hide the photo), or a
flicker of 1-2 frames that the face check rejected (most of the churn is such flickers, 2026-11-01).
mode = how head angles were known: "head" (a powered head, its pose) or "nohead" (no battery: the camera points
wherever the limp head stopped, angles are relative to that) - spots of one mode mean nothing in the other, so the
owner sets spots.mode. Saved to a JSON file, so a restart remembers them; evidence older than STATIC_KEEP_S is
forgotten, and clear() forgets everything (the robot base moved: every place is somewhere else now).
"""
import json
import logging
import math
import os
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


class StaticSpots:
    def __init__(self, path=None, radius_deg=config.STATIC_RADIUS_DEG, min_hits=config.STATIC_MIN_HITS,
                 keep_s=config.STATIC_KEEP_S, max_dev=config.STATIC_MAX_DEV_DEG, check_s=config.STATIC_CHECK_S,
                 clock=time.time):
        self.path = Path(path).expanduser() if path else None
        self.radius, self.min_hits, self.keep_s, self.max_dev = radius_deg, min_hits, keep_s, max_dev
        self.check_s = check_s
        self._clock = clock
        self._lock = threading.Lock()
        self.mode = "head"                  # how angles are known now: "head" / "nohead" (see above)
        self.hits = []                      # [pan, tilt, wall time, mode] of still non-faces that ended
        self._checks = []                   # [pan, tilt, frame time] of the last checks on spots
        self._next_save = 0.0
        if self.path and self.path.exists():
            try:
                self.hits = [(list(h) + ["head"])[:4] for h in json.loads(self.path.read_text()).get("hits", [])]
            except (OSError, ValueError):
                log.warning("static spots: could not read %s, starting empty", self.path)
        self._forget_old()

    def _near(self, pts, pan, tilt):
        return [p for p in pts if math.hypot(p[0] - pan, p[1] - tilt) <= self.radius]

    def is_static(self, pan, tilt):
        now = self._clock()
        with self._lock:
            return sum(1 for p in self._near(self.hits, pan, tilt)
                       if now - p[2] <= self.keep_s and p[3] == self.mode) >= self.min_hits

    def clear(self):
        """Forget every spot (the robot base moved / was carried: all places are somewhere else now)."""
        with self._lock:
            self.hits, self._checks = [], []
        self.save()

    def check_due(self, pan, tilt, t):
        """A detection on a spot: may it be checked now? (and note the check)."""
        with self._lock:
            self._checks = [c for c in self._checks if t - c[2] < self.check_s]
            if self._near(self._checks, pan, tilt):
                return False
            self._checks.append([pan, tilt, t])
            return True

    def note_ended(self, tr):
        """An ended track: if it never was a face and never moved (or was a rejected flicker), it is evidence."""
        if tr.is_face or tr.pos0 is None:
            return False
        path = getattr(tr, "path", None) or ()
        if tr.hits < 3:                     # a flicker: only one the face check looked at and turned down
            if tr.face_q is None or tr.face_q >= config.FACE_LOOK_MIN_QUALITY:
                return False
        else:
            dev = tr.spread()[1] if len(path) >= 3 else tr.max_dev
            if dev > self.max_dev:
                return False
        now = self._clock()
        if len(path) >= 3:                  # where it mostly was (the median), not where its first box was
            pans, tilts = sorted(p[1] for p in path), sorted(p[2] for p in path)
            at = (pans[len(pans) // 2], tilts[len(tilts) // 2])
        else:
            at = tr.pos0
        with self._lock:
            self.hits.append([round(at[0], 2), round(at[1], 2), now, self.mode])
            del self.hits[:-2000]
        if now >= self._next_save:
            self.save()
        return True

    def spots(self):
        """[(pan, tilt)] one point per static spot (for logs)."""
        out = []
        with self._lock:
            pts = list(self.hits)
        for p in pts:
            if p[3] != self.mode:
                continue
            if not any(math.hypot(p[0] - q[0], p[1] - q[1]) <= self.radius for q in out) and self.is_static(p[0], p[1]):
                out.append((round(p[0]), round(p[1])))
        return out

    def save(self):
        self._next_save = self._clock() + 30.0
        if not self.path:
            return
        self._forget_old()
        with self._lock:
            data = json.dumps({"hits": self.hits})
        tmp = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(data)
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("static spots: could not save (%s)", e)

    def _forget_old(self):
        now = self._clock()
        with self._lock:
            self.hits = [h for h in self.hits if now - h[2] <= self.keep_s]
