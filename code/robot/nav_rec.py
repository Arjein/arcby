"""
What navigation saw and did, kept to learn from later (docs/navigation.md "Recording a drive").

The floor check from the camera is tuned on the robot's own drives instead of photos taken by hand: a place the
camera called an obstacle that the robot later drove over with the sonar free and no bump was floor after all.

    rec = NavRecorder()                         # a session in DATA_DIR (robot/recorder.py: size limit, meta.json)
    rec = NavRecorder(rec=alive_recorder)       # ...or into a running session (alive.py's): close() leaves it open
    rec.step(percept, command, drive)           # every navigation step -> nav.jsonl.gz
    rec.frame(rgb, floor_view, t, sonar_free)   # every camera picture; kept while driving, 1 per NAV_REC_FRAME_S
    rec.close()

A session holds, besides meta.json:
    nav.jsonl.gz      every step while driving (1 per second while standing): yaw, sonar, head, v, turn, state, why
    floor.jsonl.gz    one per kept picture: the head, the sonar's free distance, the floor verdict (each ray's end and
                      hit, edge) - the picture is navframes/<t>.png (lossless: the floor model works on colours)
Local only, like every recording (see the privacy rules in README.md).
"""
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .recorder import Recorder

log = logging.getLogger("arcby")


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


class NavRecorder:
    def __init__(self, root=config.DATA_DIR, rec=None):
        self._own = rec is None
        self.rec = rec or Recorder(root, meta=dict(program="nav.py"))
        self.dir = Path(self.rec.dir)
        self.frames_dir = self.dir / "navframes"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.driving = False
        self.frames = 0
        self._next_frame_t = -1e9
        self._next_idle_t = -1e9

    def step(self, p, cmd, drive):
        """One navigation step. While standing still only one a second is kept (the page may stay open for long)."""
        self.driving = bool(drive)
        if not drive:
            if p.t < self._next_idle_t:
                return
            self._next_idle_t = p.t + 1.0
        s = p.safety or {}
        self.rec.write("nav", {
            "t": round(p.t, 3), "drive": bool(drive), "state": cmd.state, "note": cmd.note,
            "v": round(cmd.v, 3), "turn": round(cmd.turn, 1), "yaw": _r(p.yaw, 2), "yaw_rate": _r(p.yaw_rate),
            "sonar": _r(p.sonar), "sonar_t": _r(p.sonar_t, 3), "head": [_r(a) for a in p.head],
            "stop": bool(s.get("stop")), "reasons": list(s.get("reasons", []))})
        self.rec.tick()

    def frame(self, rgb, view, t, sonar_free):
        """A camera picture and its floor verdict; kept only while driving, at most 1 per NAV_REC_FRAME_S."""
        if not self.driving or view is None or t < self._next_frame_t or self.rec.failed:
            return None
        self._next_frame_t = t + config.NAV_REC_FRAME_S
        name = f"{t:.3f}.png"
        try:
            Image.fromarray(np.ascontiguousarray(rgb)).save(self.frames_dir / name, compress_level=1)
        except OSError as e:
            self.rec._fail("a nav picture", e)
            return None
        self.frames += 1
        self.rec.write("floor", {
            "t": round(t, 3), "frame": name, "pan": _r(view.pan), "tilt": _r(view.tilt),
            "sonar_free": None if sonar_free is None or not np.isfinite(sonar_free) else _r(sonar_free),
            "known": bool(view.known), "learnt": bool(view.learnt), "edge": bool(view.edge),
            "edge_cm": _r(view.edge_cm), "ex": [_r(x) for x in view.ex], "ey": [_r(y) for y in view.ey],
            "hit": [bool(h) for h in view.hit]})
        return name

    def close(self):
        if self._own:
            self.rec.close()
        log.info("nav: recorded %d pictures into %s", self.frames, self.dir)
