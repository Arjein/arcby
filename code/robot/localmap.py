"""
What navigation remembers around the robot (docs/navigation.md "Memory").

No wheel encoders: the robot's position is dead reckoning - the IMU's heading (good) and speed x time (an estimate),
so the memory is short and local: obstacle points are forgotten after NAV_MEMORY_S or NAV_MEMORY_CM.

Coordinates: "odom" = where the robot started (cm, x forward at start, y left); the robot frame = the axle middle
now, x forward. Three kinds of evidence, each cleared only by its own sensor (the sonar cannot see a cable on the
floor, the camera looking down cannot see a table edge 7 cm up):

  cam    obstacle feet from the floor pictures (robot/floor.py); a newer picture that sees floor there clears them
  sonar  echoes (an arc across the beam); a newer echo farther away clears them
  bump   where the robot hit something neither sensor saw (glass, a dark corner); only time clears them

Besides the obstacles it keeps the recent floor pictures' FREE fans ("seen"): navigation drives only through ground
the camera saw free a moment ago - unknown ground counts as blocked.

    m = LocalMap()
    m.move(yaw_deg, dist_cm)            # every step: IMU heading + estimated distance driven
    m.add_view(view, t)                 # a FloorView (robot coordinates when taken)
    m.add_sonar(range_cm or None, t)
    m.way(rel_deg, t, seen_s)           # free way (gap ahead of the front, cm) after turning rel_deg on the spot
    m.spin_gap()                        # room around the spin circle
"""
import math
from collections import deque

import numpy as np

from . import config

CAM, SONAR, BUMP = 0, 1, 2
STEP_CM = 5.0


class LocalMap:
    def __init__(self):
        self.x = self.y = 0.0           # odom pose
        self.h = 0.0                    # heading, deg (the IMU's yaw)
        self.px = np.zeros(0)
        self.py = np.zeros(0)
        self.pt = np.zeros(0)
        self.ps = np.zeros(0, int)
        self.views = deque()            # (t, ox, oy, bearings sorted (rad, odom), free ranges)
        self.visits = {}                # (i, j) 20 cm cells -> times entered (exploring: prefer new ground)
        self._cell = None

    # ---- pose ----
    def move(self, yaw_deg, dist_cm):
        """Dead reckoning: new heading from the IMU, dist_cm driven since the last call (- = backwards)."""
        mid = math.radians((self.h + yaw_deg) / 2)
        self.x += dist_cm * math.cos(mid)
        self.y += dist_cm * math.sin(mid)
        self.h = yaw_deg
        cell = (int(math.floor(self.x / 20)), int(math.floor(self.y / 20)))
        if cell != self._cell:
            self.visits[cell] = self.visits.get(cell, 0) + 1
            self._cell = cell

    def to_odom(self, rx, ry):
        c, s = math.cos(math.radians(self.h)), math.sin(math.radians(self.h))
        rx, ry = np.asarray(rx, float), np.asarray(ry, float)
        return self.x + rx * c - ry * s, self.y + rx * s + ry * c

    def to_robot(self, ox, oy):
        c, s = math.cos(math.radians(self.h)), math.sin(math.radians(self.h))
        dx, dy = np.asarray(ox, float) - self.x, np.asarray(oy, float) - self.y
        return dx * c + dy * s, -dx * s + dy * c

    # ---- obstacles ----
    def _add(self, rx, ry, t, src):
        ox, oy = self.to_odom(rx, ry)
        ox, oy = np.atleast_1d(ox), np.atleast_1d(oy)
        self.px = np.concatenate([self.px, ox])
        self.py = np.concatenate([self.py, oy])
        self.pt = np.concatenate([self.pt, np.full(len(ox), float(t))])
        self.ps = np.concatenate([self.ps, np.full(len(ox), src, int)])

    def _drop(self, mask):
        keep = ~mask
        self.px, self.py, self.pt, self.ps = self.px[keep], self.py[keep], self.pt[keep], self.ps[keep]

    def add_points(self, rx, ry, t, src=CAM):
        if len(np.atleast_1d(rx)):
            self._add(rx, ry, t, src)

    def forget_band(self, near_cm, far_cm, half_w):
        """Drop camera points between near_cm and far_cm ahead of the front, within half_w of the centre line: a
        floor edge being crossed (pictures that did not pass the edge test left it here as obstacles)."""
        if not len(self.px):
            return
        rx, ry = self.to_robot(self.px, self.py)
        u = rx - config.NAV_FRONT_CM
        self._drop((self.ps == CAM) & (u >= near_cm) & (u <= far_cm) & (np.abs(ry) <= half_w))

    def add_bump(self, t):
        """Something ahead that no sensor saw: an arc just in front of the robot."""
        a = np.radians(np.linspace(-35, 35, 8))
        r = config.NAV_FRONT_CM + 2.0
        self._add(r * np.cos(a), r * np.sin(a), t, BUMP)

    def add_view(self, view, t):
        """A floor picture: forget camera points it sees as floor, remember its obstacle feet and its free fan."""
        if not view.known:
            return
        b = np.arctan2(view.ey - view.oy, view.ex - view.ox)
        r = view.ranges
        order = np.argsort(b)
        b, r, hit = b[order], r[order], view.hit[order]
        free_r = np.where(hit, np.maximum(0.0, r - 2.0), r)
        if len(self.px):
            rx, ry = self.to_robot(self.px, self.py)
            pb = np.arctan2(ry - view.oy, rx - view.ox)
            pr = np.hypot(rx - view.ox, ry - view.oy)
            inside = (pb >= b[0]) & (pb <= b[-1])
            lim = np.minimum(np.interp(pb, b, free_r), _neighbour_min(pb, b, free_r))
            self._drop((self.ps == CAM) & inside & (pr < lim - 3.0))
        solid = view.hit & ~view.edge_rays if view.edge and view.edge_rays is not None else view.hit
        if view.edge and view.edge_rays is None:
            solid = np.zeros_like(view.hit)          # an edge view without its rays (old recordings, tests)
        if solid.any():
            self.add_points(view.ex[solid], view.ey[solid], t, CAM)
        # the free fan, kept in odom coordinates
        ox, oy = self.to_odom(view.ox, view.oy)
        self.views.append((t, float(ox), float(oy), b + math.radians(self.h), free_r))

    def add_sonar(self, range_cm, t):
        """An echo at range_cm from the sonar face (None = nothing within NAV_SONAR_MAX_CM)."""
        sx = config.NAV_SONAR_X_CM
        half = math.radians(config.NAV_SONAR_HALF_DEG)
        r = config.NAV_SONAR_MAX_CM if range_cm is None else float(range_cm)
        if len(self.px):
            rx, ry = self.to_robot(self.px, self.py)
            pb = np.arctan2(ry, rx - sx)
            pr = np.hypot(rx - sx, ry)
            self._drop((self.ps == SONAR) & (np.abs(pb) <= half) & (pr < r - 5.0))
        if range_cm is not None and range_cm < config.NAV_SONAR_MAX_CM:
            a = np.linspace(-half, half, 5)
            self._add(sx + r * np.cos(a), r * np.sin(a), t, SONAR)

    def forget(self, t):
        """Drop old or far points and old views."""
        if len(self.px):
            far = np.hypot(self.px - self.x, self.py - self.y) > config.NAV_MEMORY_CM
            self._drop(far | (t - self.pt > config.NAV_MEMORY_S))
        while self.views and t - self.views[0][0] > config.NAV_MEMORY_S / 3:
            self.views.popleft()

    # ---- questions ----
    def spin_gap(self):
        """Room (cm) between the spin circle and the nearest remembered point (inf if none)."""
        if not len(self.px):
            return math.inf
        d = np.hypot(self.px - self.x, self.py - self.y)
        return float(d.min()) - config.NAV_SPIN_RADIUS_CM

    def obstacle_gap(self, rel_deg, half_w=None, cap=None):
        """Free way (cm ahead of the front) through the remembered points, after turning rel_deg on the spot."""
        half_w = config.NAV_HALF_WIDTH_CM + config.NAV_MARGIN_CM if half_w is None else half_w
        cap = config.NAV_SEE_CM if cap is None else cap
        if not len(self.px):
            return cap
        rx, ry = self.to_robot(self.px, self.py)
        a = math.radians(rel_deg)
        u = rx * math.cos(a) + ry * math.sin(a)
        w = -rx * math.sin(a) + ry * math.cos(a)
        front = config.NAV_FRONT_CM
        sel = (np.abs(w) <= half_w) & (u > front - 1.0)
        if not sel.any():
            return cap
        return float(max(0.0, min(cap, (u[sel] - front).min())))

    def covered(self, qx, qy, t, max_age):
        """Which odom points lie inside a free fan seen in the last max_age seconds."""
        qx, qy = np.asarray(qx, float), np.asarray(qy, float)
        # the robot's own front strip (too close for the picture's bottom row) is inside what a turn on the spot
        # sweeps anyway: counted as seen, so turning does not make the way ahead unknown
        ok = np.hypot(qx - self.x, qy - self.y) <= config.NAV_SPIN_RADIUS_CM + 4.0
        for vt, ox, oy, b, free_r in reversed(self.views):
            if t - vt > max_age:
                break
            if len(b) < 2:
                continue
            pb = np.arctan2(qy - oy, qx - ox)
            pr = np.hypot(qx - ox, qy - oy)
            pb = b[0] + np.mod(pb - b[0], 2 * math.pi)          # same turn as the fan
            inside = pb <= b[-1]
            lim = np.minimum(np.interp(pb, b, free_r), _neighbour_min(pb, b, free_r))
            ok |= inside & (pr <= lim)
            if ok.all():
                break
        return ok

    def seen_gap(self, rel_deg, t, max_age, half_w=None, cap=None):
        """How far (cm ahead of the front) the way after turning rel_deg was seen free (centre and both sides)."""
        return float(self.seen_gaps([rel_deg], t, max_age, half_w, cap)[0])

    def seen_gaps(self, rels, t, max_age, half_w=None, cap=None):
        half_w = config.NAV_HALF_WIDTH_CM if half_w is None else half_w
        cap = config.NAV_SEE_CM if cap is None else cap
        steps = np.arange(0.0, cap + 0.1, STEP_CM)
        rels = np.radians(np.asarray(rels, float))
        u = config.NAV_FRONT_CM + steps                     # along the way
        w = np.array([0.0, half_w, -half_w])                # centre, left, right edge
        a = rels[:, None, None]
        rx = u[None, :, None] * np.cos(a) - w[None, None, :] * np.sin(a)
        ry = u[None, :, None] * np.sin(a) + w[None, None, :] * np.cos(a)
        qx, qy = self.to_odom(rx, ry)
        ok = self.covered(qx, qy, t, max_age).all(axis=2)    # candidates x steps
        out = np.empty(len(rels))
        for i, row in enumerate(ok):
            bad = np.nonzero(~row)[0]
            out[i] = cap if len(bad) == 0 else steps[bad[0]] - STEP_CM if bad[0] else 0.0
        return np.maximum(out, 0.0)

    def way(self, rel_deg, t, max_age, cap=None):
        """Free AND seen way after turning rel_deg (cm ahead of the front)."""
        return min(self.obstacle_gap(rel_deg, cap=cap), self.seen_gap(rel_deg, t, max_age, cap=cap))

    def ways(self, rels, t, max_age, cap=None):
        seen = self.seen_gaps(rels, t, max_age, cap=cap)
        return np.array([min(self.obstacle_gap(r, cap=cap), s) for r, s in zip(rels, seen)])

    def visited(self, rel_deg, ahead_cm=60.0):
        """How often the cell ahead_cm away in that direction was entered."""
        a = math.radians(self.h + rel_deg)
        x, y = self.x + ahead_cm * math.cos(a), self.y + ahead_cm * math.sin(a)
        return self.visits.get((int(math.floor(x / 20)), int(math.floor(y / 20))), 0)


def _neighbour_min(pb, b, r):
    """The shorter of the two rays either side of each bearing (between two rays the floor is only known up to the
    shorter one: an obstacle between them shows as a foot on at least one)."""
    i = np.clip(np.searchsorted(b, pb), 1, len(b) - 1)
    return np.minimum(r[i - 1], r[i])
