"""
The floor in front of the robot, from the head camera (docs/navigation.md "Seeing the floor").

The sonar is one narrow beam 7 cm above the floor: it misses cables, shoes, thin chair legs and everything to the
side. So the head looks DOWN (NAV_TILT_DEG) and every picture is split into "floor" and "not floor":

  1. what the floor looks like is learnt from the patch right in front of the robot (NAV_FLOOR_REF_CM), while the
     sonar says that patch is free - a colour histogram (r, g chromaticity: shading-proof) and a brightness one
     (appearance-based obstacle detection, Ulrich & Nourbakhsh 2000);
  2. every cell of the picture (NAV_FLOOR_CELL px) whose colour or brightness is rare in those histograms is "not
     floor";
  3. each ray (NAV_FLOOR_COLS columns) goes up the picture from the bottom; the first NAV_FLOOR_RUN non-floor cells
     in a row are the foot of an obstacle. The lens is NAV_CAM_H_CM above a flat floor, so the row of that foot
     says how far away it is (ground_points: the pixel's ray rotated by the head's tilt and pan, cut with the floor).

A ray that reaches NAV_SEE_CM (or the horizon) without a foot = floor free that far. A FLOOR EDGE (a rug, another
floor) is told apart from objects in the robot's path only (rays within NAV_EDGE_HALF_DEG of straight ahead - about
where the sonar listens): most of them hit along a straight line and the ground beyond is non-floor almost
everywhere (FloorView.edge, edge_rays). A tall box passes that test too: navigation crosses an edge only while the
sonar hears nothing that near, slowly, and learns the new floor as it drives onto it. The whole picture was too
strict - a rug is seen at its corner or at a slant most of the time (2026-11-04 drive: 14 of 112 edge views).

    seer = FloorSeer()
    view = seer.see(rgb, pan, tilt, sonar_free_cm=..., learn=True)   # FloorView in robot coordinates (cm)

Pure numpy, ~5 ms a picture on the Pi 4 (144 x 81 cells).
"""
import math
from dataclasses import dataclass

import numpy as np

from . import config
from .camera import focal_px

C_BINS = 24                 # chromaticity bins per axis (r, g in C_RANGE)
C_RANGE = (0.1, 0.6)
Y_BINS = 32                 # brightness bins (0..255)


def ground_points(xs, ys, width, height, pan_deg, tilt_deg, cam_h=None, cam_x=None, hfov_deg=None):
    """Where the rays through pixels (xs, ys) meet the floor, in robot coordinates (cm; x forward, y left).
    Returns (gx, gy, ok): ok = False for pixels at or above the horizon. Pinhole model (as camera.angles_of)."""
    cam_h = config.NAV_CAM_H_CM if cam_h is None else cam_h
    cam_x = config.NAV_CAM_X_CM if cam_x is None else cam_x
    f = focal_px(width, config.CAMERA_HFOV_DEG if hfov_deg is None else hfov_deg)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    fwd = np.ones(np.broadcast(xs, ys).shape)
    left = -(xs - width / 2) / f
    up = (height / 2 - ys) / f
    t, p = math.radians(tilt_deg), math.radians(pan_deg)
    x1 = fwd * math.cos(t) - up * math.sin(t)           # tilt: about the left axis, + = up
    z1 = fwd * math.sin(t) + up * math.cos(t)
    x2 = x1 * math.cos(p) - left * math.sin(p)          # pan: about the vertical, + = left
    y2 = x1 * math.sin(p) + left * math.cos(p)
    ok = z1 < -1e-6
    s = np.where(ok, cam_h / np.where(ok, -z1, 1.0), 0.0)
    return cam_x + s * x2, s * y2, ok


def pixel_of_ground(gx, gy, width, height, pan_deg, tilt_deg, cam_h=None, cam_x=None):
    """Inverse of ground_points: the pixel (x, y) where floor point (gx, gy) shows, or None if behind the camera."""
    cam_h = config.NAV_CAM_H_CM if cam_h is None else cam_h
    cam_x = config.NAV_CAM_X_CM if cam_x is None else cam_x
    f = focal_px(width)
    t, p = math.radians(tilt_deg), math.radians(pan_deg)
    dx, dy, dz = gx - cam_x, gy, -cam_h
    x1 = dx * math.cos(p) + dy * math.sin(p)            # undo pan
    left = -dx * math.sin(p) + dy * math.cos(p)
    fwd = x1 * math.cos(t) + dz * math.sin(t)           # undo tilt
    up = -x1 * math.sin(t) + dz * math.cos(t)
    if fwd <= 1e-6:
        return None
    return width / 2 - left / fwd * f, height / 2 - up / fwd * f


@dataclass
class FloorView:
    """One picture's verdict, robot coordinates (cm) at the time of the picture."""
    t: float
    pan: float
    tilt: float
    ox: float               # where the rays start (the lens' foot on the floor)
    oy: float
    ex: np.ndarray          # each ray's end: an obstacle's foot (hit) or free up to there
    ey: np.ndarray
    hit: np.ndarray         # bool per ray
    edge: bool = False      # the hits look like a floor edge (rug), not objects
    edge_cm: float = None   # distance of that edge from the lens' foot
    edge_rays: np.ndarray = None    # bool per ray: the hits that make the edge (the others stay obstacles)
    learnt: bool = False    # the floor model took this picture's reference patch
    known: bool = True      # False until the floor model has learnt anything (then every ray is "unknown")

    @property
    def bearings(self):
        """Each ray's direction from the robot's axle (deg, + = left)."""
        return np.degrees(np.arctan2(self.ey, self.ex))

    @property
    def ranges(self):
        """Each ray's length from the lens' foot (cm)."""
        return np.hypot(self.ex - self.ox, self.ey - self.oy)


def _blur(h, axes):
    """Spread every bin a little to its neighbours (tolerance for noise and light changes)."""
    out = h.copy()
    for ax in axes:
        out = out + 0.5 * (np.roll(out, 1, ax) + np.roll(out, -1, ax))
    return out


class FloorSeer:
    def __init__(self, width=None, height=None, cell=None):
        self.width, self.height = config.CAMERA_LORES if width is None else (width, height)
        self.cell = cell or config.NAV_FLOOR_CELL
        self.chroma = None              # learnt histograms (normalised to peak 1), None = nothing learnt yet
        self.bright = None
        self.learnt_n = 0
        self._geo_key = None
        nx, ny = self.width // self.cell, self.height // self.cell
        self.nx, self.ny = nx, ny
        self.cx = (np.arange(nx) + 0.5) * self.cell
        self.cy = (np.arange(ny) + 0.5) * self.cell
        self.cols = np.linspace(0, nx - 1, config.NAV_FLOOR_COLS).round().astype(int)

    def forget(self):
        self.chroma = self.bright = None
        self.learnt_n = 0

    # ---- pixels -> cells ----
    def _cells(self, rgb):
        c, nx, ny = self.cell, self.nx, self.ny
        a = rgb[:ny * c, :nx * c, :3].astype(np.float32).reshape(ny, c, nx, c, 3).mean(axis=(1, 3))
        s = a.sum(axis=2) + 1.0
        r, g = a[:, :, 0] / s, a[:, :, 1] / s
        y = a.mean(axis=2)
        lo, hi = C_RANGE
        ri = np.clip(((r - lo) / (hi - lo) * C_BINS).astype(int), 0, C_BINS - 1)
        gi = np.clip(((g - lo) / (hi - lo) * C_BINS).astype(int), 0, C_BINS - 1)
        yi = np.clip((y / 256.0 * Y_BINS).astype(int), 0, Y_BINS - 1)
        return ri, gi, yi

    def _geometry(self, pan, tilt):
        key = (round(pan, 1), round(tilt, 1))
        if key != self._geo_key:
            xs, ys = np.meshgrid(self.cx, self.cy)
            self._gx, self._gy, self._ok = ground_points(xs, ys, self.width, self.height, pan, tilt)
            self._geo_key = key
        return self._gx, self._gy, self._ok

    # ---- the model ----
    def floor_mask(self, ri, gi, yi):
        """True where the cell looks like the learnt floor (all False before anything was learnt)."""
        if self.chroma is None:
            return np.zeros(ri.shape, bool)
        return (self.chroma[ri, gi] >= config.NAV_FLOOR_MIN) & (self.bright[yi] >= config.NAV_FLOOR_MIN)

    def _learn(self, ri, gi, yi, sel, rate):
        ch = np.zeros((C_BINS, C_BINS))
        np.add.at(ch, (ri[sel], gi[sel]), 1.0)
        yh = np.zeros(Y_BINS)
        np.add.at(yh, yi[sel], 1.0)
        ch, yh = _blur(ch, (0, 1)), _blur(yh, (0,))
        ch /= ch.max() or 1.0
        yh /= yh.max() or 1.0
        if self.chroma is None:
            self.chroma, self.bright = ch, yh
        else:
            self.chroma = (1 - rate) * self.chroma + rate * ch
            self.bright = (1 - rate) * self.bright + rate * yh
            self.chroma /= self.chroma.max() or 1.0
            self.bright /= self.bright.max() or 1.0
        self.learnt_n += 1

    def ref_patch(self, gx, gy, ok):
        near, far = config.NAV_FLOOR_REF_CM
        x0 = config.NAV_FRONT_CM
        return ok & (gx >= x0 + near) & (gx <= x0 + far) & (np.abs(gy) <= config.NAV_FLOOR_REF_HALF_CM)

    # ---- one picture ----
    def see(self, rgb, pan, tilt, t=0.0, sonar_free_cm=None, learn=True, force_learn=False):
        """rgb = the small stream (H x W x 3) taken with the head at (pan, tilt). sonar_free_cm = the sonar's free
        distance ahead of the robot's front (None = unknown: no learning unless force_learn). force_learn = learn
        the reference patch whatever it looks like (start, crossing a floor edge onto a new floor)."""
        ri, gi, yi = self._cells(rgb)
        gx, gy, ok = self._geometry(pan, tilt)
        # cells inside or under the robot itself never count (its own front can show at the bottom)
        ok = ok & (gx > config.NAV_FRONT_CM + 1.0)
        ref = self.ref_patch(gx, gy, ok)
        learnt = False
        if learn and ref.sum() >= 6:
            free_enough = sonar_free_cm is not None and sonar_free_cm >= config.NAV_FLOOR_REF_CM[1] + 5
            if force_learn or self.chroma is None and free_enough:
                self._learn(ri, gi, yi, ref, 1.0 if self.chroma is None else config.NAV_FLOOR_RATE)
                learnt = True
            elif free_enough:
                unlike = 1.0 - self.floor_mask(ri, gi, yi)[ref].mean()
                if unlike <= config.NAV_FLOOR_MAX_NEW:
                    self._learn(ri, gi, yi, ref, config.NAV_FLOOR_RATE)
                    learnt = True
        return self._rays(t, pan, tilt, self.floor_mask(ri, gi, yi), gx, gy, ok, learnt)

    def _rays(self, t, pan, tilt, floor, gx, gy, ok, learnt):
        ox = config.NAV_CAM_X_CM
        known = self.chroma is not None
        see = config.NAV_SEE_CM
        n = len(self.cols)
        ex, ey, hit = np.zeros(n), np.zeros(n), np.zeros(n, bool)
        beyond = []                     # per hit ray: share of non-floor cells beyond the foot (edge test)
        for k, c in enumerate(self.cols):
            col_ok = ok[:, c]
            rng = np.hypot(gx[:, c] - ox, gy[:, c])
            usable = col_ok & (rng <= see)
            rows = np.nonzero(usable)[0][::-1]          # bottom (near) to top (far)
            if len(rows) == 0:
                ex[k], ey[k] = ox, 0.0
                continue
            last = rows[0]
            run, foot = 0, None
            for r in rows:
                if not known or not floor[r, c]:
                    run += 1
                    if run == 1:
                        first = r
                    # long enough in cells AND on the floor: near the robot 2 cells are ~0.6 cm - one dark tuft
                    # of a shaggy rug (2026-11-04); far away one cell is already several cm
                    if run >= config.NAV_FLOOR_RUN and rng[r] - rng[first] >= config.NAV_FLOOR_RUN_CM:
                        foot = first
                        break
                else:
                    run = 0
                    last = r
            if foot is not None:
                ex[k], ey[k], hit[k] = gx[foot, c], gy[foot, c], True
                past = rows[rows <= foot]
                beyond.append(1.0 - floor[past, c].mean() if len(past) else 1.0)
            else:
                # free up to the farthest usable row (or NAV_SEE_CM along the ray when the horizon is nearer)
                ex[k], ey[k] = gx[last, c], gy[last, c]
        view = FloorView(t, pan, tilt, ox, 0.0, ex, ey, hit, learnt=learnt, known=known)
        if known and hit.any():
            self._edge(view, np.array(beyond))
        return view

    def _edge(self, view, beyond):
        """Is what the rays hit in the robot's path a floor edge (straight line, non-floor beyond)? Sets view.edge."""
        ahead = np.abs(np.degrees(np.arctan2(view.ey, view.ex))) <= config.NAV_EDGE_HALF_DEG
        hit = view.hit
        if ahead.sum() < config.NAV_EDGE_MIN_RAYS or hit[ahead].mean() < config.NAV_EDGE_SHARE:
            return
        mine = ahead & hit
        fill = beyond[mine[hit]]                    # beyond[] holds one value per hit ray, in ray order
        if len(fill) < 3 or float(fill.mean()) < config.NAV_EDGE_FILL:
            return
        px, py = view.ex[mine], view.ey[mine]
        cx, cy = px.mean(), py.mean()
        _, _, vt = np.linalg.svd(np.stack([px - cx, py - cy], axis=1), full_matrices=False)
        off = np.abs((px - cx) * vt[1, 0] + (py - cy) * vt[1, 1])             # distance from the fitted line
        nx, ny = vt[1]
        slant = float(np.degrees(np.arccos(min(1.0, abs(nx)))))    # 0 = the line square across the robot's path
        if np.percentile(off, 80) <= config.NAV_EDGE_SPREAD_CM / 2 and slant <= config.NAV_EDGE_MAX_SLANT_DEG:
            view.edge = True
            view.edge_rays = mine
            view.edge_cm = float(abs((cx - view.ox) * nx + (cy - view.oy) * ny))   # lens foot -> the line
