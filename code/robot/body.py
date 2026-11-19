"""
Bodies: people, not just faces (docs/vision.md "Bodies"). A pose model (YOLO26n-pose, 17 COCO keypoints) finds whole or
partial people in the small stream, so the robot

  - knows someone is there when it sees only legs (it sits 14 cm above the floor) or an arm coming in at the edge,
    and where their face should be (face_hint: the head keypoints, else up from the shoulders / hips / knees,
    else past the picture's edge) - attention looks there and the face detector does the rest;
  - links each body to the face track on its head (BodyTrack.face / .person);
  - sees hand signs (robot/gestures.py): "gel gel" (beckoning) -> "beckoned", a raised still hand -> "stop_sign";
  - sees waving (a forearm swinging above the elbow) -> a "waved" event. The model runs a few times a second, too
    slow for a 2-3 Hz wave; while it sees a raised forearm, the camera loop also watches a patch round that wrist
    on every frame (15 fps: which way the moving pixels go) - that is what counts the swings (WristWatch).

The model runs in its own thread (BodyWorker, ~160 ms at 320x192 on the Pi 4 next to alive.py), a few times a
second - often while no face is in view, seldom while it follows one. Positions are head angles like the face
tracker's (pan + = left, tilt + = up), so the head turning between frames does not break a track.

    bodies = BodySense(load_pose())
    events = bodies.process(frame, pose_at, speed, face_tracks)     # each camera frame (cheap when nothing is due)
    bodies.seekable(t)                                              # bodies whose face it should look for
"""
import logging
import math
import os
import queue
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import camera, config, gestures

log = logging.getLogger(__name__)

# COCO keypoints (left / right = the person's own)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
HEAD = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
ARMS = ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST))


@dataclass
class Body:
    box: np.ndarray         # x1, y1, x2, y2 in frame pixels
    score: float
    kps: np.ndarray         # 17 x 3: x, y (frame pixels), visibility 0..1

    def has(self, i, min_vis=None):
        return self.kps[i, 2] >= (config.BODY_KP_MIN if min_vis is None else min_vis)

    @property
    def head_visible(self):
        return any(self.has(i) for i in HEAD)


# ---------- the model ----------

def letterbox(rgb, size):
    """Frame -> model input: shrunk to fit size = (w, h), centred on grey (114) like the Ultralytics training.
    Returns (1 x 3 x h x w float32, scale, (pad x, pad y))."""
    w, h = size
    fh, fw = rgb.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = max(1, round(fw * s)), max(1, round(fh * s))
    img = rgb if (nw, nh) == (fw, fh) else np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    out = np.full((h, w, 3), 114, np.uint8)
    px, py = (w - nw) // 2, (h - nh) // 2
    out[py:py + nh, px:px + nw] = img
    return out.transpose(2, 0, 1)[None].astype(np.float32) / 255.0, s, (px, py)


def nms(boxes, scores, iou):
    order = np.argsort(scores)[::-1]
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= iou]
    return keep


def decode(out, scale, pad, min_score=None, iou=None):
    """YOLO pose output (1 x 56 x N: cx, cy, w, h, score, 17 x (x, y, visibility)) -> [Body] in frame pixels."""
    min_score = config.BODY_MIN_SCORE if min_score is None else min_score
    iou = config.BODY_NMS_IOU if iou is None else iou
    rows = np.asarray(out)[0].T
    rows = rows[rows[:, 4] >= min_score]
    if not len(rows):
        return []
    px, py = pad
    cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    boxes = np.stack([cx - w / 2 - px, cy - h / 2 - py, cx + w / 2 - px, cy + h / 2 - py], axis=1) / scale
    kps = rows[:, 5:56].reshape(-1, 17, 3).copy()
    kps[:, :, 0] = (kps[:, :, 0] - px) / scale
    kps[:, :, 1] = (kps[:, :, 1] - py) / scale
    return [Body(boxes[i], float(rows[i, 4]), kps[i]) for i in nms(boxes, rows[:, 4], iou)]


class PoseDetector:
    def __init__(self, session, size=None):
        self.session = session
        self.size = tuple(size or config.BODY_INPUT)
        self.input = session.get_inputs()[0].name

    def detect(self, rgb):
        h, w = rgb.shape[:2]
        x, s, pad = letterbox(rgb, self.size)
        bodies = decode(self.session.run(None, {self.input: x})[0], s, pad)
        for b in bodies:                        # keep everything inside the frame
            b.box[[0, 2]] = np.clip(b.box[[0, 2]], 0, w)
            b.box[[1, 3]] = np.clip(b.box[[1, 3]], 0, h)
        return bodies


def crop_for_face(box, lores_size, size=None, faces_tall=None, above=None):
    """Where to look for the body under a small (far) face: a box in small-stream pixels with the model input's
    shape (no letterbox waste), the face near its top centre, `faces_tall` face heights tall (a seated person's
    knees, a standing one's thighs) - cut from the big stream (4x the pixels) it gives the model the person ~2-3x
    larger than the whole small frame does. Moved inside the frame, shrunk only if larger than it."""
    mw, mh = size or config.BODY_INPUT
    faces_tall = config.BODY_CROP_FACES_TALL if faces_tall is None else faces_tall
    above = config.BODY_CROP_ABOVE if above is None else above
    fw, fh = lores_size
    x1, y1, x2, y2 = (float(v) for v in box)
    h = max(faces_tall * (y2 - y1), config.BODY_CROP_MIN_PX)
    w = h * mw / mh
    if w > fw:                                  # never wider than the frame (nor taller)
        w, h = fw, fw * mh / mw
    if h > fh:
        w, h = fh * mw / mh, fh
    left = (x1 + x2) / 2 - w / 2
    top = y1 - above * (y2 - y1)
    left = min(max(left, 0.0), fw - w)
    top = min(max(top, 0.0), fh - h)
    return np.array([left, top, left + w, top + h])


def to_frame(bodies, origin, scale):
    """Bodies found in a big-stream crop -> small-stream pixels (crop pixel + origin = big pixel = small x scale)."""
    ox, oy = origin
    for b in bodies:
        b.box[[0, 2]] = (b.box[[0, 2]] + ox) / scale
        b.box[[1, 3]] = (b.box[[1, 3]] + oy) / scale
        b.kps[:, 0] = (b.kps[:, 0] + ox) / scale
        b.kps[:, 1] = (b.kps[:, 1] + oy) / scale
    return bodies


def load_pose(models_dir=None, threads=None):
    """The pose model, or None when its file is missing (the robot then runs on faces alone)."""
    from .faces import MODELS, _session, model_path
    path = model_path("yolo26n_pose", models_dir)
    if not os.path.exists(path):
        log.warning("body: %s missing - no bodies (make it with get_models.py --derive on the Mac)", path)
        return None
    w, h = MODELS["yolo26n_pose"]["input"]
    return PoseDetector(_session(path, threads or config.BODY_THREADS), size=(w, h))


# ---------- worker thread ----------

@dataclass
class BodyJob:
    t: float
    pose: tuple             # head (pan, tilt) when the frame was taken
    rgb: np.ndarray         # the small stream - or a big-stream crop (a copy: the frame's buffers go back)
    bodies: list = None
    ms: float = 0.0
    origin: tuple = None    # a crop: its top-left in big-stream pixels...
    scale: float = 1.0      # ...big pixels per small pixel...
    size: tuple = None      # ...and the small stream's (w, h) - results are mapped back into it
    face: int = None        # the face track it was cut under


class BodyWorker:
    """Runs the pose model in its own thread, one frame at a time (a frame handed over while one is running is
    refused: the newest frame wins next time)."""

    def __init__(self, detector):
        self.detector = detector
        self._todo, self._done = queue.Queue(), queue.Queue()
        self.busy = False
        self._thread = threading.Thread(target=self._run, name="body", daemon=True)
        self._thread.start()

    def submit(self, job):
        if self.busy:
            return False
        self.busy = True
        self._todo.put(job)
        return True

    def done(self):
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                break
        if out:
            self.busy = False
        return out

    def close(self):
        self._todo.put(None)
        self._thread.join(timeout=5)

    def _run(self):
        import time
        while True:
            job = self._todo.get()
            if job is None:
                return
            t0 = time.perf_counter()
            try:
                job.bodies = self.detector.detect(job.rgb)
            except Exception:  # noqa: BLE001 - one bad frame must not stop body sensing
                log.exception("body: pose model failed")
                job.bodies = []
            job.ms = (time.perf_counter() - t0) * 1000
            self._done.put(job)


class InlineBodies:
    """BodyWorker's interface, detecting at once (tests, photo replays)."""

    def __init__(self, detector):
        self.detector, self._done, self.busy = detector, [], False

    def submit(self, job):
        job.bodies = self.detector.detect(job.rgb)
        self._done.append(job)
        return True

    def done(self):
        out, self._done = self._done, []
        return out

    def close(self):
        pass


# ---------- geometry: pixels -> head angles, where the face should be ----------

def angles(xy, width, height, pose):
    """Pixel points (N x 2) -> head angles (N x 2: pan, tilt) that would centre them (camera.head_target)."""
    f = camera.focal_px(width)
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    pan = pose[0] - np.degrees(np.arctan((xy[:, 0] - width / 2) / f))
    tilt = pose[1] + np.degrees(np.arctan((height / 2 - xy[:, 1]) / f))
    return np.stack([pan, tilt], axis=1)


def _mid(body, ang, *idx):
    pts = [ang[i] for i in idx if body.has(i)]
    return np.mean(pts, axis=0) if pts else None


def _dist(a, b):
    return float(np.hypot(*(np.asarray(a) - np.asarray(b))))


def face_hint(body, width, height, pose):
    """Where this person's face should be, in head angles: (pan, tilt or None, how). None tilt = no idea how high
    (look at the usual face height). how = "head" (head keypoints seen) / "shoulders" / "hips" / "knees" /
    "top" (cut off at the top of the picture) / "edge" (cut off at a side: an arm coming in) / None (no idea).

    Body proportions (an adult standing, as fractions of the height H): the face centre sits ~0.11 H above the
    shoulder line, ~0.42 H above the hips, ~0.67 H above the knees; shoulder width ~0.26 H, thigh (hip -> knee)
    ~0.25 H, shin (knee -> ankle) ~0.25 H. From 14 cm above the floor the angles are only a rough guide - the head
    moves there and the face detector takes over - so close enough is enough."""
    ang = angles(body.kps[:, :2], width, height, pose)
    head = _mid(body, ang, *HEAD)
    if head is not None:
        return float(head[0]), float(head[1]), "head"
    sh, hips, knees = _mid(body, ang, L_SHOULDER, R_SHOULDER), _mid(body, ang, L_HIP, R_HIP), \
        _mid(body, ang, L_KNEE, R_KNEE)
    ankles = _mid(body, ang, L_ANKLE, R_ANKLE)
    m = config.BODY_EDGE_PX
    x1, y1, x2, _ = body.box
    at_left, at_right, at_top = x1 <= m, x2 >= width - m, y1 <= m
    hint = None
    if sh is not None:
        shw = _dist(ang[L_SHOULDER], ang[R_SHOULDER]) if body.has(L_SHOULDER) and body.has(R_SHOULDER) else 0.0
        torso = _dist(sh, hips) if hips is not None else 0.0
        up = max(0.42 * shw, 0.35 * torso, config.BODY_HINT_MIN_UP_DEG)     # 0.11 H over 0.26 H / 0.31 H
        hint = (sh[0], sh[1] + up, "shoulders")
    elif hips is not None:
        thigh = _dist(hips, knees) if knees is not None else 0.0
        up = max(1.7 * thigh, 2 * config.BODY_HINT_MIN_UP_DEG)               # 0.42 H over 0.25 H
        hint = (hips[0], hips[1] + up, "hips")
    elif knees is not None:
        shin = _dist(knees, ankles) if ankles is not None else 0.0
        up = max(2.7 * shin, 3 * config.BODY_HINT_MIN_UP_DEG)                # 0.67 H over 0.25 H
        hint = (knees[0], knees[1] + up, "knees")
    if at_top:                                  # cut off at the top: the face is above the picture at least
        top = angles([[(x1 + x2) / 2, 0.0]], width, height, pose)[0]
        above = top[1] + config.BODY_HINT_MIN_UP_DEG
        if hint is None or hint[1] < above:
            hint = (hint[0] if hint else float(top[0]), above, "top" if hint is None else hint[2])
    if hint is None and (at_left or at_right):  # only an arm / a side in view: the rest is past that edge
        x = 0.0 if at_left else float(width)
        edge = angles([[x, (body.box[1] + body.box[3]) / 2]], width, height, pose)[0]
        out = config.BODY_EDGE_DEG * (1 if at_left else -1)     # pan + = left
        return float(edge[0] + out), None, "edge"
    if hint is None:
        return None
    return float(hint[0]), float(hint[1]), hint[2]


# ---------- waving ----------

class WaveDetector:
    """A wave = a forearm held up (forearm_up) swinging sideways: the wrist's sideways offset from the
    elbow, in shoulder widths, changes direction BODY_WAVE_TURNS times with a spread of BODY_WAVE_AMP within
    BODY_WAVE_WINDOW_S. Relative to the person's own elbow, so walking or the head moving does not look like
    waving. The model runs only a few times a second - a 2-3 Hz wave is under-sampled, but the samples still
    jump from side to side, which is what is counted."""

    def __init__(self):
        self.hist = (deque(), deque())          # per arm: (t, offset) while the arm is up

    def update(self, body, t):
        """Feed one detection; True when this body is waving now."""
        scale = self._scale(body)
        waving = False
        for arm, hist in zip(ARMS, self.hist):
            _, elbow, wrist = arm
            while hist and t - hist[0][0] > config.BODY_WAVE_WINDOW_S:
                hist.popleft()
            if scale is None or not (body.has(elbow) and body.has(wrist)):
                continue
            ex, ey = body.kps[elbow, :2]
            wx, wy = body.kps[wrist, :2]
            if wy >= ey:                        # wrist below the elbow: the arm is down
                hist.clear()
                continue
            if not forearm_up(body, elbow, wrist):  # leaning far over, or reaching forward (desk, mouse, controller)
                continue
            hist.append((t, (wx - ex) / scale))
            waving = waving or self._swings([o for _, o in hist])
        return waving

    @staticmethod
    def _scale(body):
        """Shoulder width in pixels (the forearm's natural unit), or a stand-in when the person stands sideways."""
        if body.has(L_SHOULDER) and body.has(R_SHOULDER):
            w = float(np.hypot(*(body.kps[L_SHOULDER, :2] - body.kps[R_SHOULDER, :2])))
            torso = 0.0
            if body.has(L_HIP) or body.has(R_HIP):
                hip = body.kps[L_HIP if body.has(L_HIP) else R_HIP, :2]
                torso = float(np.hypot(*(body.kps[L_SHOULDER, :2] - hip)))
            return max(w, 0.5 * torso, 4.0)
        return None

    @staticmethod
    def _swings(offsets, turns_needed=None, amp=None):
        turns_needed = config.BODY_WAVE_TURNS if turns_needed is None else turns_needed
        amp = config.BODY_WAVE_AMP if amp is None else amp
        if len(offsets) < turns_needed + 2 or max(offsets) - min(offsets) < amp:
            return False
        turns, direction, anchor = 0, 0, offsets[0]
        for o in offsets[1:]:
            d = o - anchor
            if abs(d) < amp / 2:                # too small a step: not a swing yet
                continue
            s = 1 if d > 0 else -1
            if direction and s != direction:
                turns += 1
            direction, anchor = s, o
        return turns >= turns_needed


def forearm_up(body, elbow, wrist):
    """The forearm points up, not just a little: rises at least BODY_WAVE_FOREARM_UP x its sideways run. A forearm
    lying on a desk (mouse, keyboard, game controller) sits level with the elbow and wiggles - not a wave
    (2026-11-01: a PlayStation session's only "wave" was Selim's arm on the desk, rising 4 px over 35 px)."""
    if not (body.has(elbow) and body.has(wrist)):
        return False
    rise = float(body.kps[elbow, 1] - body.kps[wrist, 1])
    run = abs(float(body.kps[wrist, 0] - body.kps[elbow, 0]))
    return rise > 0 and rise >= config.BODY_WAVE_FOREARM_UP * run


def raised_wrist(body):
    """A forearm held up (forearm_up): (wrist x, wrist y, patch half-size px) of the higher one, or None."""
    best = None
    scale = WaveDetector._scale(body)
    for _, elbow, wrist in ARMS:
        if forearm_up(body, elbow, wrist):
            if best is None or body.kps[wrist, 1] < best[1]:
                r = max(config.BODY_WRIST_PATCH_MIN_PX, config.BODY_WRIST_PATCH * (scale or 0.0))
                best = (float(body.kps[wrist, 0]), float(body.kps[wrist, 1]), r)
    return best


class WristWatch:
    """Per camera frame, the patch round a raised wrist: where in it things moved since the last frame (sideways,
    in patch half-sizes). A waving hand makes that swing from side to side at the frame rate."""

    def __init__(self):
        self.hist = deque()                     # (t, offset)

    def update(self, y, prev, wrist, t):
        """y / prev = this and the last frame's grey picture (same pose: the head was still). True = waving."""
        while self.hist and t - self.hist[0][0] > config.BODY_WAVE_WINDOW_S:
            self.hist.popleft()
        x, yy, r = wrist
        h, w = y.shape
        x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
        y0, y1 = max(0, int(yy - 1.2 * r)), min(h, int(yy + 0.8 * r) + 1)    # the hand is up there, a bit above
        if x1 - x0 < 4 or y1 - y0 < 4:
            return False
        moved = np.abs(y[y0:y1, x0:x1].astype(np.int16) - prev[y0:y1, x0:x1].astype(np.int16)) > config.BODY_WAVE_DIFF
        if moved.mean() < config.BODY_WAVE_MOVED_MIN:
            return False
        cols = np.nonzero(moved)[1]
        self.hist.append((t, (float(cols.mean()) + x0 - x) / r))
        return WaveDetector._swings([o for _, o in self.hist], config.BODY_WAVE_FAST_TURNS, config.BODY_WAVE_FAST_AMP)


# ---------- tracks ----------

@dataclass
class BodyTrack:
    id: int
    body: Body
    pan: float              # box centre, head angles
    tilt: float
    width_deg: float
    height_deg: float
    first_t: float
    last_t: float
    hint: tuple = None      # face_hint() of the latest detection
    kang: np.ndarray = None  # 17 x 2 keypoint head angles of the latest detection...
    kvis: np.ndarray = None  # ...and which of them were seen
    hits: int = 1
    face: int = None        # the face track on its head (robot/tracker.Track id), None = no face seen on it
    person: str = None      # that face's person
    face_t: float = None    # last time a face was seen on it
    waved_t: float = None   # last time it was seen waving
    beckoned_t: float = None    # ...beckoning ("gel gel", robot/gestures.py)
    stop_t: float = None        # ...showing a stop sign
    path: deque = field(default_factory=lambda: deque(maxlen=200))     # (t, pan, tilt) of the box centre
    wave: WaveDetector = field(default_factory=WaveDetector)
    beckon: "gestures.Beckon" = field(default_factory=lambda: gestures.Beckon())
    stop_sign: "gestures.StopSign" = field(default_factory=lambda: gestures.StopSign())
    watch: WristWatch = field(default_factory=WristWatch)
    wrist: tuple = None     # raised_wrist() of the latest detection (None = no forearm up)
    seek_t: float = None    # attention started looking for its face at...
    given_up: bool = False  # ...and gave up (no face found there): not again until it moves
    given_up_at: tuple = None

    @property
    def still(self):
        """Never moved over BODY_STILL_S (a chair, a coat on a hook) - and no face was ever on it."""
        if self.face_t is not None or not self.path or self.path[-1][0] - self.path[0][0] < config.BODY_STILL_S:
            return False
        pans = [p[1] for p in self.path]
        tilts = [p[2] for p in self.path]
        return max(max(pans) - min(pans), max(tilts) - min(tilts)) <= config.BODY_STILL_DEG


class BodyTracker:
    def __init__(self):
        self.tracks = []
        self._ids = 0

    def update(self, found, t, width, height, pose, slack_deg=0.0):
        """found = [Body] of one frame taken at pose. Returns (new tracks, ended tracks)."""
        sightings = []
        for b in found:
            c = angles([[(b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2]], width, height, pose)[0]
            wdeg = camera.size_deg(b.box[2] - b.box[0], width)
            hdeg = camera.size_deg(b.box[3] - b.box[1], width)
            kang = angles(b.kps[:, :2], width, height, pose)
            sightings.append((b, float(c[0]), float(c[1]), wdeg, hdeg, face_hint(b, width, height, pose), kang,
                              b.kps[:, 2] >= config.BODY_KP_MIN))
        pairs = []
        for i, s in enumerate(sightings):
            for tr in self.tracks:
                d = self._distance(tr, s)
                # the gate grows with the time it was not seen (they moved meanwhile: out of view, the head turned)
                gate = max(config.BODY_GATE_MIN_DEG, config.BODY_GATE_SIZE * max(tr.width_deg, s[3])) + slack_deg + \
                    config.BODY_GATE_GROW_DPS * max(0.0, t - tr.last_t - config.BODY_VISIBLE_S)
                if d <= gate:
                    pairs.append((d, i, tr))
        pairs.sort(key=lambda p: p[0])
        used_s, used_t, new = set(), set(), []
        for d, i, tr in pairs:
            if i in used_s or tr.id in used_t:
                continue
            used_s.add(i)
            used_t.add(tr.id)
            self._apply(tr, sightings[i], t)
        for i, s in enumerate(sightings):
            if i not in used_s:
                self._ids += 1
                tr = BodyTrack(self._ids, s[0], s[1], s[2], s[3], s[4], t, t, hint=s[5], wrist=raised_wrist(s[0]),
                               kang=s[6], kvis=s[7])
                tr.path.append((t, s[1], s[2]))
                tr.wave.update(s[0], t)
                _signs(tr, s[0], t)
                self.tracks.append(tr)
                new.append(tr)
        ended = [tr for tr in self.tracks if t - tr.last_t > config.BODY_MAX_MISSING_S]
        self.tracks = [tr for tr in self.tracks if tr not in ended]
        return new, ended

    @staticmethod
    def _distance(tr, s):
        """How far a sighting is from a track (deg): the box centres, or - when both saw at least two of the same
        keypoints - those keypoints (a box changes a lot as the head turns: legs only, then legs + body)."""
        d = math.hypot(s[1] - tr.pan, s[2] - tr.tilt)
        if tr.kang is not None:
            common = tr.kvis & s[7]
            if common.sum() >= 2:
                d = min(d, float(np.median(np.hypot(*(s[6][common] - tr.kang[common]).T))))
        return d

    @staticmethod
    def _apply(tr, s, t):
        tr.body, tr.pan, tr.tilt, tr.width_deg, tr.height_deg, tr.hint = s[0], s[1], s[2], s[3], s[4], s[5]
        tr.kang, tr.kvis = s[6], s[7]
        tr.last_t, tr.hits = t, tr.hits + 1
        tr.wrist = raised_wrist(s[0])
        tr.path.append((t, s[1], s[2]))
        if tr.given_up and tr.given_up_at and math.hypot(s[1] - tr.given_up_at[0], s[2] - tr.given_up_at[1]) \
                > config.BODY_REARM_DEG:
            tr.given_up, tr.given_up_at, tr.seek_t = False, None, None      # it moved: worth another look
        if tr.wave.update(s[0], t):
            tr.waved_t = t
        _signs(tr, s[0], t)

    def visible(self, t, max_age=None):
        age = config.BODY_VISIBLE_S if max_age is None else max_age
        return [tr for tr in self.tracks if t - tr.last_t <= age]


def _signs(tr, body, t):
    """The hand signs of one detection. A beckon is not taken from an arm that waved within the window (nor a wave
    from one that beckoned: _waved) - the same swing must never be both."""
    if tr.beckon.update(body, t) and not (tr.waved_t is not None and t - tr.waved_t <= config.GESTURE_BECKON_WINDOW_S):
        tr.beckoned_t = t
    if tr.stop_sign.update(body, t):
        tr.stop_t = t


def link_faces(bodies, faces, t):
    """Put each visible face track on the body whose head it is: near the body's head keypoints, else inside the
    top part of its box. Each face goes to one body (the nearest)."""
    taken = set()
    for bt in sorted(bodies, key=lambda b: -b.height_deg):
        best, best_d = None, None
        for f in faces:
            if f.id in taken:
                continue
            if bt.hint is not None and bt.hint[2] == "head":
                d = math.hypot(f.pan - bt.hint[0], f.tilt - bt.hint[1])
                ok = d <= max(config.BODY_FACE_NEAR_DEG, 1.5 * f.sighting.size_deg)
            else:                               # no head keypoints: the face must sit in the box's top part
                top = bt.tilt + bt.height_deg / 2
                d = math.hypot(f.pan - bt.pan, f.tilt - top)
                ok = abs(f.pan - bt.pan) <= bt.width_deg / 2 and top - 0.4 * bt.height_deg <= f.tilt <= top + 2.0
            if ok and (best_d is None or d < best_d):
                best, best_d = f, d
        if best is not None:
            taken.add(best.id)
            bt.face, bt.face_t = best.id, t
            if best.person is not None:
                bt.person = best.person


# ---------- the whole body sense, per camera frame ----------

class BodySense:
    def __init__(self, detector, clock=None, inline=False):
        import time
        self.worker = None if detector is None else (InlineBodies(detector) if inline else BodyWorker(detector))
        self.tracker = BodyTracker()
        self._clock = clock or time.monotonic
        self.next_t = -math.inf
        self.stats = dict(runs=0, ms=0.0)
        self.last_found = []                    # raw bodies of the latest result (frame pixels), for the recorder
        self.last_size = (0, 0)
        self.last_pose, self.last_t = (0.0, 0.0), None     # the head pose / time of that frame
        self._prev_y = None                     # last frame's grey picture, kept only while a forearm is up
        self._said = {}                         # body track id -> time of its last "waved" event
        self._signed = {}                       # (body track id, kind) -> time of its last sign event
        self.seen_t = -math.inf                 # last time a face or a body was in view
        self._last_crop = False                 # the last run was a crop: the next one is the whole frame
        self._cropped = {}                      # face track id -> when a crop under it was last run
        self.stats.update(crops=0)

    @property
    def enabled(self):
        return self.worker is not None

    def process(self, frame, pose_at, speed, faces):
        """One camera frame: hand it to the model when due, apply results that came back. faces = visible face
        tracks. Returns events: waved."""
        if self.worker is None:
            return []
        t = frame.t
        events = []
        for job in self.worker.done():
            events += self._result(job, faces, speed)
        events += self._watch_wrists(frame, t, speed)
        if t >= self.next_t and not self.worker.busy:
            job = self._crop_job(frame, faces, t, pose_at) if not self._last_crop else None
            self._last_crop = job is not None
            if job is None:
                job = BodyJob(t=t, pose=pose_at(t), rgb=np.array(frame.lores_rgb(), copy=True))
            if self.worker.submit(job):
                self.next_t = t + self._every(faces, t)
            for job in self.worker.done():      # inline: already done
                events += self._result(job, faces, speed)
        return events

    def far_faces(self, faces):
        """Face tracks too small for the whole-frame model to find their body reliably (2026-11-02 robot shots at
        3 m: faces 10-12 px, body confidence 0.37-0.70 on the whole frame, 0.89-0.92 on a crop)."""
        out = []
        for f in faces:
            face = getattr(getattr(f, "sighting", None), "face", None)
            if face is not None and getattr(face, "box", None) is not None and \
                    face.box[2] - face.box[0] <= config.BODY_CROP_FACE_MAX_PX:
                out.append(f)
        return out

    def _crop_job(self, frame, faces, t, pose_at):
        """Every other run while a far face is in view: the model on a big-stream crop under it (the face waited
        longest first). None = run on the whole frame."""
        if not config.BODY_CROP or not hasattr(frame, "crop_main"):
            return None
        far = self.far_faces(faces)
        if not far:
            return None
        f = min(far, key=lambda f: self._cropped.get(f.id, -math.inf))
        size = tuple(frame.lores_rgb().shape[1::-1])
        box = crop_for_face(f.sighting.face.box, size)
        try:
            crop, (x0, y0, s) = frame.crop_main(box, pad=0.0)
        except Exception:  # noqa: BLE001 - no big stream in this frame: the whole frame then
            return None
        if crop.size == 0:
            return None
        self._cropped[f.id] = t
        if len(self._cropped) > 50:
            self._cropped = dict(sorted(self._cropped.items(), key=lambda kv: kv[1])[-20:])
        self.stats["crops"] += 1
        return BodyJob(t=t, pose=pose_at(t), rgb=np.array(crop, copy=True), origin=(x0, y0), scale=s, size=size,
                       face=f.id)

    def _every(self, faces, t):
        """How soon to look again: fast while an arm is up (waving?) or a body has no face yet, slow while a face is
        followed and its body known."""
        vis = self.tracker.visible(t)
        if vis or faces:
            self.seen_t = t
        if any(tr.wrist is not None for tr in vis):
            return config.BODY_EVERY_FAST_S
        if t - self.seen_t >= config.BODY_IDLE_AFTER_S:     # nobody around for a while: look less often
            return config.BODY_EVERY_IDLE_S
        if not faces or any(tr.face_t is None or t - tr.face_t > 1.0 for tr in vis):
            return config.BODY_EVERY_S
        return config.BODY_EVERY_SLOW_S

    def _result(self, job, faces, speed):
        if job.origin is not None:              # a crop: back into small-stream pixels
            to_frame(job.bodies, job.origin, job.scale)
            w, h = job.size
        else:
            h, w = job.rgb.shape[:2]
        self.last_found, self.last_size = job.bodies, (w, h)
        self.last_pose, self.last_t = tuple(job.pose), job.t
        self.stats["runs"] += 1
        self.stats["ms"] = job.ms
        self.tracker.update(job.bodies, job.t, w, h, job.pose, slack_deg=speed * config.VISION_SLACK_S)
        link_faces(self.tracker.visible(job.t), faces, job.t)
        events = []
        for tr in self.tracker.visible(job.t):
            if tr.beckoned_t == job.t:
                events += self._sign(tr, job.t, "beckoned", config.GESTURE_BECKON_AGAIN_S)
            if tr.stop_t == job.t:
                events += self._sign(tr, job.t, "stop_sign", config.GESTURE_STOP_AGAIN_S)
            if tr.waved_t == job.t:
                events += self._waved(tr, job.t, "pose")
        return events

    def _sign(self, tr, t, kind, again_s):
        """One event per sign (again_s), like waves: beckoned / stop_sign with where the body is."""
        if t - self._signed.get((tr.id, kind), -math.inf) < again_s:
            return []
        self._signed[(tr.id, kind)] = t
        if len(self._signed) > 100:
            self._signed = dict(sorted(self._signed.items(), key=lambda kv: kv[1])[-20:])
        return [dict(kind=kind, body=tr.id, track=tr.face, person=tr.person, pan=round(tr.pan, 1),
                     tilt=round(tr.tilt, 1))]

    def _watch_wrists(self, frame, t, speed):
        """Every frame while a forearm is up and the head is still: the moving pixels round that wrist."""
        up = [tr for tr in self.tracker.visible(t) if tr.wrist is not None]
        y = getattr(frame, "y_plane", None)
        if not up or y is None or speed > config.VISION_STILL_DPS:
            self._prev_y = None
            return []
        y = np.asarray(y)
        events = []
        if self._prev_y is not None and self._prev_y.shape == y.shape:
            for tr in up:
                if tr.watch.update(y, self._prev_y, tr.wrist, t):
                    tr.waved_t = t
                    events += self._waved(tr, t, "motion")
        self._prev_y = y.copy()
        return events

    def _waved(self, tr, t, how):
        """One "waved" event per wave (BODY_WAVE_AGAIN_S), whichever way it was seen - none from a beckoning arm."""
        if t - self._said.get(tr.id, -math.inf) < config.BODY_WAVE_AGAIN_S:
            return []
        if tr.beckoned_t is not None and t - tr.beckoned_t <= config.GESTURE_BECKON_WINDOW_S:
            return []
        self._said[tr.id] = t
        if len(self._said) > 100:
            self._said = dict(sorted(self._said.items(), key=lambda kv: kv[1])[-20:])
        hint = tr.hint or (tr.pan, tr.tilt, None)
        return [dict(kind="waved", body=tr.id, track=tr.face, person=tr.person, how=how,
                     pan=round(hint[0], 1), tilt=None if hint[1] is None else round(hint[1], 1))]

    def seekable(self, t):
        """Bodies worth looking for a face on: seen now, moving (not a chair), no face on them lately, not given up,
        and with an idea where the face is."""
        out = []
        for tr in self.tracker.visible(t):
            if tr.still or tr.given_up or tr.hint is None or tr.hits < config.BODY_SEEK_MIN_HITS:
                continue
            if tr.face_t is not None and t - tr.face_t < config.BODY_FACE_FRESH_S:
                continue
            out.append(tr)
        return out

    def give_up(self, tr):
        tr.given_up, tr.given_up_at = True, (tr.pan, tr.tilt)

    def close(self):
        if self.worker is not None:
            self.worker.close()


def body_summary(tracks):
    out = []
    for tr in tracks:
        how = tr.hint[2] if tr.hint else "-"
        flags = (" still" if tr.still else "") + (" arm up" if tr.wrist else "") + \
            (f" face#{tr.face}" if tr.face else "")
        out.append(f"B{tr.id} pan {tr.pan:+.0f} tilt {tr.tilt:+.0f} {tr.height_deg:.0f}deg {how}{flags}")
    return ", ".join(out) if out else "-"
