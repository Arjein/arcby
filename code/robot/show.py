"""
Showing things to the robot (docs/vision.md, part S). Selim 2026-11-02: no trigger phrase - when the
robot is SURE someone holds something up to it, it looks at the thing and the mind reacts; when asked ("bak", "bu
ne?") it takes it as certain that something is being shown.

  1. a hand held up (the body tracks of robot/body.py): wrist above the hips, in front of the body (between the
     shoulders +- SHOW_SIDE_SW shoulder widths), the forearm not hanging down - and STILL (the wrist moves under
     SHOW_STILL_SW shoulder widths for SHOW_HOLD_S; a wave goes side to side);
  2. a thing in that hand: the object model (YOLO26n, 80 COCO classes) on a big-stream crop round the wrist, a few
     times a second while a hand is held up; a box near the hand, score >= SHOW_MIN_SCORE, a class one can hold;
  3. sure = held up >= SHOW_HOLD_S + the same thing in SHOW_NEED of the last SHOW_OF runs + that person looking at
     the robot (their face on that body) -> a "showing" event (once per person and thing per SHOW_AGAIN_S);
  4. asked (`ask()`, the conversation heard "bak" / "bu ne"...) = sure by definition: the asker's raised hand (else
     their hands, else just under their face) for SHOW_ASK_S, a picture taken even if no object was found.

While something is shown, `target()` is where the head should look (attention mode "show"), and `picture()` is a
JPEG crop of the big stream round it (for the mind - only taken at that moment).

    show = ShowSense(load_objects())
    events = show.process(frame, pose_at, bodies.tracker.visible(t), face_tracks)    # each camera frame
"""
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import body as B
from . import camera, config

log = logging.getLogger(__name__)

COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush")
# things nobody holds up in one hand (or the person themselves)
NOT_HELD = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "chair", "couch", "bed", "dining table", "toilet", "microwave", "oven", "sink",
            "refrigerator", "surfboard", "skis", "snowboard"}
ASK_WORDS = ("bak", "baksana", "bakar misin", "suna bak", "buna bak", "bu ne", "ne bu", "bunu gordun mu",
             "gordun mu", "bak bakalim", "look", "what is this", "what's this", "see this")


def asked_to_look(text):
    """Did they ask it to look at something ("Bak!", "Bu ne?", "Şuna bak", "Bunu gördün mü?")?"""
    from .conversation import plain_words
    words = " ".join(plain_words(text or ""))
    return any(f" {w} " in f" {words} " for w in ASK_WORDS)


# ---------- the object model ----------

@dataclass
class Thing:
    box: np.ndarray         # x1, y1, x2, y2 (pixels of the image it was found in, then of the small stream)
    score: float
    cls: int
    names: tuple = None     # the model's own words (YOLOE home list); None = the 80 COCO classes

    @property
    def name(self):
        names = self.names or COCO
        return names[self.cls] if 0 <= self.cls < len(names) else str(self.cls)


def decode_objects(out, scale, pad, min_score=None, iou=0.5, names=None):
    """YOLO detection output -> [Thing] in input-image pixels. Two layouts: raw (1 x (4 + classes [+ 32 mask
    coefficients of a YOLOE segmentation export]) x N: cx, cy, w, h, class scores - NMS here) or end-to-end
    (1 x N x 6: x1, y1, x2, y2, score, class). names = the classes of a vocabulary model (else COCO)."""
    min_score = config.SHOW_MIN_SCORE if min_score is None else min_score
    names = tuple(names) if names else None
    nc = len(names) if names else len(COCO)
    a = np.asarray(out)[0]
    px, py = pad
    if a.ndim == 2 and a.shape[1] == 6:
        rows = a[a[:, 4] >= min_score]
        boxes = (rows[:, :4] - [px, py, px, py]) / scale
        return [Thing(boxes[i], float(rows[i, 4]), int(rows[i, 5]), names) for i in range(len(rows))]
    rows = a.T
    cls = rows[:, 4:4 + nc].argmax(1)
    score = rows[np.arange(len(rows)), 4 + cls]
    keep = score >= min_score
    rows, cls, score = rows[keep], cls[keep], score[keep]
    if not len(rows):
        return []
    cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    boxes = np.stack([cx - w / 2 - px, cy - h / 2 - py, cx + w / 2 - px, cy + h / 2 - py], axis=1) / scale
    out = []
    for c in np.unique(cls):                       # NMS per class
        idx = np.where(cls == c)[0]
        for k in B.nms(boxes[idx], score[idx], iou):
            i = idx[k]
            out.append(Thing(boxes[i], float(score[i]), int(c), names))
    return out


class ObjectDetector:
    def __init__(self, session, size=None, names=None, name="objects", min_score=None):
        self.session = session
        self.min_score = min_score              # this model's own threshold (YOLOE scores lower: 0.25)
        self.size = tuple(size or (320, 320))
        self.input = session.get_inputs()[0].name
        self.names = tuple(names) if names else None     # a vocabulary model's words (None: COCO)
        self.name = name

    def detect(self, rgb):
        x, s, pad = B.letterbox(rgb, self.size)
        return decode_objects(self.session.run(None, {self.input: x})[0], s, pad, names=self.names,
                              min_score=self.min_score)


def load_objects(models_dir=None, threads=None, order=None):
    """The object model, or None when none is there (then only asking makes it look at a hand). The first of
    SHOW_MODELS whose file exists: YOLOE with the home word list (the showing session 2026-11-02: 8 of 9 showing steps,
    no false alarm while looking at the robot), else the COCO YOLO26n (3 of 9)."""
    from .faces import MODELS, _session, model_path
    for key in order or config.SHOW_MODELS:
        path = model_path(key, models_dir)
        if os.path.exists(path):
            m = MODELS[key]
            return ObjectDetector(_session(path, threads or config.BODY_THREADS), size=m["input"],
                                  names=m.get("classes"), name=key, min_score=m.get("min_score"))
    log.info("show: no object model - things are looked at only when asked (get_models.py --derive --only "
             "yoloe_home on the Mac)")
    return None


# ---------- geometry ----------

def pixel_of(ang, width, height, pose):
    """Head angles (pan, tilt) -> pixel (x, y) in a frame taken at pose (the inverse of body.angles)."""
    f = camera.focal_px(width)
    x = width / 2 - f * math.tan(math.radians(ang[0] - pose[0]))
    y = height / 2 - f * math.tan(math.radians(ang[1] - pose[1]))
    return x, y


def _sw_deg(tr):
    """Shoulder width of a body track in degrees (None when a shoulder is not seen)."""
    if tr.kang is None or not (tr.kvis[B.L_SHOULDER] and tr.kvis[B.R_SHOULDER]):
        return None
    return float(np.hypot(*(tr.kang[B.L_SHOULDER] - tr.kang[B.R_SHOULDER]))) or None


def held_up(tr):
    """The wrists of a body track that are held up in front of it: [(wrist index, (pan, tilt))]. In head angles
    (tilt + = up): above the hips (or 1.6 shoulder widths under the shoulders when no hip is seen), between the
    shoulders +- SHOW_SIDE_SW, the forearm not hanging down (wrist not below the elbow by more than 0.3 sw)."""
    sw = _sw_deg(tr)
    if sw is None:
        return []
    k, v = tr.kang, tr.kvis
    sh_tilt = (k[B.L_SHOULDER, 1] + k[B.R_SHOULDER, 1]) / 2
    hips = [k[i, 1] for i in (B.L_HIP, B.R_HIP) if v[i]]
    low = max(hips) if hips else sh_tilt - 1.6 * sw
    left = min(k[B.L_SHOULDER, 0], k[B.R_SHOULDER, 0]) - config.SHOW_SIDE_SW * sw
    right = max(k[B.L_SHOULDER, 0], k[B.R_SHOULDER, 0]) + config.SHOW_SIDE_SW * sw
    out = []
    for _, elbow, wrist in B.ARMS:
        if not (v[wrist] and v[elbow]):
            continue
        wp, wt = k[wrist]
        if wt <= low or not (left <= wp <= right) or wt < k[elbow, 1] - 0.3 * sw:
            continue
        out.append((wrist, (float(wp), float(wt))))
    return out


def upper_body(tr, size, pose, sw_px):
    """The person's head, arms and hands as one box (small-stream pixels, at this frame's pose) + SHOW_ROI_MARGIN
    shoulder widths round them - what the object model looks at (the showing session 2026-11-02: what is held up is
    anywhere between the face and the waist, often past the wrist). None without 3 such keypoints."""
    if tr is None or tr.kang is None:
        return None
    idx = [i for i in (B.NOSE, B.L_SHOULDER, B.R_SHOULDER, B.L_ELBOW, B.R_ELBOW, B.L_WRIST, B.R_WRIST) if tr.kvis[i]]
    if len(idx) < 3:
        return None
    pts = np.array([pixel_of(tr.kang[i], size[0], size[1], pose) for i in idx])
    m = config.SHOW_ROI_MARGIN * sw_px
    x1, y1 = pts.min(0) - m
    x2, y2 = pts.max(0) + m
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(size[0]), x2), min(float(size[1]), y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


# ---------- per body and hand ----------

@dataclass
class Hand:
    path: deque = field(default_factory=lambda: deque(maxlen=60))     # (t, pan, tilt) of the wrist
    hits: deque = field(default_factory=lambda: deque(maxlen=8))      # (t, Thing | None) of the object runs
    sw: float = 10.0
    last_t: float = -math.inf

    def held_s(self, t):
        """How long the wrist has been held still (s) up to t."""
        if not self.path:
            return 0.0
        tol = config.SHOW_STILL_SW * self.sw
        _, p0, q0 = self.path[-1]
        start = self.path[-1][0]
        for tt, p, q in reversed(self.path):
            if math.hypot(p - p0, q - q0) > tol:
                break
            start = tt
        return t - start if t - self.path[-1][0] <= config.BODY_VISIBLE_S else 0.0

    def thing(self):
        """The thing seen in SHOW_NEED of the last SHOW_OF object runs (the newest box of it), or None."""
        recent = list(self.hits)[-config.SHOW_OF:]
        names = [h.name for _, h in recent if h is not None]
        for name in set(names):
            if names.count(name) >= config.SHOW_NEED:
                return next(h for _, h in reversed(recent) if h is not None and h.name == name)
        return None


class ShowSense:
    def __init__(self, detector, clock=None):
        import time
        self.worker = None if detector is None else B.BodyWorker(detector)   # one thread, newest crop wins
        self.min_score = getattr(detector, "min_score", None) or config.SHOW_MIN_SCORE   # this model's own threshold
        self.inline = None
        self._clock = clock or time.monotonic
        self.hands = {}                     # (body track id, wrist index) -> Hand
        self.next_t = -math.inf
        self._jobs = {}                     # id(job) -> (key, origin, scale)
        self._said = {}                     # (person or body, thing) -> time of its last "showing" event
        self.current = None                 # dict(pan, tilt, until, what, person, body) - where to look now
        self.asked_until = -math.inf
        self.asked_by = None
        self._want_picture = False
        self._picture = None                # (t, JPEG bytes, what)
        self.stats = dict(runs=0, ms=0.0, shown=0)
        self.session = None                 # robot/session_rec.SessionRecorder: hand crops kept while a session runs

    @classmethod
    def inline_for_tests(cls, detector, clock=None):
        s = cls(None, clock=clock)
        s.worker = B.InlineBodies(detector)
        s.min_score = getattr(detector, "min_score", None) or config.SHOW_MIN_SCORE
        return s

    @property
    def enabled(self):
        return True                         # asking works without the object model

    # ----- what the others ask -----

    def ask(self, t, person=None):
        """The conversation heard "bak" / "bu ne": look at that person's hands for SHOW_ASK_S, take a picture."""
        self.asked_until, self.asked_by = t + config.SHOW_ASK_S, person
        self._want_picture = True

    def target(self, t):
        """(pan, tilt, what) for the head while something is shown, else None."""
        c = self.current
        if c is None or t > c["until"]:
            return None
        return c["pan"], c["tilt"], c["what"]

    def picture(self, max_age_s=5.0, t=None):
        """(JPEG bytes, what) of the thing shown, taken in the last max_age_s - or None."""
        t = self._clock() if t is None else t
        if self._picture is None or t - self._picture[0] > max_age_s:
            return None
        return self._picture[1], self._picture[2]

    # ----- per frame -----

    def process(self, frame, pose_at, bodies, faces):
        """One camera frame. bodies = visible body tracks (robot/body.py), faces = visible face tracks.
        Returns events: showing."""
        t = frame.t
        events = []
        size = tuple(frame.lores_rgb().shape[1::-1])
        pose = pose_at(t)
        for tr in bodies:                   # the wrists held up, from each body's latest detection
            if tr.last_t < t - config.BODY_VISIBLE_S:
                continue
            sw = _sw_deg(tr)
            for wrist, ang in held_up(tr):
                h = self.hands.setdefault((tr.id, wrist), Hand())
                if h.last_t != tr.last_t:
                    h.path.append((tr.last_t, ang[0], ang[1]))
                    h.last_t, h.sw = tr.last_t, sw or h.sw
        self.hands = {k: h for k, h in self.hands.items() if t - h.last_t <= config.BODY_MAX_MISSING_S}
        if self.worker is not None:
            for job in self.worker.done():
                events += self._result(job, t, bodies, faces)
            if t >= self.next_t and not self.worker.busy and hasattr(frame, "crop_main"):
                held = [(k, h) for k, h in self.hands.items() if h.held_s(t) >= config.SHOW_LOOK_S]
                if held:
                    k, h = max(held, key=lambda kh: kh[1].held_s(t))
                    self._run(frame, k, h, size, pose, next((b for b in bodies if b.id == k[0]), None))
                    self.next_t = t + config.SHOW_EVERY_S
                    for job in self.worker.done():      # inline (tests): already done
                        events += self._result(job, t, bodies, faces)
        if t <= self.asked_until:
            events += self._asked(frame, t, size, pose, bodies, faces)
        if self._want_picture and self.current is not None and hasattr(frame, "crop_main"):
            self._take_picture(frame, size, pose)
        return events

    def _run(self, frame, key, hand, size, pose, tr=None):
        _, p, q = hand.path[-1]
        x, y = pixel_of((p, q), size[0], size[1], pose)
        sw_px = hand.sw / config.CAMERA_HFOV_DEG * size[0]
        box = upper_body(tr, size, pose, sw_px) if tr is not None else None
        if box is None:                         # no keypoints to frame the body: a square round the wrist
            half = max(config.SHOW_CROP_SW * sw_px / 2, 24)
            box = (x - half, y - half, x + half, y + half)
        try:
            crop, (x0, y0, s) = frame.crop_main(box, pad=0.0)
        except Exception:  # noqa: BLE001 - no big stream: no object run
            return
        if crop.size == 0:
            return
        job = B.BodyJob(t=frame.t, pose=pose, rgb=np.array(crop, copy=True), origin=(x0, y0), scale=s, size=size)
        if self.worker.submit(job):
            self._jobs[id(job)] = (key, (x, y), sw_px)

    def _result(self, job, t, bodies, faces):
        key, (hx, hy), sw_px = self._jobs.pop(id(job), (None, (0, 0), 1))
        self.stats["runs"] += 1
        self.stats["ms"] = job.ms
        hand = self.hands.get(key)
        if hand is None:
            return []
        best = None
        for th in job.bodies or []:
            th.box[[0, 2]] = (th.box[[0, 2]] + job.origin[0]) / job.scale
            th.box[[1, 3]] = (th.box[[1, 3]] + job.origin[1]) / job.scale
            if th.name in NOT_HELD or th.score < self.min_score:
                continue
            cx, cy = (th.box[0] + th.box[2]) / 2, (th.box[1] + th.box[3]) / 2
            near = math.hypot(cx - hx, cy - hy) <= config.SHOW_NEAR_SW * sw_px + (th.box[2] - th.box[0]) / 2
            if near and (best is None or th.score > best.score):
                best = th
        hand.hits.append((job.t, best))
        found = sorted(((th.name, round(th.score, 2)) for th in job.bodies or []), key=lambda x: -x[1])[:4]
        rec = self.session
        if rec is not None and rec.active():
            try:
                from .session_rec import save_crop
                save_crop(rec.dir, self.stats["runs"], job.rgb, dict(
                    wall=round(time.time(), 3), t=round(job.t, 3), body=key[0], wrist=key[1],
                    held=round(hand.held_s(t), 2), origin=list(job.origin), scale=job.scale,
                    best=[best.name, round(best.score, 2)] if best is not None else None, found=found))
            except Exception as e:  # noqa: BLE001 - a test session must never stop the robot
                log.warning("show: crop not saved (%s)", e)
        log.info("show: body %s hand held %.1f s -> %s (all: %s) %.0f ms", key[0], hand.held_s(t),
                 f"{best.name} {best.score:.2f}" if best is not None else "nothing near the hand", found, job.ms)
        return self._sure(key, hand, t, job, bodies, faces)

    def _sure(self, key, hand, t, job, bodies, faces):
        thing = hand.thing()
        if thing is None or hand.held_s(t) < config.SHOW_HOLD_S:
            return []
        tr = next((b for b in bodies if b.id == key[0]), None)
        face = next((f for f in faces if tr is not None and f.id == tr.face), None)
        if face is None or not getattr(face, "looking", False):
            return []                       # not shown TO the robot (looking at the phone themselves)
        cx, cy = (thing.box[0] + thing.box[2]) / 2, (thing.box[1] + thing.box[3]) / 2
        pan, tilt = B.angles([[cx, cy]], job.size[0], job.size[1], job.pose)[0]
        who = tr.person or f"body{tr.id}"
        self.current = dict(pan=float(pan), tilt=float(tilt), until=t + config.SHOW_KEEP_S, what=thing.name,
                            person=tr.person, body=tr.id, box=thing.box.copy(), size=job.size, pose=job.pose)
        if t - self._said.get((who, thing.name), -math.inf) < config.SHOW_AGAIN_S:
            return []
        self._said[(who, thing.name)] = t
        self._want_picture = True
        self.stats["shown"] += 1
        return [dict(kind="showing", what=thing.name, score=round(thing.score, 2), body=tr.id, track=tr.face,
                     person=tr.person, pan=round(float(pan), 1), tilt=round(float(tilt), 1), asked=False)]

    def _asked(self, frame, t, size, pose, bodies, faces):
        """Asked to look: sure by definition. The asker's (or the only / biggest) body: a hand held up, else both
        wrists, else just under their face."""
        if self.current is not None and t <= self.current["until"] and self.current.get("asked"):
            return []
        mine = [b for b in bodies if self.asked_by is not None and b.person == self.asked_by] or \
            sorted(bodies, key=lambda b: -b.height_deg)[:1]
        spot, what = None, None
        tr = mine[0] if mine else None
        if tr is not None:
            held = [(k, h) for k, h in self.hands.items() if k[0] == tr.id and h.path]
            if held:
                k, h = max(held, key=lambda kh: kh[1].held_s(t))
                _, p, q = h.path[-1]
                spot = (p, q)
                th = h.thing()
                what = th.name if th is not None else None
            elif tr.kang is not None and (tr.kvis[B.L_WRIST] or tr.kvis[B.R_WRIST]):
                pts = [tr.kang[i] for i in (B.L_WRIST, B.R_WRIST) if tr.kvis[i]]
                spot = tuple(np.mean(pts, axis=0))
        if spot is None:                    # no body: just under the asker's face (or the face looked at)
            f = next((f for f in faces if self.asked_by is not None and f.person == self.asked_by), None) or \
                next((f for f in faces if getattr(f, "looking", False)), None)
            if f is None:
                return []
            spot = (f.pan, f.tilt - 2.5 * f.sighting.size_deg)
        self.current = dict(pan=float(spot[0]), tilt=float(spot[1]), until=self.asked_until + config.SHOW_KEEP_S,
                            what=what, person=self.asked_by or (tr.person if tr else None),
                            body=tr.id if tr else None, box=None, size=size, pose=pose, asked=True)
        self._want_picture = True
        self.stats["shown"] += 1
        return [dict(kind="showing", what=what, body=tr.id if tr else None, track=tr.face if tr else None,
                     person=self.current["person"], pan=round(float(spot[0]), 1), tilt=round(float(spot[1]), 1),
                     asked=True)]

    def _take_picture(self, frame, size, pose):
        """A JPEG crop of the big stream round what is shown (for the mind), at this frame's head pose."""
        c = self.current
        x, y = pixel_of((c["pan"], c["tilt"]), size[0], size[1], pose)
        if c.get("box") is not None:
            bw = max(c["box"][2] - c["box"][0], c["box"][3] - c["box"][1])
        else:
            bw = size[0] * 0.18
        half = max(bw * 0.9, 40)
        try:
            crop, _ = frame.crop_main((x - half, y - half, x + half, y + half), pad=0.0)
            if crop.size == 0:
                return
            from io import BytesIO
            from PIL import Image
            img = Image.fromarray(crop)
            img.thumbnail((config.SHOW_PICTURE_PX, config.SHOW_PICTURE_PX))
            buf = BytesIO()
            img.save(buf, "JPEG", quality=85)
        except Exception as e:  # noqa: BLE001 - no picture this frame: the next one
            log.debug("show: picture failed (%s)", e)
            return
        self._picture = (frame.t, buf.getvalue(), c["what"])
        self._want_picture = False

    def close(self):
        if self.worker is not None:
            self.worker.close()
