"""
Vision: one camera frame in, face tracks with identities and events out. Ties together faces.py (models),
tracker.py (who is where, votes), people.py (who we know) - no hardware of its own.

Per frame:
  1. find faces in the small stream (detector, ~60 ms on the Pi 4)
  2. track them in head angles (tracker)
  3. a track that needs a look: cut the face out of the BIG stream, find precise landmarks there
     (refiner, on that crop only), align; the embedder (~0.4 s for r50_int8) runs in its own thread
     (EmbedWorker) and the result is matched against the people we know and voted on in a later frame
  4. decided tracks teach the store new angles (learn); a track that matched nobody on every look becomes
     a new person - but only from a big, sharp, roughly frontal face
  5. evidence for duplicate people: a track that matched two ids -> link; two people in one frame ->
     apart; clear cases are merged at once, the rest become "ask_same" events for the mind to ask about

    vision = Vision(detector, EmbedWorker(embedder), store, refiner=scrfd_on_crops)   # or a plain embedder:
                                                                      # embeds at once (tests, replays)
    events = vision.process(frame, pose_at, speed_at)
        # pose_at(t) -> head (pan, tilt) when the frame was taken; speed_at(t) -> how fast it moved then
        # (deg/s). A moving head blurs the frame and its pose is less sure: the tracker gate grows and no
        # face is identified or learnt until it is still again - like saccadic suppression in eyes.

Events (dicts with "kind"): face_new, person_seen, stranger, enrolled, face_lost, merged, ask_same.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import camera, config, faces, gaze
from .tracker import Sighting, Tracker


log = logging.getLogger(__name__)


@dataclass
class Look:
    """One face cut out of a frame, waiting for (or back from) the embedder."""
    track: int
    t: float                # frame time
    aligned: np.ndarray     # 112 x 112 RGB
    crop: np.ndarray        # the face region of the big stream (for photos / collection)
    width_px: float         # face width in the big stream
    yaw: float              # frontal(): 0 straight .. 1+ profile
    luma: float             # face brightness in the small stream
    pan: float
    tilt: float
    head_dps: float
    quality: float = 1.0    # SCRFD score on the big crop (0 = it found no face there)
    engagement: float = 0.0  # how much the face was turned toward the robot (Track.engagement)
    facing: float = None    # gaze.facing() of this look's big-stream landmarks (None = no landmarks there)
    ms: float = 0.0         # embedding time


class EmbedWorker:
    """Runs the embedder in its own thread, so a slow model (r50_int8: ~0.4 s on the Pi 4) never holds up the
    camera loop. submit() hands a look over (False when max_pending are already waiting); done() returns the
    finished (look, vector) pairs - call both from the same (camera) thread."""

    def __init__(self, embedder, max_pending=4):
        self.embedder, self.max_pending = embedder, max_pending
        self._todo, self._done = queue.Queue(), queue.Queue()
        self.pending = 0
        self._thread = threading.Thread(target=self._run, name="embed", daemon=True)
        self._thread.start()

    def submit(self, look):
        if self.pending >= self.max_pending:
            return False
        self.pending += 1
        self._todo.put(look)
        return True

    def done(self):
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                break
        self.pending -= len(out)
        return out

    def close(self):
        self._todo.put(None)
        self._thread.join(timeout=5)

    def _run(self):
        while True:
            look = self._todo.get()
            if look is None:
                return
            t0 = time.perf_counter()
            try:
                vec = self.embedder.embed(look.aligned)
            except Exception:  # noqa: BLE001 - one bad face must not stop recognition
                log.exception("embedding failed")
                vec = None
            look.ms = (time.perf_counter() - t0) * 1000
            self._done.put((look, vec))


class InlineEmbed:
    """EmbedWorker's interface, embedding at once in submit() (tests, photo replays)."""

    def __init__(self, embedder):
        self.embedder, self._done, self.pending = embedder, [], 0

    def submit(self, look):
        t0 = time.perf_counter()
        vec = self.embedder.embed(look.aligned)
        look.ms = (time.perf_counter() - t0) * 1000
        self._done.append((look, vec))
        return True

    def done(self):
        out, self._done = self._done, []
        return out

    def close(self):
        pass


@dataclass
class GazeJob:
    """One confirmed face's big-stream crop, waiting for (or back from) the gaze refiner."""
    track: int
    t: float                # frame time
    crop: np.ndarray
    box: np.ndarray         # the track's box in crop pixels (own_face picks the right face in the padded crop)


class GazeWorker:
    """Runs the gaze refiner (small SCRFD on big-stream crops) in its own thread, like EmbedWorker: with faces in
    view it took the camera loop from 12-15 down to 7-10 fps (2026-11-01). The crop is cut in the camera thread
    (the frame's buffers go back to the camera after it), only the model runs here; results come back as
    (job, own face or None) through done() on a later frame."""

    def __init__(self, refiner, max_pending=4):
        self.refiner, self.max_pending = refiner, max_pending
        self._todo, self._done = queue.Queue(), queue.Queue()
        self.pending = 0
        self._thread = threading.Thread(target=self._run, name="gaze", daemon=True)
        self._thread.start()

    def submit(self, job):
        if self.pending >= self.max_pending:
            return False
        self.pending += 1
        self._todo.put(job)
        return True

    def done(self):
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                break
        self.pending -= len(out)
        return out

    def close(self):
        self._todo.put(None)
        self._thread.join(timeout=5)

    def _run(self):
        while True:
            job = self._todo.get()
            if job is None:
                return
            try:
                best = own_face(self.refiner.detect(job.crop), job.box)
            except Exception:  # noqa: BLE001 - one bad crop must not stop gaze checks
                log.exception("gaze check failed")
                best = None
            self._done.put((job, best))


class InlineGaze:
    """GazeWorker's interface, refining at once in submit() (tests, replays)."""

    def __init__(self, refiner):
        self.refiner, self._done, self.pending = refiner, [], 0

    def submit(self, job):
        self._done.append((job, own_face(self.refiner.detect(job.crop), job.box)))
        return True

    def done(self):
        out, self._done = self._done, []
        return out

    def close(self):
        pass


def own_face(found, box):
    """The refiner's face that belongs to this track: its centre inside the track's box (crop pixels) or overlapping
    it by IoU >= 0.3, nearest to the box centre; None if none does. The crop is padded, so a neighbour's face (or
    the face above a gown) in the padding must not lend its quality, landmarks or gaze (2026-11-01)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    ok = []
    for f in found or []:
        cx, cy = f.centre
        inside = x1 <= cx <= x2 and y1 <= cy <= y2
        if inside or iou(f.box, box) >= 0.3:
            ok.append(f)
    return min(ok, key=lambda f: float(np.hypot(*(np.array(f.centre) - mid)))) if ok else None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def frontal(kps):
    """How straight the face looks at the camera, 0 (straight) .. 1+ (profile): nose offset from the
    middle of the eyes, in eye distances."""
    eye_l, eye_r, nose = kps[0], kps[1], kps[2]
    eye_d = float(np.hypot(*(eye_r - eye_l)))
    if eye_d < 1e-6:
        return 9.9
    return abs(float(nose[0] - (eye_l[0] + eye_r[0]) / 2)) / eye_d


class Vision:
    def __init__(self, detector, embedder, store, refiner=None, tracker=None, clock=time.monotonic,
                 enrol_min_px=config.FACE_ENROL_MIN_PX, max_yaw=0.35, collector=None, gaze_refiner=None,
                 spots=None, background=None):
        self.detector, self.store = detector, store
        # camera.Background: the camera's true pose from the picture (stillness relative to the background); None =
        # trust the commanded pose (tests, photo replays)
        self.background = background
        # an EmbedWorker (own thread) or anything with embed() (wrapped: embeds at once)
        self.worker = embedder if hasattr(embedder, "submit") else InlineEmbed(embedder)
        self.collector = collector              # robot/collect.Collector: keeps the looks (alive.py --collect)
        # small SCRFD: big-stream landmarks for gaze, ~GAZE_CHECK_S per face - a GazeWorker (own thread) or a plain
        # detector (wrapped: refines at once)
        self.gaze = None if gaze_refiner is None else (
            gaze_refiner if hasattr(gaze_refiner, "submit") else InlineGaze(gaze_refiner))
        self.gaze_waiting = set()               # track ids with a crop at the gaze refiner
        self.spots = spots                      # robot/static_spots.StaticSpots: things that keep looking like faces
        self._mode = "head"
        self.waiting = set()                    # track ids with a look at the embedder
        self.refiner = refiner                  # a (small-input) detector run on face crops for landmarks
        self.tracker = tracker or Tracker()
        self._clock = clock
        self.enrol_min_px, self.max_yaw = enrol_min_px, max_yaw
        self.track_vectors = {}                 # track id -> [(vector, quality info, crop)] of good looks
        self.asked = set()                      # merge questions already raised
        self.stats = dict(frames=0, detect_ms=0.0, embed_ms=0.0, embeds=0, head_dps=0.0)
        self.last_found = []

    @property
    def mode(self):
        """How head angles are known: "head" (a powered head) / "nohead" (no battery: angles are relative to wherever
        the limp head stopped). Places learnt in one mode mean nothing in the other (static spots, pictures)."""
        return self._mode

    @mode.setter
    def mode(self, m):
        self._mode = m
        if self.spots is not None:
            self.spots.mode = m

    # ---------- per frame ----------

    def process(self, frame, pose_at, speed_at=None):
        t = frame.t
        events = []
        rgb = frame.lores_rgb()
        h, w = rgb.shape[:2]
        t0 = self._clock()
        found = self.detector.detect(rgb)
        self.last_found = found                 # raw detections of this frame (for the recorder)
        self.stats["detect_ms"] = (self._clock() - t0) * 1000
        self.stats["frames"] += 1
        pose = pose_at(t)
        sightings = []
        for f in found:
            cx, cy = f.centre
            pan, tilt = camera.head_target(cx, cy, w, h, pose)
            sightings.append(Sighting(f, pan, tilt, camera.size_deg(f.width, w)))
        speed = speed_at(t) if speed_at else 0.0
        self.stats["head_dps"] = speed
        still = speed <= config.VISION_STILL_DPS
        bg = None
        if self.background is not None and still and (sightings or self.tracker.tracks):
            try:
                bg = self.background.update(frame.y_plane, pose, t)
            except Exception:  # noqa: BLE001 - an odd frame: this one does not count for stillness
                log.exception("background shift failed")
        counts = still and (self.background is None or bg is not None)     # unmeasured frames: not for Track.still
        new, ended = self.tracker.update(sightings, t, slack_deg=speed * config.VISION_SLACK_S, still=counts, bg=bg)
        for tr in self.tracker.tracks:              # a photo on the wall never counts as looking at the robot
            tr.picture = tr.still or (tr.person is not None and self.store.is_picture(tr.person))
        events += [dict(kind="face_new", track=tr.id, pan=tr.pan, tilt=tr.tilt) for tr in new
                   if not self._on_spot(tr)]
        for tr in ended:
            events += self._ended(tr)
        if self.gaze is not None:
            if still:
                self._gaze_checks(frame, t)
            self._gaze_results(t)
        for tr in self.tracker.tracks:
            if not (still and tr.id not in self.waiting and self.tracker.wants_embedding(tr, t)):
                continue
            if self._on_spot(tr) and not self.spots.check_due(tr.pan, tr.tilt, t):
                continue                            # the cushion again: only a check now and then, per spot
            look = self._cut(frame, tr, t, pose, speed)     # (check_due is only spent on a look really cut)
            tr.face_q = max(tr.face_q or 0.0, look.quality)
            if look.quality < config.FACE_LOOK_MIN_QUALITY:
                tr.last_embed_t = t                 # a hand, a shirt...: try again later, no vote
            elif self.worker.submit(look):
                self.waiting.add(tr.id)
        for look, vec in self.worker.done():
            self.waiting.discard(look.track)
            self.stats["embed_ms"] = look.ms
            if vec is not None:
                events += self._look_result(look, vec)
        for tr in list(self.tracker.tracks):        # a stranger that waited to prove it is alive
            if tr.unknown and len(self.track_vectors.get(tr.id, [])) >= 2 and self._may_enrol(tr):
                events += self._enrol(tr)
        for a, b in self.tracker.seen_together(t):
            self.store.apart(a, b)
        return events

    # ---------- one look at one track ----------

    def _cut(self, frame, tr, t, pose, speed):
        """Cut the face out of the big stream and align it (cheap: here, in the camera thread)."""
        crop, (x0, y0, s) = frame.crop_main(tr.sighting.face.box, pad=0.5)
        kps = tr.sighting.face.kps * s - [x0, y0]
        quality, facing = tr.sighting.face.score, None
        if self.refiner is not None:            # precise landmarks on the big crop - and is it a face at all?
            best = own_face(self.refiner.detect(crop), np.asarray(tr.sighting.face.box) * s - [x0, y0, x0, y0])
            quality = 0.0
            if best is not None:
                kps, quality = best.kps, best.score
                facing = gaze.facing(kps)
                if quality >= config.FACE_CONFIRM_QUALITY:
                    tr.update_gaze(kps, t, big=True)
        return Look(quality=quality, engagement=tr.engagement, facing=facing, track=tr.id, t=t, aligned=faces.align(crop, kps), crop=crop,
                    width_px=tr.sighting.face.width * s, yaw=frontal(kps),
                    luma=camera.mean_luma(frame.y_plane, tr.sighting.face.box), pan=tr.pan, tilt=tr.tilt,
                    head_dps=speed)

    def _gaze_checks(self, frame, t):
        """Confirmed faces not checked for GAZE_CHECK_S: cut their big-stream crop (here - cheap) and hand it to
        the gaze refiner (its own thread); _gaze_results() applies the landmarks on a later frame."""
        for tr in self.tracker.tracks:
            fresh = tr.big_gaze_t is not None and t - tr.big_gaze_t < config.GAZE_CHECK_S
            if tr.last_t != t or not tr.is_face or fresh or tr.id in self.gaze_waiting:
                continue
            crop, (x0, y0, s) = frame.crop_main(tr.sighting.face.box, pad=0.3)
            box = np.asarray(tr.sighting.face.box) * s - [x0, y0, x0, y0]
            if self.gaze.submit(GazeJob(tr.id, t, crop, box)):
                self.gaze_waiting.add(tr.id)

    def _gaze_results(self, t):
        """Landmarks back from the gaze refiner -> the track they were cut for (by id; a track that ended, or a
        result older than GAZE_STALE_S or than the track's last big-stream check, is dropped)."""
        for job, best in self.gaze.done():
            self.gaze_waiting.discard(job.track)
            tr = next((x for x in self.tracker.tracks if x.id == job.track), None)
            if tr is None or t - job.t > config.GAZE_STALE_S or (tr.big_gaze_t is not None and job.t <= tr.big_gaze_t):
                continue
            if best is not None:
                # applied now (engagement is smoothed over GAZE_TAU_S: one frame late does not matter), then dated
                # to the frame it came from, so the next check is due GAZE_CHECK_S after that frame
                tr.update_gaze(best.kps, t, big=True)
            tr.big_gaze_t = job.t                   # nothing there: try again later (small stream meanwhile)

    def _look_result(self, look, vec):
        """The embedder is back: match, vote, learn, enrol."""
        tr = next((x for x in self.tracker.tracks if x.id == look.track), None)
        self.stats["embeds"] += 1
        pid, sim = self.store.match(vec)
        if tr is None:                          # the face left while it was being embedded
            self._collect(look, pid, sim, None, vec)
            return []
        t, aligned = look.t, look.aligned
        events = []
        if self._jumped(tr, pid, sim, vec):         # now looking at a neighbour's face: new track for it
            old = tr
            tr = self.tracker.split(old, look.t, moved=pid)
            look.track = tr.id
            events = self._ended(old) + [dict(kind="face_new", track=tr.id, pan=tr.pan, tilt=tr.tilt)]
        good = (look.width_px >= self.enrol_min_px and look.yaw <= self.max_yaw
                and look.quality >= config.FACE_LEARN_MIN_QUALITY)
        matched = pid is not None and sim >= self.tracker.match_sim
        if good and not matched:                    # a stranger is enrolled only from looks that matched nobody
            keep = self.track_vectors.setdefault(tr.id, [])
            keep.append((vec, aligned))
            del keep[:-5]
        changed = self.tracker.vote(tr, pid, sim, t)
        self._collect(look, pid, sim, tr, vec)
        if tr.person is not None:
            if pid == tr.person and look.quality >= config.FACE_LEARN_MIN_QUALITY:
                self.store.learn(tr.person, vec, sim)
            if changed:
                self.store.seen(tr.person)
                events.append(dict(kind="person_seen", track=tr.id, person=tr.person,
                                   name=self.store.name(tr.person), sim=tr.sim, pan=tr.pan, tilt=tr.tilt))
        elif tr.unknown and changed:
            events.append(dict(kind="stranger", track=tr.id, pan=tr.pan, tilt=tr.tilt))
        if tr.unknown and len(self.track_vectors.get(tr.id, [])) >= 2 and self._may_enrol(tr):
            events += self._enrol(tr)
        return events

    def _jumped(self, tr, pid, sim, vec):
        """Has this decided track jumped to someone else's face? (TRACK_SWAP_LOOKS clear looks in a row)"""
        if tr.person is None or pid is None or pid == tr.person:
            tr.swap_looks = 0
            return False
        clearly_other = sim >= config.FACE_MATCH_SIM + config.TRACK_SWAP_MARGIN
        if clearly_other and self.store.similarity(tr.person, vec) < config.FACE_MATCH_SIM:
            tr.swap_looks += 1
        else:
            tr.swap_looks = 0                       # a look like both: one person stored twice, not a jump
        return tr.swap_looks >= config.TRACK_SWAP_LOOKS

    def _collect(self, look, pid, sim, tr, vec=None):
        if self.collector is not None:
            self.collector.save(look, match=pid, sim=sim, person=tr.person if tr else None,
                                name=self.store.name(tr.person) if tr and tr.person else None,
                                unknown=bool(tr and tr.unknown), vec=vec)

    def close(self):
        self.worker.close()
        if self.gaze is not None:
            self.gaze.close()
        if self.spots is not None:
            self.spots.save()

    def _enrol(self, tr):
        """A stranger with at least two good looks: store them as a new person (name comes later)."""
        looks = self.track_vectors.pop(tr.id)
        vecs = [v for v, _ in looks]
        photo = looks[-1][1]
        pid = self.store.enrol(vecs, photo=photo)
        tr.person, tr.unknown, tr.sim = pid, False, 1.0
        tr.votes.clear()
        return [dict(kind="enrolled", track=tr.id, person=pid, pan=tr.pan, tilt=tr.tilt)]

    def _may_enrol(self, tr):
        """Only a living face becomes a person: it has moved, or it was watched long enough to know it is not a
        picture (a framed photo on the wall was enrolled and then followed, 2026-11-01)."""
        if tr.still:
            return False
        secs, _ = tr.spread()
        return tr.moved or secs >= config.TRACK_STILL_S

    def _on_spot(self, tr):
        return self.spots is not None and not tr.is_face and self.spots.is_static(tr.pan, tr.tilt)

    def _ended(self, tr):
        self.track_vectors.pop(tr.id, None)
        if tr.person is not None and tr.path:       # evidence: is this person ever anywhere else? (a photo is not)
            pans, tilts = sorted(p[1] for p in tr.path), sorted(p[2] for p in tr.path)
            at = (pans[len(pans) // 2], tilts[len(tilts) // 2])
            if tr.still or tr.path[-1][0] - tr.path[0][0] >= 3.0:
                self.store.note_track(tr.person, still=tr.still, at=at, mode=self.mode)
        quiet = self._on_spot(tr)
        if self.spots is not None:
            self.spots.note_ended(tr)
        events = [] if quiet else [dict(kind="face_lost", track=tr.id, person=tr.person, face_q=tr.face_q,
                                        hits=tr.hits)]
        for a, b in self.tracker.same_person_evidence(tr):
            self.store.link(a, b)
        events += self.check_merges()
        return events

    def check_merges(self):
        """Merge clear duplicates now; raise each unclear pair once as a question."""
        events = []
        for c in self.store.merge_candidates():
            key = (c["keep"], c["drop"])
            if c["auto"]:
                try:
                    self.store.merge(c["keep"], c["drop"])
                except KeyError:                            # forgotten / renamed on another thread meanwhile
                    return events + self.check_merges()
                self.tracker.rename(c["drop"], c["keep"])
                events.append(dict(kind="merged", keep=c["keep"], drop=c["drop"], links=c["links"], sim=c["sim"]))
                return events + self.check_merges()       # ids changed: look again
            if key not in self.asked:
                self.asked.add(key)
                events.append(dict(kind="ask_same", keep=c["keep"], drop=c["drop"], sim=c["sim"]))
        return events

    # ---------- for the exposure loop ----------

    def main_face_luma(self, frame, attention_target=None):
        """Brightness of the face attention is on (else the biggest visible one), or None."""
        vis = self.tracker.visible(frame.t, faces_only=True)
        if not vis:
            return None
        tr = next((x for x in vis if x.id == attention_target), None) or max(vis, key=lambda x: x.sighting.size_deg)
        return camera.mean_luma(frame.y_plane, tr.sighting.face.box)


def track_summary(tracks, store):
    """One line per visible track, for logs."""
    out = []
    for tr in tracks:
        if tr.person:
            who = store.name(tr.person) or tr.person
        elif not tr.is_face:
            who = "not a face" if tr.face_q is not None else "unchecked"
        else:
            who = "stranger" if tr.unknown else "?"
        eye = f" eng {tr.engagement:.2f}{' LOOKING' if tr.looking else ''}"
        out.append(f"#{tr.id} {who} pan {tr.pan:+.0f} tilt {tr.tilt:+.0f} {tr.sighting.size_deg:.1f}deg{eye}")
    return ", ".join(out) if out else "-"
