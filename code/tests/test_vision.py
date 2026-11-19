"""robot/vision.py: frames -> tracks, identities, enrolment, duplicate merging (fake models, real store)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot import config
from robot.faces import ARCFACE_KPS, Face
from robot.people import PeopleStore
from robot.tracker import Tracker
from robot.vision import Vision, frontal
from tests.fakes import FakeClock

LORES = config.CAMERA_LORES
SCALE = config.CAMERA_MAIN[0] / LORES[0]


def face_at(cx, cy, width, yaw=0.0):
    """A face (lores pixels) centred at (cx, cy); yaw shifts the nose sideways (0 = frontal)."""
    k = ARCFACE_KPS.copy() * (width / 112) + [cx - width / 2, cy - width / 2]
    k[2, 0] += yaw * (k[1, 0] - k[0, 0])
    return Face((cx - width / 2, cy - width / 2, cx + width / 2, cy + width / 2), 0.9, k)


class FakeFrame:
    def __init__(self, t):
        self.t = t
        self.y_plane = np.full((LORES[1], LORES[0]), 120, np.uint8)

    def lores_rgb(self):
        return np.zeros((LORES[1], LORES[0], 3), np.uint8)

    def crop_main(self, box, pad=0.5):
        x1, y1, x2, y2 = box
        w = x2 - x1
        x0, y0 = int((x1 - pad * w) * SCALE), int((y1 - pad * w) * SCALE)
        n = int(w * (1 + 2 * pad) * SCALE) + 1
        return np.zeros((n, n, 3), np.uint8), (x0, y0, SCALE)


class ScriptDetector:
    """Returns the next frame's faces from a list."""

    def __init__(self):
        self.frames = []

    def detect(self, rgb):
        return self.frames.pop(0) if self.frames else []


class ScriptEmbedder:
    """embed() returns vectors from a function of the call number (tests decide who is 'seen')."""

    def __init__(self, who):
        self.who = who
        self.n = 0

    def embed(self, face):
        v = self.who(self.n)
        self.n += 1
        return v


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


class VisionTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.rng = np.random.default_rng(0)
        self.store = PeopleStore(self.dir.name, clock=FakeClock(0.0), match_sim=0.3)
        self.selim = unit(self.rng.normal(size=32))
        self.ayse = unit(self.rng.normal(size=32))
        self.det = ScriptDetector()

    def tearDown(self):
        self.dir.cleanup()

    def near(self, v, noise=0.05):
        return unit(v + self.rng.normal(0, noise, v.shape))

    def vision(self, who):
        return Vision(self.det, ScriptEmbedder(who), self.store,
                      tracker=Tracker(n_votes=3, match_sim=0.3, embed_every_s=0.4, recheck_s=3.0),
                      clock=lambda: 0.0)

    def run_frames(self, vision, faces_per_frame, dt=0.1, t0=0.0):
        events = []
        for i, fs in enumerate(faces_per_frame):
            self.det.frames.append(fs)
            events += vision.process(FakeFrame(t0 + i * dt), lambda t: (0.0, 0.0))
        return events

    def kinds(self, events):
        return [e["kind"] for e in events]

    def test_a_stranger_becomes_a_new_person(self):
        v = self.vision(lambda n: self.near(self.selim))
        # 30 px lores = 120 px big: enrolable; a living person sways a little (a still face waits, see below)
        ev = self.run_frames(v, [[face_at(288 + 2 * i, 150, 30)] for i in range(12)])
        self.assertEqual(self.kinds(ev)[:3], ["face_new", "stranger", "enrolled"])
        pid = ev[2]["person"]
        self.assertEqual(self.store.match(self.selim)[0], pid)
        self.assertEqual(v.tracker.tracks[0].person, pid)

    def test_a_face_that_never_moves_is_a_picture_never_enrolled_never_looking(self):
        """The framed photo on the wall (2026-11-01): enrolled, followed, 'looking' - now none of that."""
        v = self.vision(lambda n: self.near(self.selim))
        ev = self.run_frames(v, [[face_at(288, 150, 30)]] * 140)    # 14 s without moving
        self.assertNotIn("enrolled", self.kinds(ev))
        tr = v.tracker.tracks[0]
        self.assertTrue(tr.still)
        self.assertTrue(tr.picture)
        self.assertFalse(tr.looking)
        self.assertEqual(tr.engagement, 0.0)

    def test_a_photo_stays_still_relative_to_the_room_while_the_head_jitters(self):
        """With the head powered its pose error (1.1-1.6 deg) made the wall photo wobble like a sitting person
        (2026-11-01). Measured against the background it is still again; a person swaying 0.7 deg is not."""
        import math
        from robot import camera
        from tests.test_camera import room, view
        scene, rng = room(3), np.random.default_rng(5)
        f = camera.focal_px(LORES[0])

        def px(world_pan, world_tilt, pose):     # where a face at these head angles shows in the picture
            ax, ay = pose[0] - world_pan, world_tilt - pose[1]
            return LORES[0] / 2 + f * math.tan(math.radians(ax)), LORES[1] / 2 - f * math.tan(math.radians(ay))

        def run(background):
            det = ScriptDetector()
            v = Vision(det, ScriptEmbedder(lambda n: self.near(self.selim)), self.store, refiner=self.refined(0.7),
                       tracker=Tracker(n_votes=3, match_sim=0.3), background=background)
            err = np.zeros(2)
            for i in range(150):                    # 15 s; the head is told to hold (0, 0)
                t = i * 0.1
                err = np.clip(err + rng.normal(0, 0.4, 2), -1.5, 1.5)          # servo pose error, random walk
                true = tuple(err)
                person = (-8 + 0.7 * math.sin(t * 1.3), 4 + 0.4 * math.cos(t * 0.9))
                det.frames.append([face_at(*px(6, 5, true), 24), face_at(*px(*person, true), 24)])
                frame = FakeFrame(t)
                frame.y_plane = view(scene, *true, rng=rng)
                v.process(frame, lambda t: (0.0, 0.0))
            return sorted(v.tracker.tracks, key=lambda tr: -tr.pan)     # photo (pan 6) first
        photo, person = run(camera.Background())
        self.assertTrue(photo.still)
        self.assertTrue(photo.picture)
        self.assertFalse(person.still)
        photo, person = run(None)                   # the commanded pose alone: the photo seems to move
        self.assertFalse(photo.still)

    def test_a_still_face_is_enrolled_once_it_moves(self):
        v = self.vision(lambda n: self.near(self.selim))
        frames = [[face_at(288, 150, 30)]] * 20 + [[face_at(288 + 3 * i, 150, 30)] for i in range(10)]
        ev = self.run_frames(v, frames)
        self.assertIn("enrolled", self.kinds(ev))

    def test_a_person_seen_only_as_a_picture_is_marked(self):
        pid = self.store.enrol([self.selim])
        for _ in range(4):
            self.store.note_track(pid, still=True, at=(47.0, 17.0), mode="nohead")
        self.assertTrue(self.store.is_picture(pid))
        self.store.note_track(pid, still=False, at=(47.5, 16.0), mode="nohead")  # jitter at the same place: no news
        self.store.note_track(pid, still=False, at=(-58.0, 30.0), mode="head")   # other mode: nothing to compare
        self.store.note_track(pid, still=False, at=(10.0, 20.0), mode="nohead")  # one glitch is forgiven...
        self.assertTrue(self.store.is_picture(pid))
        self.store.note_track(pid, still=False, at=(-20.0, 20.0), mode="nohead")  # ...seen elsewhere again: alive
        self.assertFalse(self.store.is_picture(pid))
        self.store.set_picture(pid, True)                           # by hand wins
        self.assertTrue(self.store.is_picture(pid))
        v = self.vision(lambda n: self.near(self.selim))
        self.run_frames(v, [[face_at(288 + 2 * i, 150, 30)] for i in range(5)])
        tr = v.tracker.tracks[0]
        self.assertEqual(tr.person, pid)
        self.assertTrue(tr.picture)                                 # a known picture never looks at the robot

    def test_enrolment_uses_only_looks_that_matched_nobody(self):
        pid = self.store.enrol([self.selim])
        looks = iter([self.near(self.selim)] + [self.near(self.ayse)] * 30)
        v = self.vision(lambda n: next(looks))
        ev = self.run_frames(v, [[face_at(288 + 2 * i, 150, 30)] for i in range(40)])
        new = [e["person"] for e in ev if e["kind"] == "enrolled"]
        for p in new:                                               # Selim's look is not in the stranger's set
            self.assertLess(float((self.store.vectors[p] @ self.selim).max()), 0.5)
        self.assertEqual(self.store.match(self.selim)[0], pid)

    def test_the_refiner_face_must_be_the_tracks_own(self):
        """A face found in the padding of the crop (a neighbour, the face above a gown box) is not this one's."""
        from robot.vision import own_face
        box = (100, 100, 140, 140)
        mine = Face((105, 105, 135, 135), 0.8, np.zeros((5, 2)))
        neighbour = Face((60, 40, 95, 75), 0.9, np.zeros((5, 2)))
        self.assertIs(own_face([neighbour, mine], box), mine)
        self.assertIsNone(own_face([neighbour], box))
        self.assertIsNone(own_face([], box))
        overlap = Face((120, 125, 160, 165), 0.8, np.zeros((5, 2)))  # centre outside, IoU < 0.3
        self.assertIsNone(own_face([overlap], box))

    def test_a_known_person_is_recognised_on_the_first_good_look(self):
        pid = self.store.enrol([self.selim])
        self.store.set_name(pid, "Selim")
        v = self.vision(lambda n: self.near(self.selim))
        ev = self.run_frames(v, [[face_at(288, 150, 30)]] * 3)
        seen = [e for e in ev if e["kind"] == "person_seen"]
        self.assertEqual((seen[0]["person"], seen[0]["name"]), (pid, "Selim"))

    def test_no_looks_while_the_head_moves(self):
        v = self.vision(lambda n: self.near(self.selim))
        for i in range(12):
            self.det.frames.append([face_at(288, 150, 30)])
            v.process(FakeFrame(i * 0.1), lambda t: (0.0, 0.0), lambda t: 120.0)   # a saccade all along
        self.assertEqual(v.stats["embeds"], 0)
        self.assertEqual(len(v.tracker.tracks), 1)
        self.det.frames.append([face_at(288, 150, 30)])
        v.process(FakeFrame(1.2), lambda t: (0.0, 0.0), lambda t: 3.0)              # still again
        self.assertEqual(v.stats["embeds"], 1)

    def refined(self, score):
        """A refiner that finds the face on the crop with this score (crop of face_at(288, 150, 30))."""
        class Ref:
            def detect(s, crop):
                f = face_at(crop.shape[1] / 2, crop.shape[0] / 2, crop.shape[1] / 2)
                return [Face(f.box, score, f.kps)]
        return Ref()

    def test_not_a_face_on_the_big_crop_is_not_even_embedded(self):
        v = Vision(self.det, ScriptEmbedder(lambda n: self.near(self.selim)), self.store, refiner=self.refined(0.1),
                   tracker=Tracker(n_votes=3, match_sim=0.3, embed_every_s=0.4, recheck_s=3.0))
        ev = self.run_frames(v, [[face_at(288, 150, 30)]] * 12)
        self.assertEqual(v.stats["embeds"], 0)
        self.assertEqual(self.kinds(ev), ["face_new"])               # a hand: tracked, never a stranger
        self.assertFalse(v.tracker.tracks[0].is_face)
        self.assertEqual(v.tracker.visible(1.1, faces_only=True), [])  # so attention never looks at it

    def test_a_real_face_is_confirmed_on_its_first_look(self):
        v = Vision(self.det, ScriptEmbedder(lambda n: self.near(self.selim)), self.store, refiner=self.refined(0.7))
        self.run_frames(v, [[face_at(288, 150, 30)]])
        self.assertEqual(len(v.tracker.visible(0.0, faces_only=True)), 1)

    def test_a_weak_face_votes_but_is_never_enrolled_or_learnt(self):
        pid = self.store.enrol([self.selim])
        v = Vision(self.det, ScriptEmbedder(lambda n: unit(self.selim + 0.9 * self.ayse)), self.store,
                   refiner=self.refined(0.5), tracker=Tracker(n_votes=3, match_sim=0.3, embed_every_s=0.4))
        ev = self.run_frames(v, [[face_at(288, 150, 30)]] * 12)
        self.assertIn("person_seen", self.kinds(ev))                 # still recognised...
        self.assertEqual(len(self.store.vectors[pid]), 1)            # ...but nothing learnt from it
        self.store.forget(pid)
        v2 = self.vision(lambda n: self.near(self.ayse))
        v2.refiner = self.refined(0.5)
        ev = self.run_frames(v2, [[face_at(288, 150, 30)]] * 12, t0=5.0)
        self.assertIn("stranger", self.kinds(ev))
        self.assertNotIn("enrolled", self.kinds(ev))

    def test_gaze_from_big_stream_landmarks(self):
        def gaze_ref(yaw):
            class Ref:
                def detect(s, crop):
                    f = face_at(crop.shape[1] / 2, crop.shape[0] / 2, crop.shape[1] / 2, yaw=yaw)
                    return [Face(f.box, 0.8, f.kps)]
            return Ref()
        for yaw, looking in ((0.0, True), (0.6, False)):
            self.det.frames.clear()
            v = Vision(self.det, ScriptEmbedder(lambda n: self.near(self.selim)), self.store,
                       refiner=self.refined(0.7), gaze_refiner=gaze_ref(yaw))
            self.run_frames(v, [[face_at(288, 150, 30)]] * 20)          # 2 s in view
            self.assertEqual(v.tracker.tracks[0].looking, looking)

    def test_gaze_checks_run_in_their_own_thread_and_land_on_the_right_track(self):
        """The gaze refiner held up the camera loop (7-10 fps with faces in view, 2026-11-01): now a GazeWorker."""
        import threading
        from robot.vision import GazeWorker
        gate = threading.Event()

        class SlowRef:
            def __init__(s):
                s.calls = 0

            def detect(s, crop):
                s.calls += 1
                gate.wait(5)
                f = face_at(crop.shape[1] / 2, crop.shape[0] / 2, crop.shape[1] / 2)
                return [Face(f.box, 0.8, f.kps)]
        ref = SlowRef()
        worker = GazeWorker(ref)
        v = Vision(self.det, ScriptEmbedder(lambda n: self.near(self.selim)), self.store,
                   refiner=self.refined(0.7), gaze_refiner=worker)
        try:
            self.run_frames(v, [[face_at(150, 150, 30), face_at(420, 150, 30)]] * 5)
            self.assertEqual(len(v.gaze_waiting), 2)                     # one crop per face at a time...
            self.assertEqual(ref.calls, 1)                               # ...and the camera loop never waited
            gate.set()
            for _ in range(200):
                if worker._done.qsize() >= 2:
                    break
                threading.Event().wait(0.01)
            self.run_frames(v, [[face_at(150, 150, 30), face_at(420, 150, 30)]], t0=0.5)
            self.assertEqual(v.gaze_waiting, set())
            self.assertEqual(len(v.tracker.tracks), 2)
            for tr in v.tracker.tracks:                                  # each face got its own result back
                self.assertGreaterEqual(tr.big_gaze_t, 0.3)
                self.assertGreater(tr.engagement, 0.0)
        finally:
            gate.set()
            v.close()

    def test_gaze_results_for_a_gone_track_or_too_late_are_dropped(self):
        from robot.vision import GazeJob

        class Held:
            """A gaze worker whose results the test hands out."""
            def __init__(s):
                s.jobs, s.out = [], []

            def submit(s, job):
                s.jobs.append(job)
                return True

            def done(s):
                out, s.out = s.out, []
                return out

            def close(s):
                pass
        held = Held()
        v = Vision(self.det, ScriptEmbedder(lambda n: self.near(self.selim)), self.store,
                   refiner=self.refined(0.7), gaze_refiner=held)
        self.run_frames(v, [[face_at(150, 150, 30)]] * 2)
        tr = v.tracker.tracks[0]
        kps = face_at(50, 50, 30).kps
        front = Face((35, 35, 65, 65), 0.8, kps)
        held.out = [(GazeJob(tr.id + 99, 0.1, None, None), front),                     # no such track
                    (GazeJob(tr.id, 0.2 - config.GAZE_STALE_S - 0.05, None, None), front)]  # too old
        before = tr.engagement
        v._gaze_results(0.2)
        self.assertEqual(tr.engagement, before)
        held.out = [(GazeJob(tr.id, 0.25, None, None), front)]
        v._gaze_results(0.3)
        self.assertGreater(tr.engagement, before)
        self.assertEqual(tr.big_gaze_t, 0.25)

    def test_far_or_turned_faces_are_never_enrolled(self):
        v = self.vision(lambda n: self.near(self.selim))
        ev = self.run_frames(v, [[face_at(288, 150, 6)]] * 12)      # 6 px lores = 24 px big: too small
        ev += self.run_frames(v, [[face_at(100, 150, 30, yaw=0.6)]] * 12, t0=5.0)   # profile
        self.assertIn("stranger", self.kinds(ev))
        self.assertNotIn("enrolled", self.kinds(ev))
        self.assertEqual(self.store.people(), [])

    def test_learns_new_looks_of_a_known_person(self):
        pid = self.store.enrol([self.selim])
        side = unit(self.selim + 0.9 * self.ayse)                    # same person, another angle
        v = self.vision(lambda n: side)
        self.run_frames(v, [[face_at(288, 150, 30)]] * 3)
        self.assertEqual(len(self.store.vectors[pid]), 2)

    def test_two_people_in_one_frame_are_marked_apart(self):
        pm, pa = self.store.enrol([self.selim]), self.store.enrol([self.ayse])
        seq = iter([self.selim, self.ayse] * 20)
        v = self.vision(lambda n: next(seq))
        self.run_frames(v, [[face_at(150, 150, 30), face_at(420, 150, 30)]] * 2)
        self.assertIn("|".join(sorted((pm, pa))), self.store.apart_pairs)

    def test_one_person_stored_twice_is_merged_by_track_evidence(self):
        p1 = self.store.enrol([self.near(self.selim, 0.08)])
        p2 = self.store.enrol([self.near(self.selim, 0.08)])        # enrolled again on a bad day
        v = self.vision(lambda n: self.store.vectors[[p1, p2][n % 2]][0])   # one track matches both
        ev = self.run_frames(v, [[face_at(288, 150, 30)]] * 100)    # 10 s in view
        ev += self.run_frames(v, [[]] * 30, t0=10.0)                # the person leaves (3 s): track ends
        merged = [e for e in ev if e["kind"] == "merged"]
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(self.store.people()), 1)

    def test_a_track_that_jumps_to_a_neighbour_is_split(self):
        pm, pa = self.store.enrol([self.selim]), self.store.enrol([self.ayse])
        v = self.vision(lambda n: self.near(self.selim) if n < 1 else self.near(self.ayse))
        ev = self.run_frames(v, [[face_at(288, 150, 30)]] * 60)       # Selim, then the box holds Ayşe's face
        lost = [e for e in ev if e["kind"] == "face_lost"]
        self.assertEqual([e["person"] for e in lost], [pm])            # the Selim part ended as Selim
        self.assertEqual(v.tracker.tracks[0].person, pa)               # the face now followed is Ayşe
        self.assertEqual(self.store.links, {})                         # no "same person" evidence

    def test_similar_ids_without_evidence_raise_one_question(self):
        self.store.enrol([self.near(self.selim, 0.08)])
        self.store.enrol([self.near(self.selim, 0.08)])
        v = self.vision(lambda n: self.ayse)
        ev = v.check_merges() + v.check_merges()
        self.assertEqual(self.kinds(ev), ["ask_same"])
        self.assertEqual(len(self.store.people()), 2)


    def test_vision_waits_for_the_worker_and_never_asks_twice(self):
        from robot.vision import EmbedWorker
        import threading
        gate = threading.Event()
        calls = []

        class Slow:
            def embed(s, face):
                calls.append(1)
                gate.wait(2)
                return self.near(self.selim)
        v = Vision(self.det, EmbedWorker(Slow()), self.store,
                   tracker=Tracker(n_votes=3, match_sim=0.3, embed_every_s=0.4, recheck_s=3.0))
        self.run_frames(v, [[face_at(288, 150, 30)]] * 10)   # 1 s: would want 3 looks, the model is busy
        self.assertEqual(len(calls), 1)
        gate.set()
        for _ in range(200):                                 # until the worker has handed the result back
            if not v.worker._done.empty():
                break
            threading.Event().wait(0.01)
        self.run_frames(v, [[face_at(288, 150, 30)]], t0=1.0)
        self.assertEqual(v.stats["embeds"], 1)               # the result came back and was used
        v.close()

    def test_keeps_looks_rate_limited_and_capped(self):
        import json
        from robot.collect import Collector
        pid = self.store.enrol([self.selim])
        self.store.set_name(pid, "Selim")
        with tempfile.TemporaryDirectory() as d:
            c = Collector(d, every_s=2.0, max_bytes=10**9, clock=lambda: 1793462160.0, alike_sim=1.01,
                          per_bin=100)                               # balance rules: tests/test_collect.py
            v = Vision(self.det, ScriptEmbedder(lambda n: self.near(self.selim)), self.store, collector=c,
                       tracker=Tracker(n_votes=3, match_sim=0.3, embed_every_s=0.4, recheck_s=0.4))
            self.run_frames(v, [[face_at(288, 150, 30)]] * 60)  # 6 s in view: looks every ~0.4 s
            lines = [json.loads(x) for x in (Path(d) / "looks.jsonl").read_text().splitlines()]
            self.assertEqual(len(lines), 3)                     # one per 2 s: ~0, ~2, ~4 s
            self.assertEqual((lines[-1]["person"], lines[-1]["name"]), (pid, "Selim"))
            self.assertTrue(all((Path(d) / x["face"]).exists() and (Path(d) / x["crop"]).exists() for x in lines))
            c.max_bytes = c.used                                # full
            self.run_frames(v, [[face_at(288, 150, 30)]] * 30, t0=6.0)
            self.assertEqual(len((Path(d) / "looks.jsonl").read_text().splitlines()), 3)


class EmbedWorkerTest(unittest.TestCase):
    def test_slow_embedder_runs_beside_the_camera_loop(self):
        import threading
        from robot.vision import EmbedWorker, Look
        gate = threading.Event()

        class Slow:
            def embed(self, face):
                gate.wait(2)
                return unit(np.ones(4))
        w = EmbedWorker(Slow(), max_pending=2)
        look = lambda i: Look(i, 0.0, np.zeros((112, 112, 3), np.uint8), None, 50, 0, 100, 0, 0, 0)  # noqa: E731
        self.assertTrue(w.submit(look(1)))
        self.assertTrue(w.submit(look(2)))
        self.assertFalse(w.submit(look(3)))                 # full: try again in a later frame
        self.assertEqual(w.done(), [])                      # submit() did not wait for the model
        gate.set()
        got = []
        for _ in range(100):
            got += w.done()
            if len(got) == 2:
                break
            threading.Event().wait(0.01)
        self.assertEqual(sorted(lk.track for lk, _ in got), [1, 2])
        self.assertEqual(w.pending, 0)
        w.close()

class FrontalTest(unittest.TestCase):
    def test_straight_and_turned(self):
        self.assertLess(frontal(face_at(100, 100, 50).kps), 0.05)
        self.assertGreater(frontal(face_at(100, 100, 50, yaw=0.6).kps), 0.5)


if __name__ == "__main__":
    unittest.main()
