"""robot/static_spots.py: things that keep looking like faces at one place are learnt, remembered and forgotten."""
import tempfile
import unittest
from pathlib import Path

from robot.static_spots import StaticSpots
from robot.tracker import Sighting, Track


def ended(pan=-45.0, tilt=-20.0, hits=10, max_dev=0.1, face_q=0.0):
    return Track(1, Sighting(None, pan, tilt, 7.0), 0.0, 1.0, hits=hits, pos0=(pan, tilt), max_dev=max_dev,
                 face_q=face_q)


class StaticSpotsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "spots.json"
        self.now = [1000.0]

    def tearDown(self):
        self.dir.cleanup()

    def spots(self):
        return StaticSpots(self.path, radius_deg=2, min_hits=5, keep_s=3600, max_dev=0.5, clock=lambda: self.now[0])

    def test_a_still_never_a_face_place_becomes_a_spot(self):
        s = self.spots()
        for i in range(4):
            s.note_ended(ended())
        self.assertFalse(s.is_static(-45, -20))
        s.note_ended(ended())
        self.assertTrue(s.is_static(-45, -20))
        self.assertTrue(s.is_static(-46, -19))                      # within the radius
        self.assertFalse(s.is_static(-49, -20))
        self.assertFalse(s.is_static(0, 10))
        self.assertEqual(s.spots(), [(-45, -20)])

    def test_checked_once_per_spot_not_once_per_track(self):
        s = self.spots()
        self.assertTrue(s.check_due(-45, -20, 0.0))
        self.assertFalse(s.check_due(-45.5, -20, 3.0))             # another flicker of the same thing
        self.assertTrue(s.check_due(40, -20, 3.0))                  # another spot
        self.assertTrue(s.check_due(-45, -20, 11.0))

    def test_people_and_moving_things_do_not_count(self):
        s = self.spots()
        for _ in range(10):
            s.note_ended(ended(face_q=0.8))                         # a real face (passed the check)
            s.note_ended(ended(max_dev=1.6))                        # moved like people do
            s.note_ended(ended(hits=2, face_q=None))                # a flash nobody checked
            s.note_ended(ended(hits=1, face_q=0.6))                 # a flash that passed the face check
        self.assertFalse(s.is_static(-45, -20))

    def test_rejected_flickers_count(self):
        """Most of the churn: 1-2 frame boxes the face check turned down (face_q 0.0), at the same place."""
        s = self.spots()
        for _ in range(5):
            s.note_ended(ended(hits=1, face_q=0.0))
        self.assertTrue(s.is_static(-45, -20))

    def test_spread_not_running_max(self):
        """A still thing whose detector jitter once jumped 1 deg is still still (90 % of its positions)."""
        s = self.spots()
        for _ in range(5):
            tr = ended(max_dev=1.3)
            for i in range(30):
                tr.path.append((i * 0.1, -45.0 + (1.3 if i == 7 else 0.05 * (i % 3)), -20.0))
            s.note_ended(tr)
        self.assertTrue(s.is_static(-45, -20))
        tr = ended()                                                # a person swaying at the same place
        for i in range(30):
            tr.path.append((i * 0.1, -45.0 + 0.2 * i, -20.0))
        self.assertFalse(s.note_ended(tr))

    def test_head_and_no_head_spots_are_kept_apart_and_clear_forgets(self):
        s = self.spots()
        s.mode = "nohead"                                           # battery off: angles relative to a limp head
        for _ in range(5):
            s.note_ended(ended())
        self.assertTrue(s.is_static(-45, -20))
        s.mode = "head"
        self.assertFalse(s.is_static(-45, -20))
        self.assertEqual(s.spots(), [])
        s.mode = "nohead"
        s.clear()
        self.assertFalse(s.is_static(-45, -20))
        self.assertFalse(self.spots().hits)                         # also on disk

    def test_remembered_after_a_restart_and_forgotten_after_a_while(self):
        s = self.spots()
        for _ in range(5):
            s.note_ended(ended())
        s.save()
        self.assertTrue(self.spots().is_static(-45, -20))
        self.now[0] += 3601                                          # not seen for longer than keep_s
        self.assertFalse(self.spots().is_static(-45, -20))


class VisionSpotsTest(unittest.TestCase):
    def test_no_events_and_rare_checks_on_a_known_spot(self):
        from robot.people import PeopleStore
        from robot.vision import Vision
        from tests.test_vision import FakeFrame, ScriptDetector, ScriptEmbedder, face_at
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            spots = StaticSpots(Path(d) / "s.json", min_hits=2, clock=lambda: 5000.0)
            det = ScriptDetector()
            checks = []

            class NotAFace:
                def detect(self, crop):
                    checks.append(1)
                    return []
            v = Vision(det, ScriptEmbedder(lambda n: np.ones(8) / 8 ** 0.5), PeopleStore(d), refiner=NotAFace(),
                       spots=spots)
            events, t, learnt = [], 0.0, None
            for burst in range(4):                                  # the cushion flickers: 4 tracks in a row
                if burst == 2:
                    learnt = len(checks)
                for _ in range(5):
                    det.frames.append([face_at(100, 250, 30)])
                    events += v.process(FakeFrame(t), lambda t: (0.0, 0.0))
                    t += 0.1
                for _ in range(10):
                    det.frames.append([])
                    events += v.process(FakeFrame(t), lambda t: (0.0, 0.0))
                    t += 0.1
            kinds = [e["kind"] for e in events]
            self.assertEqual(kinds.count("face_new"), 2)            # learnt after 2 tracks: then quiet
            self.assertLessEqual(len(checks) - learnt, 1)           # then one check per spot, not per flicker


if __name__ == "__main__":
    unittest.main()
