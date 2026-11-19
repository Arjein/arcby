"""robot/collect.py: balanced face collection - only new kinds of look, per person and condition bin."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot.collect import Collector, bin_of
from robot.vision import Look


def look(track=1, t=0.0, width=120, yaw=0.05, luma=90, engagement=0.8):
    return Look(track=track, t=t, aligned=np.zeros((112, 112, 3), np.uint8), crop=np.zeros((60, 60, 3), np.uint8),
                width_px=width, yaw=yaw, luma=luma, pan=0, tilt=0, head_dps=0, engagement=engagement)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


class CollectTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.rng = np.random.default_rng(0)
        self.face = unit(self.rng.normal(size=64))

    def tearDown(self):
        self.dir.cleanup()

    def collector(self, **kw):
        return Collector(self.root / "faces", every_s=0.0, max_bytes=10**9, clock=lambda: 1793462160.0,
                         state_path=self.root / "collect_state.json", **kw)

    def lines(self):
        return [json.loads(x) for x in (self.root / "faces" / "looks.jsonl").read_text().splitlines()]

    def test_the_same_face_at_the_same_desk_is_kept_once(self):
        c = self.collector()
        saved = [c.save(look(t=i), person="selim", vec=self.face) for i in range(500)]   # hours at the desk
        self.assertEqual(sum(saved), 1)
        self.assertEqual(c.skipped, 499)

    def test_a_full_disk_switches_collecting_off_never_crashes(self):          # review 2026-11-01
        from unittest import mock
        c = self.collector()
        with mock.patch("robot.collect.Image.fromarray", side_effect=OSError(28, "No space left on device")):
            self.assertFalse(c.save(look(t=0), person="selim", vec=self.face))
        self.assertTrue(c.failed)
        self.assertFalse(c.save(look(track=2, t=100), person="emre", vec=-self.face))   # stays off

    def test_new_conditions_still_get_in(self):
        c = self.collector()
        c.save(look(t=0), person="selim", vec=self.face)
        self.assertTrue(c.save(look(t=1, luma=30), person="selim", vec=self.face))      # dark now
        self.assertTrue(c.save(look(t=2, yaw=0.5), person="selim", vec=self.face))      # side view
        self.assertTrue(c.save(look(t=3, width=50), person="selim", vec=self.face))     # far away
        self.assertTrue(c.save(look(t=4, engagement=0.0), person="selim", vec=self.face))   # not looking
        self.assertEqual(len({x["bin"] for x in self.lines()}), 5)

    def test_a_bin_holds_a_few_different_looks(self):
        c = self.collector(per_bin=4)
        saved = [c.save(look(t=i), person="selim", vec=unit(self.rng.normal(size=64))) for i in range(10)]
        self.assertEqual(sum(saved), 4)                                                 # all different, quota 4

    def test_counts_survive_a_restart_but_not_the_night(self):
        c = self.collector(per_bin=2)
        for i in range(2):
            c.save(look(t=i), person="selim", vec=unit(self.rng.normal(size=64)))
        again = self.collector(per_bin=2)
        self.assertFalse(again.save(look(t=5), person="selim", vec=unit(self.rng.normal(size=64))))
        tomorrow = Collector(self.root / "faces", every_s=0.0, max_bytes=10**9, clock=lambda: 1793462160.0 + 86400,
                             state_path=self.root / "collect_state.json", per_bin=2)
        self.assertTrue(tomorrow.save(look(t=6), person="selim", vec=unit(self.rng.normal(size=64))))

    def test_strangers_a_few_looks_per_track(self):
        c = self.collector(stranger_per_track=3)
        saved = [c.save(look(track=7, t=i, luma=20 * i), vec=unit(self.rng.normal(size=64))) for i in range(10)]
        self.assertEqual(sum(saved), 3)
        self.assertTrue(c.save(look(track=8, t=11), vec=unit(self.rng.normal(size=64))))    # someone else

    def test_bins(self):
        self.assertEqual(bin_of(look()), "near/straight/dim/looking")
        self.assertEqual(bin_of(look(width=40, yaw=0.6, luma=200, engagement=0.1)), "far/side/bright/away")


if __name__ == "__main__":
    unittest.main()
