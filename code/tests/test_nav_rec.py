import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from robot import config
from robot.floor import FloorView
from robot.nav import Command, Percept
from robot.nav_rec import NavRecorder


def lines(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def view(t=0.0):
    return FloorView(t, 0.0, -28.0, 6.3, 0.0, np.array([40.0, 60.0]), np.array([5.0, -5.0]),
                     np.array([True, False]), learnt=True)


class NavRecorderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rec = NavRecorder(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_steps_all_while_driving_one_a_second_while_standing(self):
        for i in range(10):                               # 0.5 s standing at 20 Hz
            self.rec.step(Percept(t=i * 0.05), Command(), False)
        for i in range(10):                               # 0.5 s driving
            self.rec.step(Percept(t=1.0 + i * 0.05, yaw=3.0, sonar=80.0), Command(v=0.04, state="DRIVE"), True)
        self.rec.close()
        got = lines(self.rec.dir / "nav.jsonl.gz")
        self.assertEqual(len(got), 1 + 10)
        self.assertEqual(got[-1]["state"], "DRIVE")
        self.assertEqual(got[-1]["sonar"], 80.0)
        self.assertTrue(got[-1]["drive"])

    def test_pictures_only_while_driving_rate_limited_and_lossless(self):
        rgb = np.random.default_rng(1).integers(0, 255, (324, 576, 3), dtype=np.uint8)
        self.assertIsNone(self.rec.frame(rgb, view(), 0.0, 100.0))         # not driving yet
        self.rec.step(Percept(t=0.0), Command(), True)
        self.assertIsNotNone(self.rec.frame(rgb, view(), 0.1, 100.0))
        self.assertIsNone(self.rec.frame(rgb, view(), 0.1 + config.NAV_REC_FRAME_S / 2, 100.0))
        name = self.rec.frame(rgb, view(), 0.1 + config.NAV_REC_FRAME_S, float("inf"))
        self.assertIsNotNone(name)
        self.rec.close()
        back = np.asarray(Image.open(self.rec.frames_dir / name))
        self.assertTrue(np.array_equal(back, rgb))
        got = lines(self.rec.dir / "floor.jsonl.gz")
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["hit"], [True, False])
        self.assertEqual(got[0]["ex"], [40.0, 60.0])
        self.assertIsNone(got[1]["sonar_free"])                             # inf = nothing near -> null
        self.assertTrue(got[0]["learnt"])

    def test_meta_names_the_program(self):
        meta = json.loads((Path(self.rec.dir) / "meta.json").read_text())
        self.assertEqual(meta["program"], "nav.py")


if __name__ == "__main__":
    unittest.main()
