"""robot/session_rec.py: a guided test session recorded by the running robot (frames while the flag says so)."""
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from robot.session_rec import SessionRecorder, save_crop


class Frame:
    lores_size = (64, 36)

    def __init__(self, t):
        self.t = t

    def crop_main(self, box, pad=0.4):
        return np.full((144, 256, 3), 90, np.uint8), (0, 0, 4.0)


class SessionRecTest(unittest.TestCase):
    def test_records_only_while_the_flag_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            now = [1000.0]
            flag = Path(d) / "session.json"
            rec = SessionRecorder(flag=flag, fps=4.0, clock=lambda: now[0])
            for i in range(8):                              # no flag: nothing
                rec.frame(Frame(i))
                now[0] += 0.1
            flag.write_text(json.dumps({"dir": str(Path(d) / "s1"), "until": now[0] + 3.1}))
            now[0] += 1.1                                   # the flag is read at most once a second
            for i in range(30):                             # 3 s at 10 fps: 2 s of it inside "until"
                rec.frame(Frame(i))
                now[0] += 0.1
            rec.close()
            idx = [json.loads(x) for x in (Path(d) / "s1" / "frames.jsonl").read_text().splitlines()]
            self.assertTrue(7 <= len(idx) <= 9, len(idx))   # 4 a second for ~2 s
            self.assertEqual(len(list((Path(d) / "s1" / "frames").glob("*.jpg"))), len(idx))

    def test_a_crop_with_what_was_found(self):
        with tempfile.TemporaryDirectory() as d:
            save_crop(d, 3, np.zeros((40, 40, 3), np.uint8), {"held": 1.2, "best": ["cell phone", 0.7]})
            self.assertTrue((Path(d) / "crops" / "00003.jpg").exists())
            self.assertEqual(json.loads((Path(d) / "crops.jsonl").read_text())["best"][0], "cell phone")


if __name__ == "__main__":
    unittest.main()
