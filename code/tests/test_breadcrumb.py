import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from robot import breadcrumb
from robot.breadcrumb import Breadcrumb, describe, doing, read_last


class BreadcrumbTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "sub" / "crumb.ring"
        self.t = [1762613002.0]

    def crumb(self, prog="nav.py", pwm=None, state=None, slots=5):
        return Breadcrumb(prog, pwm=pwm, state=state, path=self.path, power=lambda: ("0x0", 0.86),
                          clock=lambda: self.t[0], slots=slots, slot=192, fsync=False)

    def test_the_last_line_survives_and_is_read_back(self):
        c = self.crumb(pwm=lambda: (0.6, -0.6), state=lambda: "explore/TURN")
        self.assertIsNone(c.open())
        for _ in range(3):
            c.write()
            self.t[0] += 1
        last = read_last(self.path, 192)                # no close(): the power went
        self.assertEqual(last["seq"], 2)
        self.assertEqual(last["pwm"], [0.6, -0.6])
        self.assertEqual(last["doing"], "spinning")
        self.assertEqual(last["state"], "explore/TURN")
        self.assertEqual(last["throttled"], "0x0")
        self.assertNotIn("exit", last)

    def test_the_file_never_grows_and_wraps_around(self):
        c = self.crumb(slots=5)
        c.open()
        for _ in range(12):
            c.write()
        self.assertEqual(self.path.stat().st_size, 5 * 192)
        self.assertEqual(read_last(self.path, 192)["seq"], 11)

    def test_a_new_run_continues_the_count_and_returns_the_previous_line(self):
        a = self.crumb(prog="nav.py", pwm=lambda: (0.4, 0.4))
        a.open()
        a.write()
        a.write()
        b = self.crumb(prog="alive.py")
        prev = b.open()
        self.assertEqual((prev["prog"], prev["seq"], prev["doing"]), ("nav.py", 1, "forward"))
        b.write()
        self.assertEqual(read_last(self.path, 192)["prog"], "alive.py")
        self.assertEqual(read_last(self.path, 192)["seq"], 2)

    def test_a_clean_stop_writes_an_exit_line(self):
        stop = threading.Event()
        c = self.crumb()
        c.start(stop)
        stop.set()
        c.close()
        last = read_last(self.path, 192)
        self.assertTrue(last["exit"])
        self.assertIn("clean stop", describe(last, "x"))

    def test_close_stops_its_own_thread_also_without_the_stop_event(self):
        c = self.crumb()
        c.start(threading.Event())
        c.close()
        self.assertFalse(c.thread.is_alive())

    def test_a_half_written_slot_is_skipped(self):
        c = self.crumb()
        c.open()
        c.write()
        c.write()
        with open(self.path, "r+b") as f:               # slot 1 cut in the middle by the power
            f.seek(192 + 10)
            f.write(b"\x00" * 40)
        self.assertEqual(read_last(self.path, 192)["seq"], 0)

    def test_no_file_and_a_failing_write_never_raise(self):
        self.assertIsNone(read_last(self.path, 192))
        c = Breadcrumb("x", path="/proc/nope/crumb.ring", power=None, fsync=False)
        c.open()
        self.assertIsNotNone(c.failed)
        self.assertIsNone(c.write())
        c.close()

    def test_a_broken_pwm_or_state_callback_is_left_out(self):
        c = self.crumb(pwm=lambda: 1 / 0, state=lambda: [][1])
        c.open()
        rec = c.write()
        self.assertIsNone(rec["pwm"])
        self.assertIsNone(rec["state"])

    def test_lines_fit_their_slot(self):
        c = self.crumb(state=lambda: "x" * 500)
        c.open()
        c.write()
        raw = self.path.read_bytes()[:192]
        self.assertTrue(raw.endswith(b"\n"))
        self.assertLessEqual(len(json.loads(raw)["state"]), 40)

    def test_describe_tells_a_power_cut_from_a_crash(self):
        rec = dict(seq=4, wall=1762613002.0, boot="aaaa", prog="nav.py", pwm=[0.6, -0.6], doing="spinning",
                   state="explore/TURN", throttled="0x0", core_v=0.86)
        cut = describe(rec, "bbbb")
        self.assertIn("power cut", cut)
        self.assertIn("spinning", cut)
        self.assertIn("PWM 0.60/-0.60", cut)
        self.assertIn("killed or crashed", describe(dict(rec, boot="a0bcd92a"), "a0bcd92a-9c1e-4c2b-8f4e-2b7d1e0c5a11"))
        self.assertIn("no previous run", describe(None))

    def test_doing(self):
        self.assertEqual(doing((0.0, 0.0)), "standing")
        self.assertEqual(doing((0.5, -0.5)), "spinning")
        self.assertEqual(doing((-0.4, -0.4)), "backward")
        self.assertEqual(doing((0.2, 0.5)), "turning")
        self.assertIsNone(doing(None))

    def test_pi_power_without_vcgencmd(self):
        from unittest import mock
        with mock.patch.object(breadcrumb.subprocess, "run", side_effect=OSError):
            self.assertEqual(breadcrumb.pi_power(), (None, None))


class RobotBreadcrumbTest(unittest.TestCase):
    def test_the_robot_writes_its_pwm_only_when_asked_with_fakes(self):
        from robot import config
        from robot.imu import ImuFilter
        from robot.main import Robot
        from tests.fakes import FakeDriver
        from tests.test_main import NoisyMpu
        with tempfile.TemporaryDirectory() as d:
            old = config.BREADCRUMB_FILE
            config.BREADCRUMB_FILE = os.path.join(d, "c.ring")
            self.addCleanup(setattr, config, "BREADCRUMB_FILE", old)
            kw = dict(driver=FakeDriver(), imu_dev=NoisyMpu(), imu_filter=ImuFilter(calib_s=0.2),
                      power_reader=lambda: (False, False, "test"), blackbox_dir=None)
            with Robot(**kw) as r:
                self.assertIsNone(r.breadcrumb)                 # fakes: no breadcrumb by default
            with Robot(breadcrumb=True, **kw) as r:
                self.assertIsNotNone(r.breadcrumb)
            last = read_last(config.BREADCRUMB_FILE)
            self.assertTrue(last["exit"])
            self.assertEqual(last["pwm"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
