import csv
import logging
import tempfile
import unittest
from pathlib import Path

from robot import config
from robot.safety import Safety
from robot.state import State
from tests.fakes import FakeClock, imu_reading


class SafetyTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.state = State(clock=self.clock)
        self.box = tempfile.TemporaryDirectory()
        self.addCleanup(self.box.cleanup)
        self.s = Safety(self.state, clock=self.clock, blackbox_dir=self.box.name)

    def imu(self, **kw):
        self.state.publish("imu", imu_reading(self.clock(), **kw))

    def motors(self, moving, settled_since=None):
        self.state.publish("motors", {"moving": moving, "settled_since": settled_since})

    def test_no_imu_means_stop(self):
        v = self.s.step()
        self.assertTrue(v["stop"])
        self.assertIn("imu missing", v["reasons"][0])
        self.state.publish("imu_status", "calibrating")
        self.assertIn("calibrating", self.s.step()["reasons"][0])

    def test_fresh_imu_means_go(self):
        self.imu()
        self.assertEqual(self.s.step(), {"stop": False, "reasons": [], "latched": []})
        self.assertEqual(self.state.get("safety")["stop"], False)

    def test_stale_imu_stops(self):
        self.imu()
        self.clock.sleep(config.SAFETY_IMU_STALE_S + 0.01)
        self.assertTrue(self.s.step()["stop"])

    def test_tipped_latches_until_reset_and_upright(self):
        self.imu(tipped=True)
        self.assertEqual(self.s.step()["reasons"], ["tipped"])
        self.imu(tipped=False)
        self.assertEqual(self.s.step()["reasons"], ["tipped"])     # still latched
        self.imu(tipped=True)
        self.s.reset()                                             # still tipped: stays
        self.assertEqual(self.s.step()["reasons"], ["tipped"])
        self.imu(tipped=False)
        self.s.reset()
        self.assertFalse(self.s.step()["stop"])

    def test_bump_while_driving_steadily_latches(self):
        self.motors(True, settled_since=self.clock() - 1.0)
        self.imu(last_bump_t=self.clock())
        v = self.s.step()
        self.assertEqual(v["reasons"], ["bump"])
        self.imu(last_bump_t=self.clock())
        self.s.reset()
        self.assertFalse(self.s.step()["stop"])                    # same bump not counted twice

    def test_bump_during_start_jerk_is_ignored(self):
        t = self.clock()
        self.motors(True, settled_since=None)                      # still ramping
        self.imu(last_bump_t=t)
        self.assertFalse(self.s.step()["stop"])
        self.motors(True, settled_since=t - 0.1)                   # settled 0.1 s before the bump
        self.assertFalse(self.s.step()["stop"])                    # inside the blank window, and already seen
        self.clock.sleep(1.0)
        self.motors(True, settled_since=t)
        self.imu(last_bump_t=self.clock())                         # a NEW bump 1 s later
        self.assertEqual(self.s.step()["reasons"], ["bump"])

    def test_fresh_bump_inside_the_blank_window_is_ignored(self):
        self.motors(True, settled_since=self.clock() - 0.1)      # reached its speed 0.1 s ago
        self.imu(last_bump_t=self.clock())                       # a new jolt now (< 0.3 s)
        self.assertFalse(self.s.step()["stop"])
        self.clock.sleep(config.SAFETY_BUMP_BLANK_S)
        self.imu(last_bump_t=self.clock())                       # a new jolt after the window
        self.assertEqual(self.s.step()["reasons"], ["bump"])

    def drive_with_pitch(self, seconds, pitch_rate, settled_since):
        """Feed IMU samples at 50 Hz while 'driving' (for the calm-before-bump history)."""
        self.motors(True, settled_since=settled_since)
        for _ in range(int(seconds * 50)):
            self.clock.sleep(0.02)
            self.imu(pitch_rate=pitch_rate)
            self.s.step()

    def test_nose_landing_after_take_off_is_not_a_bump(self):
        """Floor 2026-10-29: the nose lifts at take-off and lands ~0.5 s later with a thump,
        just after the 0.3 s blank window. The nose was moving -> ignore it."""
        start = self.clock()
        self.drive_with_pitch(0.6, pitch_rate=-140.0, settled_since=start)   # nose falling fast
        self.imu(last_bump_t=self.clock(), bump_g=0.8)                       # the landing thump
        self.assertFalse(self.s.step()["stop"])
        self.assertEqual(self.s.ignored_bumps, 1)

    def test_crash_out_of_calm_driving_is_a_bump(self):
        start = self.clock()
        self.drive_with_pitch(0.6, pitch_rate=-140.0, settled_since=start)   # take-off wobble...
        self.drive_with_pitch(0.6, pitch_rate=5.0, settled_since=start)      # ...then calm driving
        self.imu(last_bump_t=self.clock(), bump_g=1.5, pitch_rate=120.0)     # hit: the crash itself pitches
        self.assertEqual(self.s.step()["reasons"], ["bump"])

    def test_black_box_is_written_on_a_new_stop(self):
        start = self.clock()
        self.drive_with_pitch(1.0, pitch_rate=3.0, settled_since=start)
        self.clock.sleep(0.02)                                               # a new IMU sample
        self.imu(last_bump_t=self.clock(), bump_g=1.2)
        self.s.step()
        files = list(Path(self.box.name).glob("*-bump.csv"))
        self.assertEqual(len(files), 1)
        rows = list(csv.DictReader(open(files[0])))
        self.assertGreater(len(rows), 40)
        self.assertEqual(float(rows[-1]["t"]), 0.0)                          # times relative to the stop
        self.assertEqual(float(rows[-1]["bump_g"]), 1.2)
        self.s.step()                                                        # still latched: no 2nd file
        self.assertEqual(len(list(Path(self.box.name).glob("*.csv"))), 1)

    def test_manual_snapshot_writes_the_last_seconds_without_stopping(self):
        start = self.clock()
        self.drive_with_pitch(1.0, pitch_rate=3.0, settled_since=start)
        path = self.s.snapshot()
        self.assertTrue(path.name.endswith("-manual.csv"))
        self.assertGreater(len(list(csv.DictReader(open(path)))), 40)
        self.assertFalse(self.s.step()["stop"])              # saving is not a stop

    def test_snapshot_before_any_data(self):
        self.assertIsNone(self.s.snapshot())

    def test_stops_are_logged_once(self):
        with self.assertLogs("arcby", level="INFO") as logs:
            self.s.step()                    # imu missing
            self.s.step()                    # same: no new line
            self.imu()
            self.s.step()                    # ok
        lines = [l for l in logs.output if "safety" in l]
        self.assertEqual(len(lines), 2)
        self.assertIn("safety stop: imu missing", lines[0])
        self.assertIn("safety ok", lines[1])

    def test_bump_while_standing_is_ignored(self):
        self.motors(False)
        self.imu(last_bump_t=self.clock())
        self.assertFalse(self.s.step()["stop"])


if __name__ == "__main__":
    unittest.main()


class LowBatteryTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.state = State(clock=self.clock)
        self.box = tempfile.TemporaryDirectory()
        self.addCleanup(self.box.cleanup)
        self.s = Safety(self.state, clock=self.clock, blackbox_dir=self.box.name)

    def power(self, under, ok=True):
        self.state.publish("power", {"ok": ok, "under_voltage": under, "since_boot": under, "source": "test"})
        self.state.publish("imu", imu_reading(self.clock()))

    def run_for(self, seconds, under, ok=True):
        v = None
        for _ in range(int(seconds / 0.1)):
            self.clock.sleep(0.1)
            self.power(under, ok)
            v = self.s.step()
        return v

    def test_short_dip_is_not_a_stop(self):
        v = self.run_for(config.POWER_UV_HOLD_S - 0.2, True)
        self.assertFalse(v["stop"])
        v = self.run_for(1.0, False)
        self.assertFalse(v["stop"])

    def test_sustained_under_voltage_latches_low_battery(self):
        v = self.run_for(config.POWER_UV_HOLD_S + 0.5, True)
        self.assertEqual(v["reasons"], ["low battery"])
        self.assertEqual(len(list(Path(self.box.name).glob("*-low-battery.csv"))), 1)
        v = self.run_for(1.0, False)                   # voltage back (e.g. motors stopped)
        self.assertEqual(v["reasons"], ["low battery"])  # still latched: charge the battery!

    def test_reset_only_when_the_voltage_is_ok(self):
        self.run_for(config.POWER_UV_HOLD_S + 0.5, True)
        self.s.reset()
        self.assertEqual(self.run_for(0.1, True)["reasons"], ["low battery"])
        self.run_for(0.5, False)
        self.s.reset()
        self.assertFalse(self.run_for(0.1, False)["stop"])

    def test_no_power_information_is_not_a_stop(self):
        self.assertFalse(self.run_for(2.0, True, ok=False)["stop"])   # e.g. not on a Pi
        self.state = State(clock=self.clock)
        s = Safety(self.state, clock=self.clock, blackbox_dir=None)
        self.state.publish("imu", imu_reading(self.clock()))
        self.assertFalse(s.step()["stop"])                           # no "power" at all


class SafetyEdgeTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.state = State(clock=self.clock)

    def test_no_black_box_folder_configured(self):
        s = Safety(self.state, clock=self.clock, blackbox_dir=None)
        self.state.publish("imu", imu_reading(self.clock(), tipped=True))
        self.assertEqual(s.step()["reasons"], ["tipped"])          # still stops, just no file

    def test_black_box_that_cannot_be_written_does_not_break_safety(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / "box"
            blocker.write_text("a file where the folder should be")
            s = Safety(self.state, clock=self.clock, blackbox_dir=blocker)
            self.state.publish("imu", imu_reading(self.clock(), tipped=True))
            with self.assertLogs("arcby", level="ERROR") as logs:
                v = s.step()
            self.assertEqual(v["reasons"], ["tipped"])
            self.assertIn("black box not written", logs.output[-1])

    def test_run_loop_survives_falling_behind(self):
        import threading
        stop = threading.Event()

        class SlowClock:
            t, n = 0.0, 0
            def __call__(self):
                self.t += 0.05
                self.n += 1
                if self.n > 40:
                    stop.set()
                return self.t
        s = Safety(self.state, clock=SlowClock(), blackbox_dir=None)
        s.run(stop)
        self.assertTrue(stop.is_set())
        self.assertTrue(self.state.get("safety")["stop"])          # no IMU -> stop
