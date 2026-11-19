"""robot/legs.py: alive.py's drive - starts, every way it stops, and that each stop stops the motors in time."""
import logging
import threading
import time
import unittest

import numpy as np

from robot import config
from robot.imu import ImuFilter
from robot.legs import Legs
from robot.main import Robot
from robot.nav import Command
from robot.state import State
from tests.fakes import FakeDriver, imu_reading
from tests.test_main import NoisyMpu, wait_for


class FakeMotors:
    def __init__(self):
        self.reqs = []
        self.applied = (0.0, 0.0)

    def request_velocity(self, v, turn, source="?"):
        self.reqs.append((v, turn, source))


class FakeSafety:
    def __init__(self, state):
        self.state, self.resets = state, 0

    def reset(self):
        self.resets += 1


class FakeRobot:
    def __init__(self):
        self.state = State()
        self.motors = FakeMotors()
        self.safety = FakeSafety(self.state)
        self.alive, self.fault = True, None
        self.verdict()

    def verdict(self, latched=(), reasons=()):
        self.state.publish("safety", {"stop": bool(latched or reasons), "reasons": list(reasons) + list(latched),
                                      "latched": list(latched)})
        self.state.publish("imu", imu_reading(time.monotonic()))


class FakeSonar:
    def __init__(self, cm=120.0):
        self.last = (time.monotonic(), cm)


class StubNav:
    """Stands in for the Navigator: always drives forward (or asks for HELP)."""

    def __init__(self, cmd=None, state="DRIVE"):
        self.cmd, self.state, self.note = cmd or Command(v=0.4, turn=10.0, head=(5.0, -28.0)), state, ""
        self.stats, self.paused = {"driven_cm": 3.0}, 0

    def step(self, p):
        c = Command(v=self.cmd.v, turn=self.cmd.turn, head=self.cmd.head, state=self.state)
        return c

    def pause(self):
        self.paused += 1


class LegsTest(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.legs = Legs(self.robot, FakeSonar(), head_now=lambda: (0.0, -28.0))

    def drive(self):
        ok, why = self.legs.start("explore", why="test")
        self.assertTrue(ok, why)
        self.legs.nav = StubNav()
        return self.legs.step()

    def test_starts_drives_and_stops_on_request(self):
        self.assertIsNone(self.legs.head_target)
        cmd = self.drive()
        self.assertEqual(self.robot.motors.reqs[-1], (0.4, 10.0, "legs"))
        self.assertEqual(self.legs.head_target, (5.0, -28.0))
        self.assertEqual(cmd.state, "DRIVE")
        self.assertTrue(self.legs.stop("Arcby dur"))
        self.assertEqual(self.robot.motors.reqs[-1][:2], (0.0, 0.0))
        self.assertFalse(self.legs.active)
        self.assertIsNone(self.legs.head_target)
        self.assertEqual(self.legs.why, "Arcby dur")
        self.assertIsNone(self.legs.step())                       # stays stopped
        self.assertEqual(self.robot.motors.reqs[-1][:2], (0.0, 0.0))
        ev = self.legs.drain()
        self.assertEqual([(e["what"], e["why"]) for e in ev], [("started", "test"), ("stopped", "Arcby dur")])
        self.assertEqual(self.legs.drain(), [])

    def test_a_latched_safety_stop_ends_the_drive(self):
        self.drive()
        self.robot.verdict(latched=["bump"])
        self.legs.step()
        self.assertFalse(self.legs.active)
        self.assertEqual(self.legs.why, "safety: bump")

    def test_a_new_start_clears_a_bump_but_not_tipped(self):
        self.robot.verdict(latched=["bump"])
        self.assertTrue(self.legs.start("explore")[0])           # bump: reset() clears it (fake: verdict lags)
        self.assertEqual(self.robot.safety.resets, 1)
        self.legs.stop("x")
        self.robot.verdict(latched=["tipped"])
        ok, why = self.legs.start("explore")
        self.assertFalse(ok)
        self.assertIn("tipped", why)
        self.assertEqual(self.legs.drain()[-1]["what"], "refused")

    def test_a_dead_robot_refuses_and_stops(self):
        self.drive()
        self.robot.alive, self.robot.fault = False, "imu crashed"
        self.legs.step()
        self.assertFalse(self.legs.active)
        self.assertIn("imu crashed", self.legs.why)
        self.assertFalse(self.legs.start("explore")[0])

    def test_time_up(self):
        t = [100.0]
        self.legs.clock = lambda: t[0]
        self.assertTrue(self.legs.start("explore", max_s=120)[0])
        self.legs.nav = StubNav()
        self.legs.step()
        self.assertTrue(self.legs.active)
        t[0] += 121
        self.legs.step()
        self.assertFalse(self.legs.active)
        self.assertIn("time up", self.legs.why)

    def test_help_from_the_navigator_stops(self):
        self.legs.start("explore")
        self.legs.nav = StubNav(state="HELP")
        self.legs.step()
        self.assertFalse(self.legs.active)
        self.assertIn("stuck", self.legs.why)

    def test_a_failing_step_stops_the_drive_not_the_loop(self):
        self.legs.start("explore")

        class Broken(StubNav):
            def step(self, p):
                raise RuntimeError("boom")
        self.legs.nav = Broken()
        stop = threading.Event()
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        th = threading.Thread(target=self.legs.run, args=(stop,))
        th.start()
        self.assertTrue(wait_for(lambda: not self.legs.active, 1.0))
        self.assertIn("boom", self.legs.why)
        stop.set()
        th.join(1.0)
        self.assertFalse(th.is_alive())

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            self.legs.start("dance")

    def test_frames_only_while_driving_and_head_still(self):
        rgb = np.full((config.CAMERA_LORES[1], config.CAMERA_LORES[0], 3), 120, np.uint8)
        self.assertIsNone(self.legs.frame(rgb, 1.0, 0.0, -28.0, 0.0))
        self.legs.start("explore")
        self.assertIsNone(self.legs.frame(rgb, 1.0, 0.0, -28.0, 80.0))     # head moving fast: blurred
        self.assertIsNotNone(self.legs.frame(rgb, 1.0, 0.0, -28.0, 0.0))
        self.assertIsNotNone(self.legs.view)

    def test_status(self):
        self.drive()
        s = self.legs.status()
        self.assertEqual((s["mode"], s["state"], s["sonar"]), ("explore", "DRIVE", 120))
        self.legs.stop("dashboard")
        s = self.legs.status()
        self.assertEqual((s["mode"], s["why"]), ("stopped", "dashboard"))


class FollowPartsTest(unittest.TestCase):
    def track(self, tid, pan, t, person=None, first_t=None, box=(250, 40, 330, 290), height=40.0):
        from types import SimpleNamespace
        body = SimpleNamespace(box=np.array(box, np.float32), score=0.9)
        return SimpleNamespace(id=tid, pan=pan, tilt=0.0, last_t=t, first_t=t if first_t is None else first_t,
                               person=person, body=body, height_deg=height)

    def test_keeps_the_beckoner_then_refinds_them_when_the_track_breaks(self):
        from robot.legs import FollowTarget
        ft = FollowTarget(3, None, pan=10.0)
        self.assertEqual(ft.pick([self.track(1, -30, 1.0), self.track(3, 12, 1.0)], 1.0).id, 3)
        # track 3 is gone; a new body near where they were (and a far one) shows up
        got = ft.pick([self.track(1, -30, 2.0, first_t=0.0), self.track(7, 18, 2.0, first_t=1.8)], 2.0)
        self.assertEqual(got.id, 7)
        self.assertIsNone(ft.pick([self.track(1, -30, 3.0, first_t=0.0)], 3.0))      # an old body far off: not them

    def test_the_same_face_wins(self):
        from robot.legs import FollowTarget
        ft = FollowTarget(3, "emre", pan=0.0, t=0.0)
        got = ft.pick([self.track(9, -40, 1.0, person="emre"), self.track(8, 2, 1.0, first_t=0.9)], 1.0)
        self.assertEqual(got.id, 9)

    def test_target_person(self):
        from robot.legs import target_person
        w, h = config.CAMERA_LORES
        p = target_person(self.track(1, 15.0, 1.0, box=(250, 40, 330, h - 30)), (w, h), (0.0, config.NAV_TILT_DEG))
        self.assertEqual(p.bearing, 15.0)
        self.assertIsNotNone(p.dist)                            # feet in the picture
        p = target_person(self.track(1, 5.0, 1.0, box=(250, 0, 330, h)), (w, h), (0.0, 10.0))
        self.assertIsNone(p.dist)                               # cut off at both ends: the sonar has to say
        self.assertGreater(p.tilt, 0.0)


class FollowLegsTest(unittest.TestCase):
    def test_follow_starts_and_lost_them_stops(self):
        robot = FakeRobot()
        legs = Legs(robot, FakeSonar(), head_now=lambda: (0.0, 10.0))
        self.assertTrue(legs.start("follow", why="gel gel (Emre)")[0])
        legs.nav = StubNav(state="DONE")
        legs.nav.note = "lost them"
        legs.step()
        self.assertFalse(legs.active)
        self.assertEqual(legs.why, "lost them")
        self.assertEqual(legs.drain()[-1], dict(kind="drive", what="stopped", mode="follow", why="lost them"))

    def test_no_floor_check_while_following(self):
        legs = Legs(FakeRobot(), FakeSonar(), head_now=lambda: (0.0, 10.0))
        legs.start("follow")
        rgb = np.zeros((config.CAMERA_LORES[1], config.CAMERA_LORES[0], 3), np.uint8)
        self.assertIsNone(legs.frame(rgb, 1.0, 0.0, 10.0, 0.0))

    def test_target_reaches_the_navigator(self):
        from robot.nav import Person
        legs = Legs(FakeRobot(), FakeSonar(), head_now=lambda: (0.0, 10.0))
        legs.start("follow")
        legs.follow_target(Person(12.0, 150.0, tilt=8.0), 5.0)
        p = legs.percept()
        self.assertEqual((p.target.bearing, p.target_t), (12.0, 5.0))


class BoredomTest(unittest.TestCase):
    def setUp(self):
        from robot.legs import Boredom
        self.hour = [14]
        import datetime
        tz = datetime.timezone.utc
        self.b = Boredom(t=0.0, wall=lambda: datetime.datetime(2026, 11, 8, self.hour[0], 0, tzinfo=tz).timestamp(),
                         tz=tz)
        self.legs = Legs(FakeRobot(), FakeSonar(), head_now=lambda: (0.0, -28.0))

    def test_wanders_after_a_long_quiet_then_stops_for_someone(self):
        self.assertIsNone(self.b.update(config.BORED_EXPLORE_S - 1, False, self.legs))
        self.assertEqual(self.b.update(config.BORED_EXPLORE_S + 1, False, self.legs), "started")
        self.assertEqual((self.legs.mode, self.legs.why, self.legs.max_s), ("explore", "bored", config.WANDER_MAX_S))
        self.assertEqual(self.b.update(config.BORED_EXPLORE_S + 30, True, self.legs), "stopped")
        self.assertFalse(self.legs.active)

    def test_not_at_night(self):
        self.hour[0] = 23
        self.assertIsNone(self.b.update(config.BORED_EXPLORE_S * 3, False, self.legs))
        self.assertFalse(self.legs.active)

    def test_not_soon_after_a_person_stopped_it(self):
        t = config.BORED_EXPLORE_S * 2
        self.b.update(t, False, self.legs, [dict(kind="drive", what="stopped", why="voice: 'dur'")])
        self.assertIsNone(self.b.update(t + config.BORED_EXPLORE_S + 1, False, self.legs))
        self.assertEqual(self.b.update(t + config.BORED_AFTER_STOP_S + 1, False, self.legs), "started")

    def test_someone_does_not_stop_a_drive_it_was_asked_for(self):
        self.legs.start("explore", why="dashboard")
        self.assertIsNone(self.b.update(5.0, True, self.legs))
        self.assertTrue(self.legs.active)

    def test_after_a_wander_another_quiet_stretch(self):
        t = config.BORED_EXPLORE_S + 1
        self.b.update(t, False, self.legs)
        self.legs.stop("time up (120 s)")
        self.assertIsNone(self.b.update(t + 125, False, self.legs, self.legs.drain()))


class LegsOnTheRobotTest(unittest.TestCase):
    """Real motors, safety and IMU threads (fake chip + driver): every stop path brings the wheels to 0 within
    0.3 s (docs/navigation.md "Tests")."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.drv = FakeDriver()
        self.robot = Robot(driver=self.drv, imu_dev=NoisyMpu(), imu_filter=ImuFilter(calib_s=0.2),
                           power_reader=lambda: (False, False, "test"), blackbox_dir=None).__enter__()
        self.addCleanup(self.robot.__exit__, None, None, None)
        self.legs = Legs(self.robot, FakeSonar(), head_now=lambda: (0.0, -28.0))
        self.stop = threading.Event()
        self.th = threading.Thread(target=self.legs.run, args=(self.stop,), daemon=True)
        self.th.start()
        self.addCleanup(self.th.join, 1.0)
        self.addCleanup(self.stop.set)
        self.assertTrue(wait_for(lambda: not self.robot.state.get("safety")["stop"], 3.0))

    def driving(self, **kw):
        self.assertTrue(self.legs.start("explore", **kw)[0])
        self.legs.nav = StubNav(Command(v=0.5))
        self.assertTrue(wait_for(lambda: self.drv.out[0] > 0.3, 2.0), self.robot.state.get("motors"))

    def stops_within(self, s=0.3):
        t0 = time.monotonic()
        self.assertTrue(wait_for(lambda: self.drv.out == (0.0, 0.0), 1.0))
        self.assertLessEqual(time.monotonic() - t0, s)
        self.assertTrue(wait_for(lambda: not self.legs.active, 0.3))      # the drive knows by its next step

    def test_stop_request(self):
        self.driving()
        self.legs.stop("dashboard")
        self.stops_within()

    def test_bump(self):
        self.driving()
        with self.robot.safety._lock:
            self.robot.safety.latched.add("bump")
        self.stops_within()
        self.assertIn("bump", self.legs.why)

    def test_time_up(self):
        self.driving(max_s=0.5)
        time.sleep(0.4)
        self.stops_within(0.4)
        self.assertIn("time up", self.legs.why)

    def test_alive_stopping(self):
        self.driving()
        self.stop.set()
        self.stops_within()


if __name__ == "__main__":
    unittest.main()
