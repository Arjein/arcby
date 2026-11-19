"""robot/gestures.py: "gel gel" (beckoning) and the stop sign from synthetic keypoint tracks - and that a wave, talking
with the hands, reaching or walking past is neither."""
import unittest

import numpy as np

from robot import body as B
from robot import config
from robot.gestures import Beckon, StopSign
from tests.test_body import FakeFrame, FakePose, standing, waving

HZ = 6.0                        # the pose model while an arm is up (BODY_EVERY_FAST_S = as fast as it can, ~160 ms)


def beckoning(t, hz=1.5, x0=288):
    """The right forearm up in front of the body, the wrist pumping toward the shoulder and back (up-down)."""
    b = standing(cx=x0)
    ex, ey = b.kps[B.R_ELBOW, :2]
    b.kps[B.R_WRIST, :2] = (ex + 3, ey - 45 + 14 * np.sin(2 * np.pi * hz * t))
    return b


def stop_sign(t, jitter=0.0, x0=288):
    """The right hand raised level with the head, palm to the robot, held still (a little jitter)."""
    b = standing(cx=x0)
    sx, sy = b.kps[B.R_SHOULDER, :2]
    b.kps[B.R_ELBOW, :2] = (sx - 15, sy + 20)
    b.kps[B.R_WRIST, :2] = (sx - 12 + jitter * np.sin(7 * t), sy - 30 + jitter * np.cos(5 * t))
    return b


def talking_hands(t, seed=0):
    """Talking with the hands: both forearms up-ish, wrists and elbows wandering around."""
    rng = np.random.default_rng(int(t * 1000) + seed)
    b = standing()
    for elbow, wrist in ((B.R_ELBOW, B.R_WRIST), (B.L_ELBOW, B.L_WRIST)):
        ex, ey = b.kps[elbow, :2] + rng.normal(0, 8, 2)
        b.kps[elbow, :2] = (ex, ey)
        b.kps[wrist, :2] = (ex + rng.normal(0, 15), ey - 30 + rng.normal(0, 15))
    return b


def reaching(t):
    """Lifting something: the whole arm goes up and down (elbow too)."""
    b = standing()
    lift = 40 * np.sin(2 * np.pi * 1.2 * t)
    b.kps[B.R_ELBOW, 1] -= lift
    b.kps[B.R_WRIST, :2] = (b.kps[B.R_ELBOW, 0], b.kps[B.R_ELBOW, 1] - 40 + 0.4 * lift)
    return b


def run(detector, make, seconds=3.0, hz=HZ):
    return [detector.update(make(i / hz), i / hz) for i in range(int(seconds * hz))]


class BeckonTest(unittest.TestCase):
    def test_gel_gel_is_seen(self):
        self.assertTrue(any(run(Beckon(), beckoning)))

    def test_slow_and_fast_beckons(self):
        for hz in (1.0, 2.0):
            self.assertTrue(any(run(Beckon(), lambda t: beckoning(t, hz=hz))), hz)

    def test_one_pump_is_not_enough(self):
        self.assertFalse(any(run(Beckon(), beckoning, seconds=0.5)))

    def test_a_wave_is_not_a_beckon(self):
        self.assertFalse(any(run(Beckon(), waving, hz=8.0)))

    def test_a_beckon_is_not_a_wave(self):
        d = B.WaveDetector()
        self.assertFalse(any(d.update(beckoning(i / HZ), i / HZ) for i in range(int(3 * HZ))))

    def test_a_still_raised_hand_is_not(self):
        self.assertFalse(any(run(Beckon(), lambda t: beckoning(0.0))))

    def test_talking_with_the_hands_is_not(self):
        for seed in range(5):
            self.assertFalse(any(run(Beckon(), lambda t: talking_hands(t, seed), seconds=20.0)), seed)

    def test_lifting_the_whole_arm_is_not(self):
        self.assertFalse(any(run(Beckon(), reaching)))

    def test_walking_past_with_an_arm_up_is_not(self):
        self.assertFalse(any(run(Beckon(), lambda t: beckoning(0.0, x0=150 + 100 * t))))

    def test_arm_down_is_not(self):
        self.assertFalse(any(run(Beckon(), lambda t: standing())))


class StopSignTest(unittest.TestCase):
    def test_a_raised_still_hand_is_a_stop_sign_after_the_hold(self):
        res = run(StopSign(), lambda t: stop_sign(t, jitter=2.0), seconds=2.0)
        first = res.index(True)
        self.assertGreaterEqual(first / HZ, config.GESTURE_STOP_HOLD_S - 1e-6)
        self.assertLess(first / HZ, config.GESTURE_STOP_HOLD_S + 0.4)

    def test_a_wave_is_not_a_stop_sign(self):
        self.assertFalse(any(run(StopSign(), waving, hz=8.0)))

    def test_hand_down_is_not(self):
        self.assertFalse(any(run(StopSign(), lambda t: standing())))

    def test_seen_from_the_side_is_not(self):
        def side(t):
            b = stop_sign(t)
            b.kps[B.L_SHOULDER, 2] = 0.0             # one shoulder hidden: not facing the robot
            return b
        self.assertFalse(any(run(StopSign(), side)))

    def test_lowering_the_hand_starts_the_hold_again(self):
        d = StopSign()
        for i in range(3):
            d.update(stop_sign(i / HZ), i / HZ)
        d.update(standing(), 3 / HZ)
        self.assertFalse(d.update(stop_sign(4 / HZ), 4 / HZ))


class SignEventsTest(unittest.TestCase):
    def events(self, make, seconds=4.0):
        sense = B.BodySense(FakePose(lambda i: [make(i / HZ)]), inline=True)
        out = []
        for i in range(int(seconds * HZ)):
            sense.next_t = -1
            out += sense.process(FakeFrame(i / HZ), lambda t: (0.0, 0.0), 0.0, [])
        return [e["kind"] for e in out]

    def test_a_beckon_makes_one_beckoned_event_and_no_wave(self):
        kinds = self.events(beckoning)
        self.assertEqual(kinds.count("beckoned"), 1)
        self.assertNotIn("waved", kinds)

    def test_a_wave_makes_no_beckon(self):
        kinds = self.events(waving)
        self.assertNotIn("beckoned", kinds)

    def test_a_stop_sign_event(self):
        kinds = self.events(lambda t: stop_sign(t, jitter=1.0))
        self.assertIn("stop_sign", kinds)
        self.assertNotIn("beckoned", kinds)


class ReplayTest(unittest.TestCase):
    def test_recorded_frames_replay_to_the_same_signs(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hwtest"))
        import signs

        def rec(t, b):
            return {"t": t, "head": [0.0, 0.0], "bodies": [dict(box=b.box.tolist(), score=b.score,
                                                                 kps=b.kps.tolist())]}
        rows = [rec(i / HZ, beckoning(i / HZ)) for i in range(int(4 * HZ))]
        rows += [{"t": 5.0, "head": [0, 0], "bodies": None}]               # no fresh pose result: skipped
        rows += [rec(6 + i / HZ, stop_sign(6 + i / HZ)) for i in range(int(2 * HZ))]
        res = signs.replay_records(rows)
        self.assertEqual([k for _, k, _ in res["signs"]].count("beckoned"), 1)
        self.assertIn("stop_sign", [k for _, k, _ in res["signs"]])
        self.assertGreater(res["bodies_s"], 4.0)


if __name__ == "__main__":
    unittest.main()
