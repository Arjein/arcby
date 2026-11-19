import unittest

from robot import config
from robot.teleop import Teleop, car_velocity


class TeleopTest(unittest.TestCase):
    def setUp(self):
        self.t = Teleop(speed=0.4, turn=0.1, lease_s=0.6)

    def test_nothing_pressed_is_stop(self):
        self.assertEqual(self.t.command(0.0), (0.0, 0.0, "stop"))

    def test_hold_to_drive(self):
        self.t.press("w", 10.0)
        self.assertEqual(self.t.command(10.5), (0.4, 0.4, "forward"))
        self.assertEqual(self.t.command(10.61), (0.0, 0.0, "stop"))     # let go -> stops
        for i in range(20):                                             # held: auto-repeat every 33 ms
            self.t.press("w", 20.0 + i * 0.033)
        self.assertEqual(self.t.command(20.0 + 19 * 0.033 + 0.5)[2], "forward")

    def test_all_drive_keys(self):
        cases = {"w": (0.4, 0.4), "s": (-0.4, -0.4), "a": (-0.1, 0.1), "d": (0.1, -0.1),
                 "q": (0.2, 0.4), "e": (0.4, 0.2), "W": (0.4, 0.4)}
        for k, want in cases.items():
            self.t.press(k, 0.0)
            l, r, _ = self.t.command(0.1)
            self.assertAlmostEqual(l, want[0], msg=k)
            self.assertAlmostEqual(r, want[1], msg=k)

    def test_space_stops_at_once(self):
        self.t.press("w", 0.0)
        self.t.press(" ", 0.1)
        self.assertEqual(self.t.command(0.2)[2], "stop")

    def test_new_key_replaces_old(self):
        self.t.press("w", 0.0)
        self.t.press("a", 0.1)
        self.assertEqual(self.t.command(0.2)[2], "spin left")

    def test_speed_limits(self):
        for _ in range(20):
            self.t.press("+", 0.0)
        self.assertEqual(self.t.speed, 1.0)
        for _ in range(20):
            self.t.press("-", 0.0)
        self.assertEqual(self.t.speed, 0.1)

    def test_reset_and_quit(self):
        self.t.press("r", 0.0)
        self.assertTrue(self.t.want_reset)
        self.t.press("x", 0.0)
        self.assertTrue(self.t.want_quit)
        t2 = Teleop()
        t2.press("\x03", 0.0)
        self.assertTrue(t2.want_quit)

    def test_unknown_key_changes_nothing(self):
        self.t.press("w", 0.0)
        self.assertIsNone(self.t.press("z", 0.1))
        self.assertEqual(self.t.command(0.2)[2], "forward")

    def test_lease_shorter_than_nothing(self):
        # the motors' own watchdog must be shorter than the key lease would be pointless
        self.assertLess(config.DRIVE_CMD_TIMEOUT_S, config.KEY_LEASE_S)


if __name__ == "__main__":
    unittest.main()


class CarVelocityTest(unittest.TestCase):
    def test_nothing_pressed(self):
        self.assertEqual(car_velocity(0, 0, 0.5, 90), (0.0, 0.0))

    def test_gas_and_reverse(self):
        self.assertEqual(car_velocity(1, 0, 0.5, 90), (0.5, 0.0))
        self.assertEqual(car_velocity(-1, 0, 0.5, 90), (-0.5, 0.0))

    def test_steering_forward_and_standing(self):
        self.assertEqual(car_velocity(1, 1, 0.5, 90), (0.5, -90.0))     # W + D: right = negative
        self.assertEqual(car_velocity(1, -1, 0.5, 90), (0.5, 90.0))     # W + A: left
        self.assertEqual(car_velocity(0, 1, 0.5, 90), (0.0, -90.0))     # D standing: turn right

    def test_reversing_steers_like_a_car(self):
        # S + D: back + right -> nose swings LEFT -> positive turn rate
        self.assertEqual(car_velocity(-1, 1, 0.5, 90), (-0.5, 90.0))

    def test_inputs_are_clamped(self):
        self.assertEqual(car_velocity(5, -5, 0.5, 90), (0.5, 90.0))

    def test_no_minus_zero(self):
        import math
        for throttle in (1, 0, -1):
            v, turn = car_velocity(throttle, 0, 0.5, 90)
            self.assertEqual(math.copysign(1, turn), 1.0)            # never -0.0
