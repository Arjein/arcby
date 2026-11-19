import unittest

from robot.state import State
from tests.fakes import FakeClock


class StateTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.state = State(clock=self.clock)

    def test_missing_key_is_none(self):
        self.assertIsNone(self.state.get("imu"))
        self.assertIsNone(self.state.age("imu"))

    def test_latest_value_wins(self):
        self.state.publish("imu", 1)
        self.state.publish("imu", 2)
        self.assertEqual(self.state.get("imu"), 2)

    def test_stale_value_counts_as_missing(self):
        self.state.publish("imu", 1)
        self.clock.sleep(0.2)
        self.assertEqual(self.state.get("imu", max_age=0.25), 1)
        self.assertAlmostEqual(self.state.age("imu"), 0.2)
        self.clock.sleep(0.1)
        self.assertIsNone(self.state.get("imu", max_age=0.25))

    def test_explicit_timestamp(self):
        self.state.publish("imu", 1, t=self.clock() - 1.0)
        self.assertIsNone(self.state.get("imu", max_age=0.5))


if __name__ == "__main__":
    unittest.main()
