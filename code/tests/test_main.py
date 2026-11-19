"""The whole robot wired together: fake IMU chip + fake motor driver, real threads."""
import logging
import random
import time
import unittest

from robot.imu import ImuFilter
from robot.main import Robot
from tests.fakes import FakeDriver, FakeMpu6050


class NoisyMpu(FakeMpu6050):
    def __init__(self):
        super().__init__()
        self.rnd = random.Random(5)
        self.accel = (0.0, 0.0, -1.0)        # chip down, robot flat

    def _sample(self):
        self.temp_c = 25.0 + self.rnd.uniform(-0.05, 0.05)
        return super()._sample()


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class RobotTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)       # the crash test logs a traceback on purpose
        self.addCleanup(logging.disable, logging.NOTSET)

    def make(self, dev=None):
        self.drv = FakeDriver()
        self.dev = dev or NoisyMpu()
        self.power = [False]
        return Robot(driver=self.drv, imu_dev=self.dev, imu_filter=ImuFilter(calib_s=0.2),
                     power_reader=lambda: (self.power[0], False, "test"), blackbox_dir=None)

    def test_no_motion_until_imu_ready_then_drives(self):
        with self.make() as r:
            r.motors.request(0.5, 0.5, "test")
            time.sleep(0.05)
            v = r.state.get("safety")
            self.assertTrue(v["stop"])
            self.assertIn("imu missing", v["reasons"][0])
            self.assertEqual(self.drv.out, (0.0, 0.0))

            def driving():
                r.motors.request(0.5, 0.5, "test")
                return self.drv.out[0] > 0.3
            self.assertTrue(wait_for(driving), r.state.get("motors"))
            self.assertFalse(r.state.get("safety")["stop"])
        self.assertEqual(self.drv.out, (0.0, 0.0))
        self.assertTrue(self.drv.closed)

    def test_sender_goes_quiet_motors_stop(self):
        with self.make() as r:
            self.assertTrue(wait_for(lambda: not r.state.get("safety")["stop"]))
            for _ in range(30):
                r.motors.request(0.5, 0.5, "test")
                time.sleep(0.02)
            self.assertGreater(self.drv.out[0], 0)
            self.assertTrue(wait_for(lambda: self.drv.out == (0.0, 0.0), timeout=1.0))
            self.assertEqual(r.state.get("motors")["reason"], "command timeout")

    def test_lost_imu_stops_motors(self):
        with self.make() as r:
            self.assertTrue(wait_for(lambda: not r.state.get("safety")["stop"]))
            def driving():
                r.motors.request(0.5, 0.5, "test")
                return self.drv.out[0] > 0.3
            self.assertTrue(wait_for(driving))
            self.dev.fail_reads = 10_000                 # I2C wire pulled
            def stopped():
                r.motors.request(0.5, 0.5, "test")
                return self.drv.out == (0.0, 0.0)
            self.assertTrue(wait_for(stopped, timeout=1.0))
            self.assertIn("imu missing", r.state.get("safety")["reasons"][0])

    def test_low_battery_stops_the_motors(self):
        with self.make() as r:
            def driving():
                r.motors.request(0.5, 0.5, "test")
                return self.drv.out[0] > 0.3
            self.assertTrue(wait_for(driving))
            self.power[0] = True                         # battery sagging: Pi under-voltage
            def stopped():
                r.motors.request(0.5, 0.5, "test")
                return self.drv.out == (0.0, 0.0)
            self.assertTrue(wait_for(stopped, timeout=3.0))
            self.assertIn("low battery", r.state.get("safety")["reasons"])

    def test_crashing_thread_stops_everything(self):
        with self.make() as r:
            self.assertTrue(wait_for(lambda: not r.state.get("safety")["stop"]))
            def boom():
                raise RuntimeError("test crash")
            r.safety.step = boom                         # the safety loop dies
            self.assertTrue(wait_for(lambda: not r.alive, timeout=1.0))
            self.assertIn("safety crashed", r.fault)
            self.assertTrue(wait_for(lambda: self.drv.out == (0.0, 0.0), timeout=1.0))
        self.assertTrue(self.drv.closed)

    def test_imu_not_answering_fails_at_start_and_releases_motors(self):
        dev = NoisyMpu()
        dev.fail_reads = 1
        with self.assertRaises(OSError):
            with self.make(dev):
                pass
        self.assertTrue(self.drv.closed)


if __name__ == "__main__":
    unittest.main()


class RobotRealBusTest(unittest.TestCase):
    def test_opens_and_closes_the_real_i2c_device_when_none_is_given(self):
        from unittest import mock
        from robot import i2c
        fake = NoisyMpu()
        fake.closed = False
        fake.close = lambda: setattr(fake, "closed", True)
        with mock.patch.object(i2c, "I2CDevice", return_value=fake) as opened:
            with Robot(driver=FakeDriver(), imu_filter=ImuFilter(calib_s=0.2),
                       power_reader=lambda: (False, False, "test"), blackbox_dir=None) as r:
                self.assertTrue(r.alive)
        opened.assert_called_once_with("/dev/i2c-1", 0x68)
        self.assertTrue(fake.closed)                      # we opened it, so we close it
