"""Unit tests for robot/imu.py against a simulated MPU6050.

Run on the Pi or the Mac (no hardware, standard library only):
    python3 -m unittest discover -s code/tests -t code -v
"""
import math
import random
import struct
import threading
import time
import unittest

from robot import config, imu
from robot.imu import Imu, ImuError, ImuFilter, Mpu6050
from robot.state import State
from tests.fakes import FakeClock, FakeMpu6050

DT = 0.01   # 100 Hz


def pack(*vals):
    return struct.pack(">7h", *vals)


def calibrated_filter(bias=(0.0, 0.0, 0.0), accel=(0.0, 0.0, 1.0), t0=0.0, **kw):
    """ImuFilter fed a still calibration period; returns (filter, t of its last sample)."""
    f = ImuFilter(**kw)
    t = t0 - DT
    while f.calibrating:
        t += DT
        f.update(t, accel, bias, 25.0)
    return f, t


# ---------- raw data ----------

class DecodeTest(unittest.TestCase):
    def test_signed_big_endian(self):
        raw = bytes([0x7F, 0xFF, 0x80, 0x00, 0xFF, 0xFF, 0x00, 0x00,
                     0x00, 0x01, 0x12, 0x34, 0xED, 0xCC])
        self.assertEqual(imu.decode(raw), (32767, -32768, -1, 0, 1, 0x1234, -0x1234))

    def test_wrong_length(self):
        with self.assertRaises(ImuError):
            imu.decode(bytes(13))

    def test_units(self):
        accel, gyro, temp = imu.to_units((8192, -4096, 0, 0, 328, -164, 0))
        self.assertEqual(accel, (1.0, -0.5, 0.0))
        self.assertAlmostEqual(gyro[0], 10.0)       # +-1000 deg/s: 32.8 LSB per deg/s
        self.assertAlmostEqual(gyro[1], -5.0, places=9)
        self.assertAlmostEqual(temp, 36.53)
        self.assertAlmostEqual(imu.to_units((0, 0, 0, -3240, 0, 0, 0))[2], 27.0006, places=3)

    def test_saturation_ignores_temperature(self):
        self.assertFalse(imu.is_saturated((0, 0, 0, -32768, 0, 0, 0)))
        self.assertTrue(imu.is_saturated((0, 0, 0, 0, 0, 32767, 0)))
        self.assertTrue(imu.is_saturated((-32768, 0, 0, 0, 0, 0, 0)))


class FactoryTrimTest(unittest.TestCase):
    def test_code_one_everywhere(self):
        # accel code 1 = high bits 0, low bits 01 for x/y/z -> 0x10 = 0b00_01_01_01
        accel, gyro = imu.factory_trim([0x01, 0x01, 0x01, 0x15])
        for a in accel:
            self.assertAlmostEqual(a, 4096 * 0.34)
        self.assertEqual([round(g, 6) for g in gyro], [3275.0, -3275.0, 3275.0])

    def test_code_31_everywhere(self):
        accel, gyro = imu.factory_trim([0xFF, 0xFF, 0xFF, 0x3F])
        for a in accel:
            self.assertAlmostEqual(a, 4096 * 0.92)
        self.assertAlmostEqual(gyro[0], 3275 * 1.046 ** 30)

    def test_bits_per_axis(self):
        # accel x=22 (0b101_10), y=9 (0b010_01), z=3 (0b000_11); gyro x=5, y=17, z=30
        accel, gyro = imu.factory_trim([0xA5, 0x51, 0x1E, 0x27])

        def ft_a(c):
            return 4096 * 0.34 * (0.92 / 0.34) ** ((c - 1) / 30)

        def ft_g(c):
            return 3275 * 1.046 ** (c - 1)
        for got, want in zip(accel, [ft_a(22), ft_a(9), ft_a(3)]):
            self.assertAlmostEqual(got, want)
        for got, want in zip(gyro, [ft_g(5), -ft_g(17), ft_g(30)]):
            self.assertAlmostEqual(got, want)

    def test_zero_code_means_no_trim(self):
        accel, gyro = imu.factory_trim([0, 0, 0, 0])
        self.assertEqual(accel + gyro, [0.0] * 6)
        axis = imu.SelfTestAxis("gyro x", 0.0, 100.0)
        self.assertIsNone(axis.change)
        self.assertFalse(axis.ok)

    def test_limit_14_percent(self):
        self.assertTrue(imu.SelfTestAxis("a", 1000.0, 1139.0).ok)
        self.assertFalse(imu.SelfTestAxis("a", 1000.0, 1141.0).ok)
        self.assertTrue(imu.SelfTestAxis("a", -1000.0, -861.0).ok)
        self.assertFalse(imu.SelfTestAxis("a", -1000.0, 1000.0).ok)


# ---------- chip driver ----------

class Mpu6050Test(unittest.TestCase):
    def setUp(self):
        self.fake = FakeMpu6050()
        self.chip = Mpu6050(self.fake, sleep=lambda s: None)

    def test_init_resets_wakes_and_configures(self):
        self.assertEqual(self.chip.init(), 0x68)
        self.assertEqual(self.fake.writes[0], (imu.PWR_MGMT_1, 0x80))
        self.assertEqual(self.fake.regs[imu.PWR_MGMT_1], 0x01)
        for reg, value in imu.RUN_CONFIG.items():
            self.assertEqual(self.fake.regs[reg], value)
        self.assertEqual(self.fake.regs[imu.SMPLRT_DIV], 9)

    def test_init_converts_like_the_chip(self):
        self.chip.init()
        self.fake.accel = (0.5, -0.25, 1.0)
        self.fake.gyro = (100.0, -200.0, 0.0)
        accel, gyro, temp = imu.to_units(imu.decode(self.chip.read_raw()))
        self.assertEqual(accel, (0.5, -0.25, 1.0))
        for got, want in zip(gyro, (100.0, -200.0, 0.0)):
            self.assertAlmostEqual(got, want, places=9)
        self.assertAlmostEqual(temp, 25.0, places=2)

    def test_clone_id_accepted(self):
        self.fake.who = 0x70
        self.fake.regs[imu.WHO_AM_I] = 0x70
        self.assertEqual(self.chip.init(), 0x70)

    def test_unknown_id_rejected(self):
        self.fake.who = 0x00
        self.fake.regs[imu.WHO_AM_I] = 0x00
        with self.assertRaisesRegex(ImuError, "not an MPU6050"):
            self.chip.init()

    def test_register_that_does_not_stick(self):
        self.fake.ignore_writes = {imu.GYRO_CONFIG}
        with self.assertRaisesRegex(ImuError, "0x1b"):
            self.chip.init()

    def test_not_connected_raises_oserror(self):
        self.fake.fail_reads = 1
        with self.assertRaises(OSError):
            self.chip.init()

    def test_self_test_pass(self):
        self.chip.init()
        self.fake.set_trim([0x01, 0x01, 0x01, 0x15])
        self.fake.st_response = [1392.64, 1392.64, 1392.64, 3275.0, -3275.0, 3275.0]
        self.fake.accel = (0.1, 0.0, -1.0)     # gravity cancels out
        result = self.chip.self_test(samples=5)
        self.assertEqual([r.axis for r in result],
                         ["accel x", "accel y", "accel z", "gyro x", "gyro y", "gyro z"])
        for r in result:
            self.assertTrue(r.ok, r)
            self.assertLess(abs(r.change), 0.001)

    def test_self_test_catches_one_bad_axis(self):
        self.chip.init()
        self.fake.set_trim([0x01, 0x01, 0x01, 0x15])
        self.fake.st_response = [1392.64, 1392.64, 1392.64, 3275.0, -3275.0, 3275.0 * 0.8]
        result = {r.axis: r for r in self.chip.self_test(samples=5)}
        self.assertFalse(result["gyro z"].ok)
        self.assertAlmostEqual(result["gyro z"].change, -0.2, places=3)
        self.assertTrue(all(r.ok for name, r in result.items() if name != "gyro z"))

    def test_self_test_restores_run_config(self):
        self.chip.init()
        self.chip.self_test(samples=2)
        self.assertEqual(self.fake.regs[imu.GYRO_CONFIG], imu.RUN_CONFIG[imu.GYRO_CONFIG])
        self.assertEqual(self.fake.regs[imu.ACCEL_CONFIG], imu.RUN_CONFIG[imu.ACCEL_CONFIG])


# ---------- axis mapping ----------

class AxesTest(unittest.TestCase):
    def test_level_trim_turns_the_rest_stance_upright(self):
        from robot.imu import level_rotation, rotate
        up = (-0.02, 0.147, 1.038)                               # measured on the robot, ~8 deg off
        m = level_rotation(up)
        x, y, z = rotate(m, up)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertGreater(z, 1.0)
        for v in ((1, 0, 0), (0, 1, 0), (0.3, -0.2, 0.9)):        # a rotation: lengths stay
            self.assertAlmostEqual(math.hypot(*rotate(m, v)), math.hypot(*v), places=9)
        self.assertIsNone(level_rotation(None))
        self.assertIsNone(level_rotation((0, 0, 1)))

    def test_parse(self):
        self.assertEqual(imu.parse_axes(("+x", "-y", "-z")), ((0, 1), (1, -1), (2, -1)))
        self.assertEqual(imu.parse_axes(("+y", "-x", "+z")), ((1, 1), (0, -1), (2, 1)))

    def test_mirror_rejected(self):
        with self.assertRaisesRegex(ValueError, "mirror"):
            imu.parse_axes(("+x", "+y", "-z"))

    def test_bad_specs(self):
        for spec in [("+x", "+x", "+z"), ("x", "+y", "+z"), ("+x", "+y"), ("+a", "+y", "+z")]:
            with self.assertRaises(ValueError, msg=spec):
                imu.parse_axes(spec)

    def test_configured_axes_are_valid_and_upside_down(self):
        mapping = imu.parse_axes(config.IMU_AXES)
        # chip down, robot flat: the sensor feels gravity on -z; robot must see +1 g up
        self.assertEqual(imu.apply_axes(mapping, (0.0, 0.0, -1.0))[2], 1.0)

    def test_apply(self):
        m = imu.parse_axes(("+y", "-x", "+z"))
        self.assertEqual(imu.apply_axes(m, (1, 2, 3)), (2, -1, 3))


# ---------- filter ----------

class CalibrationTest(unittest.TestCase):
    def test_bias_from_noisy_still_samples(self):
        rnd = random.Random(1)
        bias = (0.8, -1.3, 0.4)
        f = ImuFilter()
        t, outputs = 0.0, []
        while f.calibrating:
            g = tuple(b + rnd.gauss(0, 0.2) for b in bias)
            outputs.append(f.update(t, (0.0, 0.0, 1.0), g, 25.0))
            t += DT
        self.assertEqual(len(outputs), int(config.IMU_CALIB_S * config.IMU_RATE_HZ))
        self.assertTrue(all(o is None for o in outputs[:-1]))
        self.assertIsNotNone(outputs[-1])
        for got, want in zip(f.bias, bias):
            self.assertAlmostEqual(got, want, delta=0.05)

    def test_moving_robot_restarts_calibration(self):
        f = ImuFilter(calib_s=0.5)
        for i in range(f.calib_n):
            f.update(i * DT, (0.0, 0.0, 1.0), (0.0, 0.0, 30.0 * math.sin(i / 5)), 25.0)
        self.assertTrue(f.calibrating)
        self.assertEqual(f.calib_failures, 1)
        for i in range(f.calib_n):
            f.update(1 + i * DT, (0.0, 0.0, 1.0), (0.0, 0.0, 2.0), 25.0)
        self.assertFalse(f.calibrating)
        self.assertAlmostEqual(f.bias[2], 2.0)

    def test_shaking_robot_restarts_calibration(self):
        f = ImuFilter(calib_s=0.5)
        for i in range(f.calib_n):
            f.update(i * DT, (0.2 * (i % 2), 0.0, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertTrue(f.calibrating)

    def test_recalibrate(self):
        f, t = calibrated_filter(bias=(0.0, 0.0, 1.0))
        f.recalibrate()
        self.assertIsNone(f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 3.0), 25.0))
        self.assertTrue(f.calibrating)


class YawTest(unittest.TestCase):
    def test_constant_left_turn(self):
        bias = (0.3, -0.2, 1.5)
        f, t = calibrated_filter(bias=bias)
        self.assertEqual(f.yaw, 0.0)
        for _ in range(100):          # 1 s at 90 deg/s
            t += DT
            r = f.update(t, (0.0, 0.0, 1.0), (bias[0], bias[1], bias[2] + 90.0), 25.0)
        # trapezoid: the first 10 ms average the old 0 and the new 90 deg/s
        self.assertAlmostEqual(r.yaw_deg, 90.0 * DT * (100 - 0.5), places=6)
        self.assertAlmostEqual(r.yaw_rate_dps, 90.0)

    def test_right_turn_is_negative(self):
        f, t = calibrated_filter()
        for _ in range(50):
            t += DT
            r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, -60.0), 25.0)
        self.assertAlmostEqual(r.yaw_deg, -60.0 * DT * (50 - 0.5), places=6)

    def test_trapezoid_is_exact_for_a_ramp(self):
        f, t = calibrated_filter()
        t0 = t
        for i in range(1, 101):       # rate rises 0 -> 100 deg/s over 1 s
            r = f.update(t0 + i * DT, (0.0, 0.0, 1.0), (0.0, 0.0, float(i)), 25.0)
        self.assertAlmostEqual(r.yaw_deg, 50.0, places=6)

    def test_irregular_timing(self):
        f, t = calibrated_filter()
        t += DT
        f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 45.0), 25.0)
        start = f.yaw
        rnd = random.Random(2)
        elapsed = 0.0
        while elapsed < 2.0:
            dt = rnd.uniform(0.005, 0.03)
            t += dt
            elapsed += dt
            r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 45.0), 25.0)
        self.assertAlmostEqual(r.yaw_deg - start, 45.0 * elapsed, places=6)

    def test_yaw_is_not_wrapped(self):
        f, t = calibrated_filter()
        for _ in range(500):          # 5 s at 100 deg/s
            t += DT
            r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 100.0), 25.0)
        self.assertAlmostEqual(r.yaw_deg, 100.0 * DT * (500 - 0.5), places=6)

    def test_gap_is_not_integrated(self):
        f, t = calibrated_filter()
        for _ in range(10):
            t += DT
            f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 100.0), 25.0)
        before = f.yaw
        t += 0.5                      # half a second without data
        r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 100.0), 25.0)
        self.assertEqual(r.yaw_deg, before)
        self.assertEqual(r.gaps, 1)
        t += DT
        r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 100.0), 25.0)
        self.assertAlmostEqual(r.yaw_deg, before + 1.0)

    def test_repeated_timestamp_adds_nothing(self):
        f, t = calibrated_filter()
        t += DT
        f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 100.0), 25.0)
        yaw = f.yaw
        r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 100.0), 25.0)
        self.assertEqual(r.yaw_deg, yaw)
        self.assertEqual(r.gaps, 0)

    def test_reset_yaw(self):
        f, t = calibrated_filter()
        for _ in range(10):
            t += DT
            f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 100.0), 25.0)
        f.reset_yaw()
        t += DT
        r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertAlmostEqual(r.yaw_deg, 0.5)   # trapezoid half step from 100 to 0


def tilted(pitch_deg=0.0, roll_deg=0.0):
    """Gravity reading (g) for a robot standing still with that pitch then roll."""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    return (-math.sin(r), math.sin(p) * math.cos(r), math.cos(p) * math.cos(r))


class TiltTest(unittest.TestCase):
    def test_start_from_gravity(self):
        f, _ = calibrated_filter(accel=tilted(pitch_deg=30))
        self.assertAlmostEqual(f.pitch, 30.0, places=6)
        self.assertAlmostEqual(f.roll, 0.0, places=6)
        f, _ = calibrated_filter(accel=tilted(roll_deg=-20))
        self.assertAlmostEqual(f.roll, -20.0, places=6)
        self.assertAlmostEqual(f.pitch, 0.0, places=6)

    def test_gyro_and_gravity_agree_on_nose_up(self):
        """Gyro X + and the gravity-based pitch must mean the same direction;
        otherwise the complementary filter would fight itself."""
        f, t = calibrated_filter()
        for i in range(1, 101):       # nose up at 30 deg/s for 1 s
            r = f.update(t + i * DT, tilted(pitch_deg=0.3 * i), (30.0, 0.0, 0.0), 25.0)
            self.assertAlmostEqual(r.pitch_deg, 0.3 * i, delta=0.2)
        self.assertAlmostEqual(r.tilt_deg, 30.0, delta=0.2)

    def test_gyro_and_gravity_agree_on_right_side_down(self):
        f, t = calibrated_filter()
        for i in range(1, 101):
            r = f.update(t + i * DT, tilted(roll_deg=0.3 * i), (0.0, 30.0, 0.0), 25.0)
            self.assertAlmostEqual(r.roll_deg, 0.3 * i, delta=0.2)

    def test_gravity_pulls_out_gyro_error(self):
        f, t = calibrated_filter()
        for i in range(1, 501):       # flat but gyro X has a leftover 2 deg/s error
            r = f.update(t + i * DT, (0.0, 0.0, 1.0), (2.0, 0.0, 0.0), 25.0)
        self.assertLess(abs(r.pitch_deg), 1.5)     # pure gyro would be 10 deg

    def test_hard_acceleration_is_ignored(self):
        f, t = calibrated_filter()
        for i in range(1, 101):       # 1 s of a 1 g push while flat (|a| = 1.41)
            r = f.update(t + i * DT, (0.0, 1.0, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertAlmostEqual(r.pitch_deg, 0.0, places=6)
        self.assertFalse(r.tipped)

    def test_normal_push_leans_a_little_then_recovers(self):
        """Known limit: a moderate push (|a| inside the trust window) leaks into
        pitch. A 0.3 g start for 0.2 s - more than the TT motors manage - stays
        far below the tipped limit and is gone again 2 s later."""
        f, t = calibrated_filter()
        for i in range(1, 21):
            r = f.update(t + i * DT, (0.0, 0.3, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertLess(r.pitch_deg, 6.0)
        for i in range(21, 221):
            r = f.update(t + i * DT, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertLess(abs(r.pitch_deg), 0.5)

    def test_fast_spin_on_tilted_floor_is_not_tipping(self):
        """Floor run 2026-10-29: robot 10 deg nose-up, spinning 420 deg/s. The rotation is
        about the vertical, so in the robot frame it shows up partly on the Y gyro; adding
        that to roll made 'tipped' trip. The tilt must stay 10 deg."""
        up = tilted(pitch_deg=10)
        shaken = tuple(1.3 * v for v in up)         # vibration: |a| outside the trust window
        f, t = calibrated_filter(accel=up)
        for i in range(1, 101):
            r = f.update(t + i * DT, shaken, tuple(420.0 * v for v in up), 25.0)
        self.assertAlmostEqual(r.tilt_deg, 10.0, delta=0.5)
        self.assertAlmostEqual(r.pitch_deg, 10.0, delta=0.5)
        self.assertFalse(r.tipped)
        self.assertAlmostEqual(r.yaw_rate_dps, 420.0, delta=0.5)
        self.assertAlmostEqual(r.yaw_deg, 420.0 * DT * (100 - 0.5), delta=0.5)

    def test_yaw_rate_is_about_the_vertical(self):
        f, t = calibrated_filter(accel=tilted(pitch_deg=30))
        r = f.update(t + DT, tilted(pitch_deg=30), (0.0, 0.0, 100.0), 25.0)
        self.assertAlmostEqual(r.yaw_rate_dps, 100.0 * math.cos(math.radians(30)), places=2)

    def test_tipped(self):
        f, t = calibrated_filter()
        r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertFalse(r.tipped)
        f, t = calibrated_filter(accel=(-1.0, 0.0, 0.0))     # lying on its right side
        r = f.update(t, (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 25.0)
        self.assertAlmostEqual(r.roll_deg, 90.0, places=6)
        self.assertTrue(r.tipped)
        f, t = calibrated_filter(accel=tilted(pitch_deg=40))
        self.assertFalse(f.update(t, tilted(pitch_deg=40), (0.0, 0.0, 0.0), 25.0).tipped)
        f, t = calibrated_filter(accel=tilted(pitch_deg=50))
        self.assertTrue(f.update(t, tilted(pitch_deg=50), (0.0, 0.0, 0.0), 25.0).tipped)


class BumpTest(unittest.TestCase):
    def test_sudden_hit_is_a_bump(self):
        f, t = calibrated_filter()
        for _ in range(20):
            t += DT
            r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertIsNone(r.last_bump_t)
        t += DT
        r = f.update(t, (0.0, -1.0, 1.0), (0.0, 0.0, 0.0), 25.0)   # hit from the front
        self.assertEqual(r.last_bump_t, t)
        self.assertAlmostEqual(r.bump_g, 1.0)
        hit = t
        for _ in range(50):
            t += DT
            r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertEqual(r.last_bump_t, hit)     # remembered, not repeated

    def test_small_jolt_is_not_a_bump(self):
        f, t = calibrated_filter()
        t += DT
        r = f.update(t, (0.3, 0.2, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertIsNone(r.last_bump_t)

    def test_smooth_acceleration_is_not_a_bump(self):
        f, t = calibrated_filter()
        for i in range(200):          # 0 -> 0.8 g over 2 s
            t += DT
            r = f.update(t, (0.0, 0.004 * i, 1.0), (0.0, 0.0, 0.0), 25.0)
        self.assertIsNone(r.last_bump_t)

    def test_saturation_counted(self):
        f, t = calibrated_filter()
        t += DT
        r = f.update(t, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 25.0, saturated=True)
        self.assertEqual(r.saturated, 1)


# ---------- module loop ----------

class ImuModuleTest(unittest.TestCase):
    AXES = ("+x", "-y", "-z")

    def setUp(self):
        self.clock = FakeClock()
        self.state = State(clock=self.clock)
        self.fake = FakeMpu6050()
        self.fake.accel = (0.0, 0.0, -1.0)       # chip down, robot flat
        self.imu = Imu(self.state, dev=self.fake, axes=self.AXES, level_up=None,
                       clock=self.clock, sleep=lambda s: None,
                       filt=ImuFilter(calib_s=0.2))
        self.imu.start()
        self.rnd = random.Random(3)

    def tick(self, n=1, noise=True):
        r = None
        for _ in range(n):
            if noise:   # the real chip never returns two identical frames for long
                self.fake.temp_c = 25.0 + self.rnd.uniform(-0.05, 0.05)
            self.clock.sleep(DT)
            r = self.imu.step()
        return r

    def test_calibrates_then_publishes_robot_frame(self):
        self.assertEqual(self.state.get("imu_status"), "calibrating")
        self.tick(self.imu.filter.calib_n - 1)
        self.assertIsNone(self.state.get("imu"))
        r = self.tick()
        self.assertEqual(self.state.get("imu_status"), "ok")
        self.assertIs(self.state.get("imu", max_age=0.05), r)
        self.assertAlmostEqual(r.accel_g[2], 1.0, places=3)
        self.assertAlmostEqual(r.tilt_deg, 0.0, places=3)
        self.assertFalse(r.tipped)

    def test_left_turn_through_the_whole_chain(self):
        self.tick(self.imu.filter.calib_n)
        self.fake.gyro = (0.0, 0.0, -90.0)      # sensor Z points down: left = negative
        r = self.tick(100)
        self.assertAlmostEqual(r.yaw_deg, 90.0 * DT * (100 - 0.5), delta=0.01)

    def test_single_glitch_is_skipped(self):
        self.tick(self.imu.filter.calib_n + 5)
        self.fake.fail_reads = 1
        self.assertIsNone(self.tick())
        self.assertEqual(self.state.get("imu_status"), "ok")
        self.assertIsNotNone(self.tick())
        self.assertEqual(self.imu.reinits, 0)
        self.assertEqual(self.imu.total_errors, 1)

    def test_lost_chip_goes_stale_and_recovers(self):
        self.tick(self.imu.filter.calib_n + 5)
        self.fake.fail_reads = 10_000            # wire pulled off
        self.tick(config.IMU_MAX_ERRORS)
        self.assertTrue(self.state.get("imu_status").startswith("error: I2C"))
        self.tick(10)
        self.assertIsNone(self.state.get("imu", max_age=0.1))   # safety sees: no IMU
        self.fake.fail_reads = 0                 # wire back on
        r = self.tick()
        self.assertIsNotNone(r)
        self.assertEqual(self.state.get("imu_status"), "ok")

    def test_chip_reset_is_detected_and_repaired(self):
        self.tick(self.imu.filter.calib_n + 5)
        self.fake.write(imu.PWR_MGMT_1, 0x80)   # brown-out: chip resets into sleep
        self.tick(config.IMU_FROZEN_SAMPLES)
        self.assertIn("frozen", self.state.get("imu_status"))
        self.assertEqual(self.imu.reinits, 1)
        self.assertEqual(self.fake.regs[imu.PWR_MGMT_1], 0x01)
        self.assertIsNotNone(self.tick(2))
        self.assertEqual(self.state.get("imu_status"), "ok")

    def test_reinit_at_most_once_a_second(self):
        self.tick(self.imu.filter.calib_n + 5)
        attempts = []
        real_init = self.imu.chip.init
        self.imu.chip.init = lambda: (attempts.append(self.clock()), real_init())[1]
        self.fake.fail_reads = 10_000
        self.tick(300)                           # 3 s of failures
        self.assertEqual(self.imu.reinits, 0)    # every init failed too
        self.assertEqual(len(attempts), 3)
        self.assertTrue(all(b - a >= 1.0 - 1e-9 for a, b in zip(attempts, attempts[1:])))
        self.assertIsNone(self.state.get("imu", max_age=0.1))


class RunLoopTest(unittest.TestCase):
    def test_runs_at_rate_and_stops(self):
        state = State()
        rnd = random.Random(4)

        class Noisy(FakeMpu6050):
            def _sample(self):
                self.temp_c = 25.0 + rnd.uniform(-0.05, 0.05)
                return super()._sample()
        m = Imu(state, dev=Noisy(), axes=("+x", "+y", "+z"), level_up=None, sleep=lambda s: None,
                filt=ImuFilter(calib_s=0.05))
        steps = []
        real_step = m.step
        m.step = lambda: (steps.append(1), real_step())[1]
        stop = threading.Event()
        th = threading.Thread(target=m.run, args=(stop,))
        th.start()
        time.sleep(0.5)
        stop.set()
        th.join(timeout=1.0)
        self.assertFalse(th.is_alive())
        self.assertEqual(state.get("imu_status"), "ok")
        r = state.get("imu", max_age=0.1)
        self.assertIsNotNone(r)
        self.assertGreater(r.yaw_deg, -1.0)
        self.assertLess(abs(r.yaw_deg), 1.0)
        # 0.5 s at 100 Hz = about 50 samples (generous bounds: the Pi may be busy)
        self.assertGreaterEqual(len(steps), 35)
        self.assertLessEqual(len(steps), 55)


if __name__ == "__main__":
    unittest.main()


class RunLoopBehindTest(unittest.TestCase):
    def test_loop_survives_falling_behind(self):
        stop = threading.Event()

        class SlowClock:
            t, n = 0.0, 0
            def __call__(self):
                self.t += 0.05                            # each read 50 ms later: always behind
                self.n += 1
                if self.n > 60:
                    stop.set()
                return self.t
        fake = FakeMpu6050()
        m = Imu(State(), dev=fake, axes=("+x", "+y", "+z"), level_up=None, clock=SlowClock(), sleep=lambda s: None,
                filt=ImuFilter(calib_s=0.05))
        m.run(stop)                                       # must return
        self.assertTrue(stop.is_set())
