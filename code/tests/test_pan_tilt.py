"""robot/pan_tilt.py against a fake PCA9685."""
import unittest

from robot import config, pan_tilt
from robot.pan_tilt import PanTilt, PanTiltError
from tests.fakes import FakePca9685


def head(chip=None, **kw):
    chip = chip or FakePca9685()
    sleeps = []
    h = PanTilt(chip, sleep=sleeps.append, **kw)
    return h, chip, sleeps


def us(chip, ch):
    """Pulse width the fake chip outputs on a channel, in microseconds."""
    tick = (chip.regs[pan_tilt.PRE_SCALE] + 1) * 1e6 / config.PTZ_OSC_HZ
    return chip.counts(ch) * tick


class MathTest(unittest.TestCase):
    def test_prescale_for_50hz(self):
        self.assertEqual(pan_tilt.prescale_for(50), 121)     # datasheet example: 25 MHz / (4096 * 50) - 1

    def test_prescale_out_of_range(self):
        with self.assertRaises(ValueError):
            pan_tilt.prescale_for(5)

    def test_angle_to_pulse(self):
        self.assertEqual(pan_tilt.angle_to_us(0), 500)
        self.assertEqual(pan_tilt.angle_to_us(90), 1500)
        self.assertEqual(pan_tilt.angle_to_us(180), 2500)

    def test_counts_for_1500us(self):
        self.assertEqual(pan_tilt.us_to_counts(1500, 121), 307)   # 1.5 ms / (20 ms / 4096)


class InitTest(unittest.TestCase):
    def test_sets_frequency_and_wakes(self):
        h, chip, _ = head()
        h.init()
        self.assertEqual(chip.regs[pan_tilt.PRE_SCALE], 121)
        self.assertFalse(chip.regs[pan_tilt.MODE1] & pan_tilt.MODE1_SLEEP)
        self.assertTrue(chip.regs[pan_tilt.MODE1] & pan_tilt.MODE1_AI)

    def test_prescale_written_while_asleep(self):
        h, chip, _ = head()
        h.init()
        i = chip.writes.index((pan_tilt.PRE_SCALE, 121))
        self.assertTrue(chip.writes[i - 1][1] & pan_tilt.MODE1_SLEEP)

    def test_init_does_not_move_servos(self):
        h, chip, _ = head()
        h.init()
        self.assertEqual(chip.counts(config.PTZ_PAN_CH), 0)
        self.assertIsNone(h.pan)

    def test_no_answer_raises(self):
        chip = FakePca9685()
        chip.fail = True
        h, _, _ = head(chip)
        with self.assertRaisesRegex(PanTiltError, "does not answer"):
            h.init()

    def test_wrong_chip_raises(self):
        chip = FakePca9685()
        chip.write = lambda reg, value: None     # ignores every write (e.g. another chip at 0x40)
        h, _, _ = head(chip)
        with self.assertRaisesRegex(PanTiltError, "PRE_SCALE"):
            h.init()


class MoveTest(unittest.TestCase):
    """Angle math with a plain head (centre 90/90, signs +1), independent of the tuned config."""

    def setUp(self):
        self.h, self.chip, self.sleeps = head(centre=(90.0, 90.0), sign=(1, 1))
        self.h.init()

    def test_centre_is_1500us_on_both_channels(self):
        self.h.set(0, 0)
        for ch in (config.PTZ_PAN_CH, config.PTZ_TILT_CH):
            self.assertAlmostEqual(us(self.chip, ch), 1500, delta=5)

    def test_pan_left_is_larger_servo_angle(self):
        self.h.set(45, 0)
        self.assertAlmostEqual(us(self.chip, config.PTZ_PAN_CH), angle_us(135), delta=5)
        self.assertAlmostEqual(us(self.chip, config.PTZ_TILT_CH), 1500, delta=5)

    def test_sign_flips_direction(self):
        h, chip, _ = head(centre=(90.0, 90.0), sign=(-1, 1))
        h.init()
        h.set(45, 0)
        self.assertAlmostEqual(us(chip, config.PTZ_PAN_CH), angle_us(45), delta=5)

    def test_clamped_to_servo_limits(self):
        reached = self.h.set(170, -170)
        lo_t = config.PTZ_TILT_SERVO[0]
        self.assertEqual(reached, (config.PTZ_PAN_SERVO[1] - 90, lo_t - 90))
        self.assertAlmostEqual(us(self.chip, config.PTZ_TILT_CH), angle_us(lo_t), delta=5)

    def test_not_a_number_is_straight_ahead(self):                            # review 2026-11-01
        self.assertEqual(self.h.set(float("nan"), float("inf")), (0.0, 0.0))

    def test_move_is_smooth(self):
        self.h.set(0, 0)
        self.chip.writes.clear()
        self.sleeps.clear()
        self.h.move(45, 0, speed_dps=90, step_s=0.02)
        pan_offs = [v for r, v in self.chip.writes if r == 0x06 + 4 * config.PTZ_PAN_CH + 2]
        self.assertEqual(len(pan_offs), 25)                   # 45 deg at 1.8 deg per step
        self.assertAlmostEqual(sum(self.sleeps), 0.48, places=6)
        self.assertEqual((self.h.pan, self.h.tilt), (45, 0))

    def test_first_move_jumps(self):
        self.h.move(10, 5)
        self.assertEqual(self.sleeps, [0.001])                # only init's wake-up wait
        self.assertEqual((self.h.pan, self.h.tilt), (10, 5))

    def test_release_turns_outputs_off(self):
        self.h.set(0, 0)
        self.h.release()
        self.assertIsNone(self.chip.counts(config.PTZ_PAN_CH))
        self.assertIsNone(self.chip.counts(config.PTZ_TILT_CH))
        self.assertIsNone(self.h.pan)


def angle_us(angle):
    return pan_tilt.angle_to_us(angle)


if __name__ == "__main__":
    unittest.main()


class AnswersTest(unittest.TestCase):
    def test_answers_only_reads(self):
        from robot.pan_tilt import PanTilt

        class Dev:
            def __init__(self, ok):
                self.ok, self.writes = ok, 0

            def read(self, reg, n):
                if not self.ok:
                    raise OSError(121, "Remote I/O error")
                return bytes(n)

            def write(self, *a):
                self.writes += 1

            write_block = write
        for ok in (True, False):
            d = Dev(ok)
            self.assertEqual(PanTilt(d).answers(), ok)
            self.assertEqual(d.writes, 0)
