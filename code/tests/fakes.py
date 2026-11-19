"""Fake hardware for unit tests (no Pi needed)."""
import struct

from robot import imu


class FakeMpu6050:
    """Register-level MPU6050 behind the same read/write interface as I2CDevice.

    Physical values are set in g / deg/s (sensor frame) and converted with
    whatever range the code configured, like the real chip. Self-test adds
    the LSB responses given in st_response while the ST bits are set.
    """

    def __init__(self, who_am_i=0x68):
        self.regs = bytearray(128)
        self.who = who_am_i
        self._reset()
        self.accel = (0.0, 0.0, 1.0)
        self.gyro = (0.0, 0.0, 0.0)
        self.temp_c = 25.0
        self.st_response = [0.0] * 6    # accel xyz, gyro xyz, LSB
        self.fail_reads = 0             # next n reads raise OSError
        self.ignore_writes = set()      # registers that do not stick
        self.writes = []
        self._frozen = None

    def _reset(self):
        self.regs[:] = bytes(128)
        self.regs[imu.PWR_MGMT_1] = 0x40     # power-on: sleep
        self.regs[imu.WHO_AM_I] = self.who

    def set_trim(self, trim4):
        self.regs[imu.SELF_TEST_X:imu.SELF_TEST_X + 4] = bytes(trim4)

    def write(self, reg, value):
        self.writes.append((reg, value))
        if reg in self.ignore_writes:
            return
        if reg == imu.PWR_MGMT_1 and value & 0x80:
            trim = bytes(self.regs[imu.SELF_TEST_X:imu.SELF_TEST_X + 4])
            self._reset()
            self.regs[imu.SELF_TEST_X:imu.SELF_TEST_X + 4] = trim
            return
        self.regs[reg] = value

    def read(self, reg, n):
        if self.fail_reads:
            self.fail_reads -= 1
            raise OSError(5, "Input/output error")
        if reg == imu.ACCEL_XOUT_H and n == 14:
            return self._sample()
        return bytes(self.regs[reg:reg + n])

    def _sample(self):
        asleep = self.regs[imu.PWR_MGMT_1] & 0x40
        if asleep:              # a sleeping chip keeps returning its last sample
            return self._frozen or bytes(14)
        a_lsb = 16384.0 / (1 << ((self.regs[imu.ACCEL_CONFIG] >> 3) & 3))
        g_lsb = (131.0, 65.5, 32.8, 16.4)[(self.regs[imu.GYRO_CONFIG] >> 3) & 3]   # datasheet
        a_st = self.regs[imu.ACCEL_CONFIG] & 0xE0
        g_st = self.regs[imu.GYRO_CONFIG] & 0xE0
        vals = []
        for i in range(3):
            vals.append(self.accel[i] * a_lsb + (self.st_response[i] if a_st else 0))
        vals.append((self.temp_c - 36.53) * 340.0)
        for i in range(3):
            vals.append(self.gyro[i] * g_lsb + (self.st_response[3 + i] if g_st else 0))
        clipped = [max(-32768, min(32767, round(v))) for v in vals]
        self._frozen = struct.pack(">7h", *clipped)
        return self._frozen


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeDriver:
    """Motor driver stand-in: remembers what the motors were told."""

    def __init__(self):
        self.out = (0.0, 0.0)
        self.history = []
        self.stops = 0
        self.closed = False

    def set(self, left, right):
        self.out = (left, right)
        self.history.append(self.out)

    def stop(self):
        self.out = (0.0, 0.0)
        self.stops += 1

    def close(self):
        self.closed = True


def imu_reading(t, yaw=0.0, tipped=False, last_bump_t=None, tilt=0.0, pitch_rate=0.0, bump_g=0.0,
                yaw_rate=0.0):
    """A minimal ImuReading for safety / motors tests (pitch_rate = robot-frame gyro X)."""
    from robot.imu import ImuReading
    return ImuReading(t=t, yaw_deg=yaw, yaw_rate_dps=yaw_rate, roll_deg=0.0, pitch_deg=0.0,
                      tilt_deg=tilt, tipped=tipped, accel_g=(0.0, 0.0, 1.0),
                      gyro_dps=(pitch_rate, 0.0, 0.0), temp_c=25.0, last_bump_t=last_bump_t,
                      bump_g=bump_g, saturated=0, gaps=0)


class FakePca9685:
    """Register-level PCA9685 (pan-tilt PTZ board) behind the I2CDevice interface."""

    def __init__(self):
        self.regs = bytearray(256)
        self.regs[0x00] = 0x11      # power-on: sleep + all-call
        self.regs[0x01] = 0x04
        self.regs[0xFE] = 0x1E      # power-on prescale (200 Hz)
        self.fail = False           # every access raises OSError (board not answering)
        self.writes = []
        self.closed = False

    def _check(self):
        if self.fail:
            raise OSError(121, "Remote I/O error")

    def write(self, reg, value):
        self._check()
        self.writes.append((reg, value))
        if reg == 0xFE and not self.regs[0x00] & 0x10:
            return                  # PRE_SCALE only sticks while asleep (datasheet)
        self.regs[reg] = value & (0x7F if reg == 0x00 else 0xFF)   # RESTART bit reads back 0

    def write_block(self, reg, values):
        self._check()
        if not self.regs[0x00] & 0x20:
            raise AssertionError("block write without auto-increment")
        for i, v in enumerate(values):
            self.write(reg + i, v)

    def read(self, reg, n):
        self._check()
        return bytes(self.regs[reg:reg + n])

    def close(self):
        self.closed = True

    def counts(self, ch):
        """OFF count of a channel (None = full off)."""
        base = 0x06 + 4 * ch
        if self.regs[base + 3] & 0x10:
            return None
        return self.regs[base + 2] | (self.regs[base + 3] & 0x0F) << 8
