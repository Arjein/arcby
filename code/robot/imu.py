"""
IMU module: MPU6050 (GY-521) on the shared I2C bus.

Three layers, so the maths can be tested without the chip:
  Mpu6050    register access: init, raw read, unit conversion, factory self-test
  ImuFilter  pure maths in the robot frame: gyro bias, yaw, roll/pitch, tipped, bump
  Imu        the 100 Hz loop: read -> map axes -> filter -> publish "imu" in State

Published in State:
  "imu"         ImuReading (only while the chip delivers good data, so a dead
                IMU shows up as a stale reading for safety.py)
  "imu_status"  "calibrating" / "ok" / "error: ..." (for the dashboard / logs)

Yaw comes from the gyro alone (no magnetometer): good for straight lines and
exact turns, but it drifts slowly (a few degrees per minute).

Tilt is tracked as the "up" direction (a unit vector in the robot frame) that
the gyro rotates and gravity slowly pulls back. That stays right while the
robot spins fast on a tilted floor, where adding gyro rates to roll/pitch
angles would drift (it showed "tipped" during floor spins). Yaw rate is the
rotation about that up direction, so it is also right when the robot sits tilted.
"""
import math
import time
from dataclasses import dataclass

from . import config

# ---------- registers (MPU-6000/6050 register map rev 4.2) ----------
SELF_TEST_X = 0x0D      # 0x0D-0x10: factory self-test trim codes
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B     # 14 bytes: accel xyz, temp, gyro xyz (big-endian int16)
PWR_MGMT_1 = 0x6B
WHO_AM_I = 0x75

# Genuine MPU6050 = 0x68. Clones answer with other IDs; they usually work,
# but the factory self-test below is only defined for the genuine chip.
WHO_AM_I_GENUINE = 0x68
WHO_AM_I_CLONES = {0x70, 0x72, 0x98}

# Running configuration (written by init, checked by reading back).
RUN_CONFIG = {
    SMPLRT_DIV: 1000 // config.IMU_RATE_HZ - 1,   # 1 kHz / (1 + 9) = 100 Hz
    CONFIG: 0x03,           # low-pass filter ~44 Hz: cuts motor vibration
    GYRO_CONFIG: 0x10,      # +-1000 deg/s (floor spins reach 500 deg/s, measured 2026-10-29)
    ACCEL_CONFIG: 0x08,     # +-4 g (room for bumps)
}
GYRO_LSB_PER_DPS = 32.8     # at +-1000 deg/s
ACCEL_LSB_PER_G = 8192.0    # at +-4 g
RAW_LIMITS = (-32768, 32767)


class ImuError(Exception):
    pass


def decode(raw):
    """14 raw bytes -> 7 signed ints (ax, ay, az, temp, gx, gy, gz)."""
    if len(raw) != 14:
        raise ImuError(f"expected 14 bytes, got {len(raw)}")
    vals = []
    for i in range(0, 14, 2):
        v = (raw[i] << 8) | raw[i + 1]
        vals.append(v - 65536 if v & 0x8000 else v)
    return tuple(vals)


def to_units(vals):
    """Decoded ints -> (accel g xyz, gyro deg/s xyz, temp C), sensor frame."""
    ax, ay, az, t, gx, gy, gz = vals
    accel = (ax / ACCEL_LSB_PER_G, ay / ACCEL_LSB_PER_G, az / ACCEL_LSB_PER_G)
    gyro = (gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS)
    return accel, gyro, t / 340.0 + 36.53


def is_saturated(vals):
    """True if any accel/gyro value sits at the end of its range (reading is clipped)."""
    return any(v in RAW_LIMITS for i, v in enumerate(vals) if i != 3)


@dataclass(frozen=True)
class SelfTestAxis:
    axis: str           # "accel x" ... "gyro z"
    factory: float      # expected response from the trim code (LSB)
    measured: float     # response measured now (LSB)

    @property
    def change(self):
        """Relative deviation from the factory value; datasheet limit +-14 %."""
        return None if self.factory == 0 else (self.measured - self.factory) / self.factory

    @property
    def ok(self):
        return self.change is not None and abs(self.change) <= 0.14


def factory_trim(trim):
    """4 bytes from 0x0D-0x10 -> expected self-test responses in LSB:
    ([accel x, y, z] at +-8 g, [gyro x, y, z] at +-250 deg/s). 0 = no trim stored."""
    a_codes = [((trim[0] >> 3) & 0x1C) | ((trim[3] >> 4) & 0x03),
               ((trim[1] >> 3) & 0x1C) | ((trim[3] >> 2) & 0x03),
               ((trim[2] >> 3) & 0x1C) | (trim[3] & 0x03)]
    g_codes = [trim[i] & 0x1F for i in range(3)]
    accel = [0.0 if c == 0 else 4096 * 0.34 * (0.92 / 0.34) ** ((c - 1) / 30) for c in a_codes]
    gyro = [0.0 if c == 0 else 25 * 131 * 1.046 ** (c - 1) for c in g_codes]
    gyro[1] = -gyro[1]      # the datasheet formula for Y is negative
    return accel, gyro


class Mpu6050:
    def __init__(self, dev, sleep=time.sleep):
        self.dev = dev          # I2CDevice (or a fake with read/write)
        self._sleep = sleep

    def who_am_i(self):
        return self.dev.read(WHO_AM_I, 1)[0]

    def init(self):
        """Reset, wake up and configure the chip; raise ImuError if it does not stick."""
        who = self.who_am_i()
        if who != WHO_AM_I_GENUINE and who not in WHO_AM_I_CLONES:
            raise ImuError(f"WHO_AM_I = 0x{who:02x}: not an MPU6050")
        self.dev.write(PWR_MGMT_1, 0x80)     # reset all registers
        self._sleep(0.1)
        self.dev.write(PWR_MGMT_1, 0x01)     # wake up, clock from the X gyro (more stable)
        self._sleep(0.05)
        for reg, value in RUN_CONFIG.items():
            self.dev.write(reg, value)
        for reg, value in {**RUN_CONFIG, PWR_MGMT_1: 0x01}.items():
            got = self.dev.read(reg, 1)[0]
            if got != value:
                raise ImuError(f"register 0x{reg:02x} reads 0x{got:02x}, wrote 0x{value:02x}")
        self._sleep(0.05)                    # let the low-pass filter settle
        return who

    def read_raw(self):
        return self.dev.read(ACCEL_XOUT_H, 14)

    def _mean(self, n):
        sums = [0.0] * 7
        for _ in range(n):
            for i, v in enumerate(decode(self.read_raw())):
                sums[i] += v / n
            self._sleep(0.01)
        return sums

    def self_test(self, samples=50):
        """Factory self-test (keep the robot still). Returns 6 SelfTestAxis.
        Only meaningful for a genuine MPU6050 (WHO_AM_I 0x68)."""
        try:
            self.dev.write(GYRO_CONFIG, 0x00)       # self-test runs at +-250 deg/s
            self.dev.write(ACCEL_CONFIG, 0x10)      # and +-8 g
            self._sleep(0.1)
            off = self._mean(samples)
            self.dev.write(GYRO_CONFIG, 0xE0)       # self-test on, all gyro axes
            self.dev.write(ACCEL_CONFIG, 0xF0)      # self-test on, all accel axes, +-8 g
            self._sleep(0.1)
            on = self._mean(samples)
        finally:
            self.dev.write(GYRO_CONFIG, RUN_CONFIG[GYRO_CONFIG])
            self.dev.write(ACCEL_CONFIG, RUN_CONFIG[ACCEL_CONFIG])
            self._sleep(0.1)
        accel_ft, gyro_ft = factory_trim(self.dev.read(SELF_TEST_X, 4))
        result = []
        for i, name in enumerate("xyz"):
            result.append(SelfTestAxis(f"accel {name}", accel_ft[i], on[i] - off[i]))
        for i, name in enumerate("xyz"):
            result.append(SelfTestAxis(f"gyro {name}", gyro_ft[i], on[4 + i] - off[4 + i]))
        return result


# ---------- axis mapping ----------

def parse_axes(spec):
    """("+x", "-y", "-z") -> ((0, 1), (1, -1), (2, -1)): for each robot axis the
    sensor axis index and sign. Must be a proper rotation (no mirror image)."""
    if len(spec) != 3:
        raise ValueError("need 3 entries (robot X, Y, Z)")
    mapping = []
    for s in spec:
        if len(s) != 2 or s[0] not in "+-" or s[1] not in "xyz":
            raise ValueError(f"bad axis {s!r}: use e.g. '+x' or '-z'")
        mapping.append(("xyz".index(s[1]), 1 if s[0] == "+" else -1))
    if sorted(i for i, _ in mapping) != [0, 1, 2]:
        raise ValueError(f"{spec}: every sensor axis must be used exactly once")
    m = [[0] * 3 for _ in range(3)]
    for row, (i, sign) in enumerate(mapping):
        m[row][i] = sign
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
           - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if det != 1:
        raise ValueError(f"{spec} is a mirror image, not a rotation - one sign is wrong")
    return tuple(mapping)


def apply_axes(mapping, v):
    return tuple(sign * v[i] for i, sign in mapping)


def level_rotation(up):
    """3x3 rotation (rows) that turns the measured rest gravity `up` onto +Z, the smallest such turn
    (Rodrigues). None -> None (no trim)."""
    if up is None:
        return None
    n = math.sqrt(sum(c * c for c in up))
    ux, uy, uz = (c / n for c in up)
    # axis = up x Z, cos = up . Z
    ax, ay, az = uy, -ux, 0.0
    s, c = math.hypot(ax, ay), uz
    if s < 1e-9:
        return None if c > 0 else ((1, 0, 0), (0, -1, 0), (0, 0, -1))
    ax, ay = ax / s, ay / s
    t = 1 - c
    return ((t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay),
            (t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax),
            (t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c))


def rotate(m, v):
    return v if m is None else tuple(sum(m[r][k] * v[k] for k in range(3)) for r in range(3))


# ---------- filter (pure maths, robot frame) ----------

@dataclass(frozen=True)
class ImuReading:
    t: float                # time.monotonic() of the sample
    yaw_deg: float          # continuous (not wrapped), + = turned left since start / reset_yaw
    yaw_rate_dps: float     # rotation about vertical, + = turning left
    roll_deg: float         # + = right side down
    pitch_deg: float        # + = nose (caster side) up
    tilt_deg: float         # angle between robot Z and vertical
    tipped: bool
    accel_g: tuple          # robot frame, bias not removed (1 g up when standing still)
    gyro_dps: tuple         # robot frame, bias removed
    temp_c: float
    last_bump_t: float      # time of the last bump, or None
    bump_g: float           # current sudden horizontal accel (compare with IMU_BUMP_G)
    saturated: int          # clipped samples since start
    gaps: int               # sample gaps too long to integrate (yaw less trustworthy)


class ImuFilter:
    def __init__(self, rate_hz=config.IMU_RATE_HZ, calib_s=config.IMU_CALIB_S,
                 calib_max_gyro_std=config.IMU_CALIB_MAX_GYRO_STD,
                 calib_max_accel_std=config.IMU_CALIB_MAX_ACCEL_STD,
                 tilt_alpha=config.IMU_TILT_ALPHA, tipped_deg=config.IMU_TIPPED_DEG,
                 bump_g=config.IMU_BUMP_G, max_dt=config.IMU_MAX_DT_S, bump_tau_s=0.2):
        self.calib_n = max(2, int(calib_s * rate_hz))
        self.calib_max_gyro_std = calib_max_gyro_std
        self.calib_max_accel_std = calib_max_accel_std
        self.tilt_alpha = tilt_alpha
        self.tipped_deg = tipped_deg
        self.bump_g = bump_g
        self.max_dt = max_dt
        self.bump_tau_s = bump_tau_s
        self.saturated = 0
        self.gaps = 0
        self.calib_failures = 0
        self.bias = None
        self.recalibrate()

    @property
    def calibrating(self):
        return self.bias is None

    def recalibrate(self):
        """Measure the gyro bias again (the robot must stand still meanwhile)."""
        self.bias = None
        self._calib = []
        self._last = None       # (t, gyro) of the previous sample

    def reset_yaw(self, yaw_deg=0.0):
        self.yaw = yaw_deg

    def update(self, t, accel, gyro, temp_c, saturated=False):
        """One sample in robot frame. Returns ImuReading, or None while calibrating."""
        if saturated:
            self.saturated += 1
        if self.bias is None:
            self._calibrate(accel, gyro)
            if self.bias is None:
                return None
            self._start(t, accel)
            dt = None
        else:
            dt = t - self._last[0]
        g = tuple(v - b for v, b in zip(gyro, self.bias))
        rate = None

        if dt is not None and dt > 0:
            if dt <= self.max_dt:
                # a direction fixed in the world turns the other way in the robot frame
                w = tuple(math.radians(v) * dt for v in g)
                u = self.up
                u = _unit((u[0] + u[1] * w[2] - u[2] * w[1],
                           u[1] + u[2] * w[0] - u[0] * w[2],
                           u[2] + u[0] * w[1] - u[1] * w[0]))
                if 0.85 <= _norm(accel) <= 1.15:   # trust gravity only when not accelerating hard
                    a = self.tilt_alpha
                    ua = _unit(accel)
                    u = _unit(tuple(a * ui + (1 - a) * ai for ui, ai in zip(u, ua)))
                self.up = u
                rate = _dot(g, u)
                self.yaw += 0.5 * (self._last_rate + rate) * dt     # trapezoid
                k = dt / (self.bump_tau_s + dt)
            else:
                self.gaps += 1
                self.up = _unit(accel)
                k = 1.0
            dev = math.hypot(accel[0] - self._lp[0], accel[1] - self._lp[1])
            self._lp = (self._lp[0] + k * (accel[0] - self._lp[0]),
                        self._lp[1] + k * (accel[1] - self._lp[1]))
            self.bump_now = dev
            if dev > self.bump_g:
                self.last_bump_t = t
        if rate is None:
            rate = _dot(g, self.up)
        if dt is None or dt > 0:
            self._last = (t, g)
            self._last_rate = rate

        self.pitch, self.roll = _accel_tilt(self.up)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, self.up[2]))))
        return ImuReading(
            t=t, yaw_deg=self.yaw, yaw_rate_dps=rate, roll_deg=self.roll,
            pitch_deg=self.pitch, tilt_deg=tilt, tipped=tilt > self.tipped_deg,
            accel_g=tuple(accel), gyro_dps=g, temp_c=temp_c,
            last_bump_t=self.last_bump_t, bump_g=self.bump_now,
            saturated=self.saturated, gaps=self.gaps)

    def _calibrate(self, accel, gyro):
        self._calib.append((accel, gyro))
        if len(self._calib) < self.calib_n:
            return
        samples, self._calib = self._calib, []
        gyro_std = max(_std([s[1][i] for s in samples]) for i in range(3))
        accel_std = max(_std([s[0][i] for s in samples]) for i in range(3))
        if gyro_std > self.calib_max_gyro_std or accel_std > self.calib_max_accel_std:
            self.calib_failures += 1        # robot was moving: start over
            return
        n = len(samples)
        self.bias = tuple(sum(s[1][i] for s in samples) / n for i in range(3))

    def _start(self, t, accel):
        self.yaw = 0.0
        self.up = _unit(accel)
        self._last_rate = 0.0
        self.pitch, self.roll = _accel_tilt(self.up)
        self._lp = (accel[0], accel[1])
        self.bump_now = 0.0
        self.last_bump_t = None


def _norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _unit(v):
    n = _norm(v)
    return (0.0, 0.0, 1.0) if n == 0 else (v[0] / n, v[1] / n, v[2] / n)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _accel_tilt(a):
    """(pitch, roll) in degrees from the gravity vector, robot frame."""
    pitch = math.degrees(math.atan2(a[1], a[2]))
    roll = math.degrees(math.atan2(-a[0], math.hypot(a[1], a[2])))
    return pitch, roll


def _std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ---------- the module loop ----------

class Imu:
    """Reads the chip at IMU_RATE_HZ and publishes into State. Use run(stop_event)
    in its own thread, or call step() yourself (tests, hwtest)."""

    def __init__(self, state, dev=None, axes=config.IMU_AXES, rate_hz=config.IMU_RATE_HZ, level_up=config.IMU_LEVEL_UP,
                 clock=time.monotonic, sleep=time.sleep, filt=None):
        self.state = state
        self.dev = dev
        self.mapping = parse_axes(axes)
        self.level = level_rotation(level_up)
        self.period = 1.0 / rate_hz
        self.clock = clock
        self.sleep = sleep
        self.filter = filt or ImuFilter(rate_hz=rate_hz)
        self.chip = None
        self.errors = 0             # I2C errors in a row
        self.total_errors = 0
        self.reinits = 0
        self._prev_raw = None
        self._same = 0
        self._next_reinit = 0.0

    def start(self):
        if self.dev is None:
            from .i2c import I2CDevice
            self.dev = I2CDevice(config.I2C_BUS, config.IMU_ADDR)
        self.chip = Mpu6050(self.dev, sleep=self.sleep)
        self.chip.init()
        self._status("calibrating")

    def step(self):
        """Read one sample and publish it. Returns the ImuReading or None."""
        now = self.clock()
        try:
            raw = self.chip.read_raw()
        except OSError as e:
            return self._fail(now, f"I2C read failed ({e})")
        self.errors = 0

        if raw == self._prev_raw:
            self._same += 1
            if self._same >= config.IMU_FROZEN_SAMPLES:
                return self._fail(now, "data frozen (chip reset or asleep?)", count=False)
        else:
            self._same = 0
        self._prev_raw = raw

        vals = decode(raw)
        accel, gyro, temp = to_units(vals)
        reading = self.filter.update(now, rotate(self.level, apply_axes(self.mapping, accel)),
                                     rotate(self.level, apply_axes(self.mapping, gyro)), temp,
                                     saturated=is_saturated(vals))
        if reading is None:
            self._status("calibrating")
            return None
        self.state.publish("imu", reading, t=now)
        self._status("ok")
        return reading

    def _fail(self, now, why, count=True):
        if count:
            self.errors += 1
            self.total_errors += 1
            if self.errors < config.IMU_MAX_ERRORS:
                return None             # a single glitch: just skip the sample
        self._status(f"error: {why}")
        if now >= self._next_reinit:    # try to bring the chip back, at most once a second
            self._next_reinit = now + 1.0
            try:
                self.chip.init()
                self.reinits += 1
                self._same = 0
                self._prev_raw = None
            except (OSError, ImuError):
                pass
        return None

    def _status(self, text):
        if self.state.get("imu_status") != text:
            self.state.publish("imu_status", text)

    def run(self, stop):
        """Loop until the threading.Event `stop` is set."""
        self.start()
        self.run_started(stop)

    def run_started(self, stop):
        """The loop of run(), for when start() was already called (robot/main.py)."""
        next_t = self.clock()
        while not stop.is_set():
            self.step()
            next_t += self.period
            delay = next_t - self.clock()
            if delay > 0:
                stop.wait(delay)
            else:
                next_t = self.clock()   # fell behind: do not try to catch up
