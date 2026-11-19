"""
Pan-tilt camera head: Arducam B0283 with a PCA9685 PWM chip on its PTZ board (I2C).

The PCA9685 makes the servo pulses by itself, so a position holds after the program
exits (the servos keep their torque until release() or power-off).

    head = PanTilt(I2CDevice(config.I2C_BUS, config.PTZ_ADDR))
    head.init()
    head.move(30, 10)       # look 30 deg left, 10 deg up, smoothly
    head.centre()

Head angles are degrees from straight ahead: pan + = left, tilt + = up (config.PTZ_*).
"""
import math
import time

from . import config

MODE1 = 0x00
MODE2 = 0x01
LED0_ON_L = 0x06            # channel n: 4 registers from 0x06 + 4 n (ON_L, ON_H, OFF_L, OFF_H)
PRE_SCALE = 0xFE

MODE1_AI = 0x20             # register auto-increment (needed for one-transaction channel writes)
MODE1_SLEEP = 0x10          # oscillator off; PRE_SCALE can only be written while asleep
MODE1_RESTART = 0x80
MODE2_OUTDRV = 0x04         # totem-pole outputs (power-on default)
FULL_OFF = 0x10             # bit 4 of OFF_H: output always low (servo limp)


class PanTiltError(Exception):
    pass


def prescale_for(hz, osc_hz=config.PTZ_OSC_HZ):
    """PRE_SCALE value for a PWM frequency (datasheet 7.3.5)."""
    value = round(osc_hz / (4096 * hz)) - 1
    if not 3 <= value <= 255:
        raise ValueError(f"{hz} Hz is out of the PCA9685 range")
    return value


def angle_to_us(angle, pulse_us=config.PTZ_PULSE_US):
    lo, hi = pulse_us
    return lo + (hi - lo) * angle / 180.0


def us_to_counts(us, prescale, osc_hz=config.PTZ_OSC_HZ):
    """Pulse width in microseconds -> PCA9685 OFF count (0-4095) at the configured PWM period."""
    tick_us = (prescale + 1) * 1e6 / osc_hz
    return max(0, min(4095, round(us / tick_us)))


def clamp(v, limits):
    return max(limits[0], min(limits[1], v))


class PanTilt:
    def __init__(self, dev, sleep=time.sleep, pan_ch=config.PTZ_PAN_CH, tilt_ch=config.PTZ_TILT_CH,
                 centre=(config.PTZ_PAN_CENTRE, config.PTZ_TILT_CENTRE),
                 sign=(config.PTZ_PAN_SIGN, config.PTZ_TILT_SIGN),
                 limits=(config.PTZ_PAN_SERVO, config.PTZ_TILT_SERVO)):
        self.dev = dev              # I2CDevice (or a fake with read/write/write_block)
        self._sleep = sleep
        self.channels = (pan_ch, tilt_ch)
        self.centre_servo = centre
        self.sign = sign
        self.limits = limits
        self.prescale = prescale_for(config.PTZ_PWM_HZ)
        self.pan = self.tilt = None     # last head angles sent (None = unknown until the first move)

    def init(self):
        """Set the PWM frequency and wake the chip; raise PanTiltError if it does not stick.
        Does not move the servos (their outputs keep whatever they had)."""
        try:
            self.dev.write(MODE1, MODE1_SLEEP | MODE1_AI)
            self.dev.write(PRE_SCALE, self.prescale)
            self.dev.write(MODE2, MODE2_OUTDRV)
            self.dev.write(MODE1, MODE1_AI)              # wake up
            self._sleep(0.001)                           # oscillator needs 500 us
            self.dev.write(MODE1, MODE1_AI | MODE1_RESTART)
            got = self.dev.read(PRE_SCALE, 1)[0]
            mode1 = self.dev.read(MODE1, 1)[0]
        except OSError as e:
            raise PanTiltError(f"PTZ board does not answer: {e}") from e
        if got != self.prescale:
            raise PanTiltError(f"PRE_SCALE reads {got}, wrote {self.prescale}: not a PCA9685?")
        if mode1 & MODE1_SLEEP:
            raise PanTiltError(f"MODE1 = 0x{mode1:02x}: chip did not wake up")
        return mode1

    def answers(self):
        """Read-only: does the board answer (is the servo buck - the battery - on)? Never writes."""
        try:
            self.dev.read(MODE1, 1)
            return True
        except OSError:
            return False

    def servo_angles(self, pan, tilt):
        """Head angles -> clamped servo angles (pan servo, tilt servo). Not a number (NaN, inf) = straight ahead
        on that axis (clamp(nan) gave the servo's end stop)."""
        return tuple(clamp(c + s * (a if math.isfinite(a) else 0.0), lim) for c, s, a, lim
                     in zip(self.centre_servo, self.sign, (float(pan), float(tilt)), self.limits))

    def head_angles(self, servo):
        """Inverse of servo_angles (after clamping): the head angles really reached."""
        return tuple((v - c) * s for v, c, s in zip(servo, self.centre_servo, self.sign))

    def _write_channel(self, ch, counts):
        self.dev.write_block(LED0_ON_L + 4 * ch, (0, 0, counts & 0xFF, counts >> 8))

    def set(self, pan, tilt):
        """Jump straight to the head angles (clamped). Returns the angles reached."""
        servo = self.servo_angles(pan, tilt)
        for ch, angle in zip(self.channels, servo):
            self._write_channel(ch, us_to_counts(angle_to_us(angle), self.prescale))
        self.pan, self.tilt = self.head_angles(servo)
        return self.pan, self.tilt

    def move(self, pan, tilt, speed_dps=config.PTZ_SPEED_DPS, step_s=0.02):
        """Go to the head angles smoothly at speed_dps (both axes arrive together)."""
        if self.pan is None:
            return self.set(pan, tilt)
        target = self.head_angles(self.servo_angles(pan, tilt))
        start = (self.pan, self.tilt)
        dist = max(abs(t - s) for t, s in zip(target, start))
        steps = max(1, int(dist / (speed_dps * step_s)))
        for i in range(1, steps + 1):
            f = i / steps
            self.set(*(s + (t - s) * f for s, t in zip(start, target)))
            if i < steps:
                self._sleep(step_s)
        return self.pan, self.tilt

    def centre(self, speed_dps=config.PTZ_SPEED_DPS):
        return self.move(0.0, 0.0, speed_dps)

    def release(self):
        """Outputs off: the servos go limp (the head may sag under the camera's weight)."""
        for ch in self.channels:
            self.dev.write_block(LED0_ON_L + 4 * ch, (0, 0, 0, FULL_OFF))
        self.pan = self.tilt = None

    def close(self):
        """Release the servos and close the I2C device."""
        try:
            self.release()
        finally:
            self.dev.close()
