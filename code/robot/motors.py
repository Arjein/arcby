"""
Motors module: the ONLY code that drives the L298N (docs/architecture.md).

Two ways to ask for motion (both must be repeated faster than
DRIVE_CMD_TIMEOUT_S, or the motors stop):

  request(left, right)          signed wheel speeds -1..1, + = forward
  request_velocity(v, turn_dps) speed -1..1 + turn rate in deg/s (+ = left) -
                                what a driver or a robot "brain" wants. The gyro
                                closes the loop on the turn rate, so a curve is
                                the same on carpet or tiles, full or low battery.

Every DRIVE_RATE_HZ the loop turns the latest request into PWM:

  watchdog      no fresh request for DRIVE_CMD_TIMEOUT_S -> target 0
  safety        "safety" in State says stop (or is missing / too old) -> 0 at once
  dead band     any non-zero speed gets at least DRIVE_MIN (the wheels start there)
  soft start    PWM rises at most DRIVE_ACCEL per second, falls at DRIVE_DECEL
  heading hold  driving straight keeps the heading with the IMU yaw
  turn rate     velocity mode: wheel difference = feed-forward + PI on the
                measured yaw rate. Turning on the spot: both wheels opposite
                for fast turns (power from the measured spin curve - both
                wheels cannot turn slower than ~250 deg/s on carpet), one wheel
                at a time for slower ones (forward/backward mixed so it does
                not walk away); TURN_ON_SPOT = "auto" / "spin" / "pivot"

Publishes "motors" in State: target / applied PWM, whether it is moving, when
it reached the target speed (safety uses that to ignore start jerks), and why
it is stopped.
"""
import math
import subprocess
import threading
import time
from dataclasses import dataclass

from . import config


class GpioDriver:
    """Both wheels as signed PWM -1..1 through gpiozero (same as robot_drive.py)."""

    def __init__(self, pin_factory=None, battery_check=None):
        """The real pins need the battery: without it (Pi on a wall adapter) power.BatteryOff is raised before any
        pin is touched. A pin_factory (tests, mock pins) skips the check unless battery_check is given."""
        if pin_factory is None or battery_check is not None:
            from .power import battery_on, require_battery
            require_battery("the motors", battery_check or battery_on)
        from gpiozero import Motor
        kw = {"pin_factory": pin_factory} if pin_factory else {}
        self.motors = (Motor(**config.MOTOR_LEFT, **kw), Motor(**config.MOTOR_RIGHT, **kw))

    def set(self, left, right):
        for m, v in zip(self.motors, (left, right)):
            if v > 0:
                m.forward(min(1.0, v))
            elif v < 0:
                m.backward(min(1.0, -v))
            else:
                m.stop()

    def stop(self):
        for m in self.motors:
            m.stop()

    def close(self):
        self.stop()
        for m in self.motors:
            m.close()
        # gpiozero leaves the pins floating; hold the L298N inputs LOW (as at boot)
        subprocess.run(["pinctrl", "set", ",".join(map(str, config.MOTOR_PINS)), "ip", "pd"],
                       capture_output=True)


def to_pwm(v):
    """Request -1..1 -> PWM with the dead band: 0 stays 0, anything else starts at DRIVE_MIN."""
    if abs(v) < 0.01:
        return 0.0
    mag = config.DRIVE_MIN + (config.DRIVE_MAX - config.DRIVE_MIN) * min(1.0, abs(v))
    return mag if v > 0 else -mag


def _ramp(cur, target, dt):
    if cur == target:
        return target
    speeding_up = cur * target >= 0 and abs(target) > abs(cur)
    lim = (config.DRIVE_ACCEL if speeding_up else config.DRIVE_DECEL) * dt
    return cur + max(-lim, min(lim, target - cur))


def _clamp(v, lim):
    """Within -lim..lim; not a number (NaN, inf) = 0 = stop (max/min with NaN gave full forward)."""
    return max(-lim, min(lim, v)) if math.isfinite(v) else 0.0


@dataclass(frozen=True)
class Request:
    left: float
    right: float
    t: float
    source: str
    v: float = None             # velocity mode when v is not None
    turn: float = 0.0           # deg/s, + = left


class Motors:
    def __init__(self, state, driver, clock=time.monotonic):
        self.state = state
        self.driver = driver
        self.clock = clock
        self.period = 1.0 / config.DRIVE_RATE_HZ
        self._lock = threading.Lock()
        self._req = None
        self._last_t = None
        self.base = (0.0, 0.0)          # ramped PWM without the heading / turn correction
        self.applied = (0.0, 0.0)       # what the driver got
        self._target = (0.0, 0.0, 0.0)
        self._settled_since = None
        self._hold_yaw = None
        self._turn_ref = 0.0            # velocity mode: smoothed turn-rate request (deg/s)
        self._turn_i = 0.0              # velocity mode: integral part (PWM difference)
        self._turn_goal = 0.0           # velocity mode: turn rate actually aimed at (incl. heading hold)
        self._creep = 0.0               # pivots: + = moved the robot forward, - = backward (PWM x s)
        self._pivot_fwd = True          # pivot by driving a wheel forward (else backward)
        self._spinning = False          # "auto": currently spinning on both wheels
        self._start_t = None            # velocity mode: when a forward drive started from standstill
        self._spin_dir = 0              # spinning last step: +1 left / -1 right, 0 = not spinning
        self._spin_lim = 0.0            # soft spin start: the highest PWM a spin may have yet (SPIN_RAMP_S)
        self._now = 0.0

    def request(self, left, right, source="?"):
        """Ask for wheel speeds -1..1 (+ = forward)."""
        left, right = _clamp(float(left), 1.0), _clamp(float(right), 1.0)
        with self._lock:
            self._req = Request(left, right, self.clock(), source)

    def request_velocity(self, v, turn_dps, source="?"):
        """Ask for a speed -1..1 (+ = forward) and a turn rate in deg/s (+ = left)."""
        v, turn = _clamp(float(v), 1.0), _clamp(float(turn_dps), config.TURN_MAX_DPS)
        with self._lock:
            self._req = Request(0.0, 0.0, self.clock(), source, v=v, turn=turn)

    def stop(self, source="?"):
        self.request(0.0, 0.0, source)

    def step(self):
        now = self.clock()
        dt = self.period if self._last_t is None else min(0.1, max(0.0, now - self._last_t))
        self._last_t = now
        with self._lock:
            req = self._req

        reason = None
        velocity = req is not None and req.v is not None
        if req is None:
            target, reason = (0.0, 0.0, 0.0), "no command yet"
        elif now - req.t > config.DRIVE_CMD_TIMEOUT_S:
            target, reason = (0.0, 0.0, 0.0), "command timeout"
        elif velocity:
            pv = to_pwm(req.v)
            target = (pv, pv, req.turn)
        else:
            target = (to_pwm(req.left), to_pwm(req.right), 0.0)

        safety = self.state.get("safety", max_age=config.SAFETY_MAX_AGE_S)
        if safety is None:
            halt = "safety not running"
        elif safety["stop"]:
            halt = "safety: " + ", ".join(safety["reasons"])
        else:
            halt = None

        if target != self._target:
            self._target = target
            self._settled_since = None
        if not (velocity and target[0] > 0.0) or halt:
            self._start_t = None
        elif self._start_t is None and self.base == (0.0, 0.0):
            self._start_t = now
        self._now = now
        if halt or not velocity:
            self._spin_dir = 0
        if halt:
            self.base = self.applied = (0.0, 0.0)
            self._hold_yaw = None
            self._settled_since = None
            self._turn_ref = self._turn_i = self._turn_goal = 0.0
            self.driver.stop()
            reason = halt
        else:
            self.base = (_ramp(self.base[0], target[0], dt), _ramp(self.base[1], target[1], dt))
            if velocity:
                out, smooth = self._velocity_out(target, dt)
            else:
                out, smooth = self._wheels_out(target), True
            if self.base == target[:2] and smooth and self._settled_since is None:
                self._settled_since = now
            self.applied = out
            self.driver.set(*out)

        moving = target != (0.0, 0.0, 0.0) and not halt
        self.state.publish("motors", {
            "target": target[:2], "turn_target": target[2], "turn_goal": self._turn_goal,
            "applied": self.applied, "moving": moving,
            "settled_since": self._settled_since if moving else None,
            "hold_yaw": self._hold_yaw, "reason": reason,
            "source": req.source if req else None,
        }, t=now)
        return self.applied

    def _wheels_out(self, target):
        """Wheel mode: ramped PWM, plus the heading hold when the request is straight."""
        out = self.base
        straight = target[0] == target[1] != 0.0
        imu = self.state.get("imu", max_age=config.SAFETY_IMU_STALE_S) if straight else None
        if imu is not None:
            if self._hold_yaw is None:
                self._hold_yaw = imu.yaw_deg
            err = self._hold_yaw - imu.yaw_deg          # + = drifted right -> steer left
            steer = _clamp(config.HEADING_KP * err, config.HEADING_MAX_STEER)
            cap = config.DRIVE_MAX
            out = (_clamp(out[0] - steer, cap), _clamp(out[1] + steer, cap))
        elif not straight:
            self._hold_yaw = None
        return out

    def _velocity_out(self, target, dt):
        """Velocity mode: base speed + a wheel difference that makes the gyro read the asked rate.
        Returns (out, settled) - settled when speed and turn rate have reached the request."""
        pv, _, turn = target
        step = config.TURN_ACCEL_DPS2 * dt
        self._turn_ref += _clamp(turn - self._turn_ref, step)
        imu = self.state.get("imu", max_age=config.SAFETY_IMU_STALE_S)
        base = self.base[0]

        goal = self._turn_ref
        if turn == 0.0 and pv != 0.0 and imu is not None and abs(self._turn_ref) < 1e-9:
            if self._hold_yaw is None:
                self._hold_yaw = imu.yaw_deg
            goal = _clamp(config.HEADING_RATE_KP * (self._hold_yaw - imu.yaw_deg), config.HEADING_MAX_RATE)
        elif turn != 0.0 or pv == 0.0:
            self._hold_yaw = None
        self._turn_goal = goal

        if goal == 0.0 and pv == 0.0:
            self._turn_i = 0.0
            return (base, base), abs(self._turn_ref) < 1e-9
        spin = pv == 0.0 and self._spin_now(turn)
        if spin and turn == 0.0:
            # turn released while spinning on the spot: stop now. A spin cannot slow down below
            # SPIN_MIN_DPS anyway, and the sign of a zero request (-0.0 from the keys!) must never
            # pick a direction - that spun the robot RIGHT after releasing A (Selim, 2026-10-29).
            self._turn_ref = self._turn_i = self._turn_goal = 0.0
            return (0.0, 0.0), True
        if spin and abs(goal) < config.SPIN_MIN_DPS:
            # a both-wheel spin cannot go slower than this - aim there at once (no wind-up on the ramp)
            goal = math.copysign(config.SPIN_MIN_DPS, turn)
            self._turn_goal = goal
        if pv != 0.0:
            self._creep = 0.0                             # driving: pivot drift no longer matters
        pi = 0.0
        if imu is not None:
            err = goal - imu.yaw_rate_dps
            if abs(err) < config.TURN_I_ERR_MAX:
                self._turn_i = _clamp(self._turn_i + config.TURN_KI * err * dt, config.TURN_I_MAX)
            pi = config.TURN_KP * err + self._turn_i
        diff = goal / config.TURN_K + pi
        cap = config.DRIVE_MAX
        was_dir, self._spin_dir = self._spin_dir, 0
        if spin:
            # both wheels opposite, about the centre: power from the measured spin curve + correction
            ff = config.SPIN_START + max(0.0, abs(goal) - config.SPIN_MIN_DPS) / config.SPIN_SLOPE
            mag = max(config.SPIN_START, min(cap, ff + (pi if goal > 0 else -pi) / 2))
            self._spin_dir = 1 if goal > 0 else -1
            if config.SPIN_RAMP_S > 0:
                # soft start from standstill / on reversing: rises at SPIN_START per SPIN_RAMP_S (the PI part alone
                # asked ~0.5 PWM at the first step) - less start current on the battery (power cut, 2026-11-08)
                if was_dir != self._spin_dir:
                    self._spin_lim = 0.0
                self._spin_lim = min(cap, self._spin_lim + config.SPIN_START / config.SPIN_RAMP_S * dt)
                mag = min(mag, self._spin_lim)
            out = (-mag, mag) if goal > 0 else (mag, -mag)
        elif pv == 0.0:
            out = self._pivot(diff, dt)                   # slow: one wheel at a time
        else:
            out = (base - diff / 2, base + diff / 2)
            if self._start_t is not None and self._now - self._start_t < config.DRIVE_START_S:
                out = (out[0], out[1] + config.DRIVE_START_RIGHT)     # the stiffer right wheel breaks free too
        out = (_clamp(out[0], cap), _clamp(out[1], cap))
        return out, abs(self._turn_ref - turn) < 1e-9

    def _spin_now(self, turn):
        """Turn on the spot with both wheels? 'auto' decides from the REQUESTED rate (not the ramp
        towards it), so a fast turn does not start with a pivot; the correction restarts on a switch."""
        mode = config.TURN_ON_SPOT
        was = self._spinning
        if mode != "auto":
            self._spinning = mode == "spin"
        elif self._spinning and abs(turn) < config.SPIN_AUTO_DOWN:
            self._spinning = False
        elif not self._spinning and abs(turn) > config.SPIN_AUTO_UP:
            self._spinning = True
        if self._spinning != was:
            self._turn_i = 0.0
        return self._spinning

    def _pivot(self, diff, dt):
        """Turn on the spot with one wheel (right - left = diff). Driving that wheel forward moves
        the centre forward, backward moves it back - switch between them so it stays put."""
        mag = min(abs(diff), config.DRIVE_MAX)
        if self._creep > config.PIVOT_CREEP_MAX:
            self._pivot_fwd = False
        elif self._creep < -config.PIVOT_CREEP_MAX:
            self._pivot_fwd = True
        self._creep += (mag if self._pivot_fwd else -mag) * dt
        left_turn = diff > 0
        if self._pivot_fwd:
            return (0.0, mag) if left_turn else (mag, 0.0)       # outer wheel forward
        return (-mag, 0.0) if left_turn else (0.0, -mag)         # inner wheel backward

    def run(self, stop):
        """Loop until the threading.Event `stop` is set; the motors are stopped on the way out."""
        try:
            nxt = self.clock()
            while not stop.is_set():
                self.step()
                nxt += self.period
                delay = nxt - self.clock()
                if delay > 0:
                    stop.wait(delay)
                else:
                    nxt = self.clock()
        finally:
            self.driver.stop()
            self.applied = self.base = (0.0, 0.0)

    def close(self):
        self.driver.close()
