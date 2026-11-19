"""
Safety module: has the last word over the motors (docs/architecture.md).

Every SAFETY_RATE_HZ it publishes "safety" = {"stop": bool, "reasons": [...],
"latched": [...]}; motors.py stops at once while "stop" is true, and also when
this verdict is missing or older than SAFETY_MAX_AGE_S (safety loop dead).

  imu missing   no IMU reading for SAFETY_IMU_STALE_S (still calibrating, wire off)
  tipped        tilt above IMU_TIPPED_DEG                 -> latched until reset()
  bump          sudden jolt while driving at steady speed -> latched until reset()
  low battery   Pi under-voltage for POWER_UV_HOLD_S (battery ~7.0 V, robot/power.py)
                                                          -> latched until reset() and voltage ok

A jolt is only a bump if the robot was driving calmly before it:
  - not right after a speed change (SAFETY_BUMP_BLANK_S after the motors
    reached their target) - starting and stopping jerk the robot as hard as a hit;
  - not while the nose was moving (pitch rate above BUMP_CALM_DPS during the
    BUMP_CALM_S before the jolt) - at take-off the nose lifts and lands with a
    thump ~0.5 s later (measured 2026-10-29). A real crash comes out of calm driving.

Every new stop writes a "black box" CSV with the last BLACKBOX_S seconds of IMU
and motor data to BLACKBOX_DIR, for tuning.
"""
import csv
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from . import config

log = logging.getLogger("arcby")
BOX_FIELDS = ["t", "yaw", "yaw_rate", "pitch", "tilt", "pitch_rate", "roll_rate", "ax", "ay", "az",
              "bump_g", "applied_l", "applied_r", "moving", "settled"]


class Safety:
    def __init__(self, state, clock=time.monotonic, blackbox_dir=config.BLACKBOX_DIR):
        self.state = state
        self.clock = clock
        self.period = 1.0 / config.SAFETY_RATE_HZ
        self._lock = threading.Lock()
        self.latched = set()
        self._bump_seen = None
        self._last_reasons = None
        self._last_imu_t = None
        self.history = deque(maxlen=int(config.BLACKBOX_S * config.SAFETY_RATE_HZ * 2))
        self.blackbox_dir = Path(blackbox_dir).expanduser() if blackbox_dir else None
        self.ignored_bumps = 0
        self._uv_since = None           # first time the current under-voltage was seen

    def reset(self):
        """Clear latched stops (operator key). 'tipped' stays while the robot is still tipped."""
        imu = self.state.get("imu", max_age=config.SAFETY_IMU_STALE_S)
        with self._lock:
            self.latched.discard("bump")
            if imu is None or not imu.tipped:
                self.latched.discard("tipped")
            if not self._under_voltage():
                self.latched.discard("low battery")

    def _under_voltage(self):
        p = self.state.get("power", max_age=5 * config.POWER_POLL_S)
        return bool(p and p.get("ok") and p.get("under_voltage"))

    def _check_power(self):
        if not self._under_voltage():
            self._uv_since = None
            return False
        now = self.clock()
        if self._uv_since is None:
            self._uv_since = now
            log.warning("Pi under-voltage - battery low?")
        return now - self._uv_since >= config.POWER_UV_HOLD_S

    def step(self):
        reasons = []
        imu = self.state.get("imu", max_age=config.SAFETY_IMU_STALE_S)
        with self._lock:
            if self._check_power() and "low battery" not in self.latched:
                self.latched.add("low battery")
                self._dump(["low-battery"])
            if imu is None:
                status = self.state.get("imu_status") or "no data"
                reasons.append(f"imu missing ({status})")
            else:
                m = self.state.get("motors")
                self._remember(imu, m)
                new = set()
                if imu.tipped:
                    new.add("tipped")
                bump_t = imu.last_bump_t
                if (bump_t is not None and bump_t != self._bump_seen and m and m["moving"]
                        and m["settled_since"] is not None
                        and bump_t > m["settled_since"] + config.SAFETY_BUMP_BLANK_S):
                    peak = self._pitch_peak(bump_t - config.BUMP_CALM_S, bump_t - 0.03)
                    if peak < config.BUMP_CALM_DPS:
                        new.add("bump")
                    else:
                        self.ignored_bumps += 1
                        log.info("bump ignored: nose was moving (pitch rate up to %.0f deg/s), "
                                 "bump %.2f g", peak, imu.bump_g)
                if bump_t is not None:
                    self._bump_seen = bump_t
                fresh = new - self.latched
                self.latched |= new
                if fresh:
                    self._dump(sorted(fresh))
            reasons += sorted(self.latched)
            verdict = {"stop": bool(reasons), "reasons": reasons, "latched": sorted(self.latched)}
        if tuple(reasons) != self._last_reasons:
            if reasons:
                log.warning("safety stop: %s", ", ".join(reasons))
            elif self._last_reasons is not None:
                log.info("safety ok")
            self._last_reasons = tuple(reasons)
        self.state.publish("safety", verdict)
        return verdict

    def snapshot(self, label="manual"):
        """Write the last BLACKBOX_S seconds to a black box file now (operator 'save' key).
        Returns the file path, or None."""
        with self._lock:
            return self._dump([label])

    def _remember(self, imu, m):
        if imu.t == self._last_imu_t:
            return
        self._last_imu_t = imu.t
        applied = m.get("applied", (0.0, 0.0)) if m else (0.0, 0.0)
        self.history.append((
            imu.t, imu.yaw_deg, imu.yaw_rate_dps, imu.pitch_deg, imu.tilt_deg,
            imu.gyro_dps[0], imu.gyro_dps[1], *imu.accel_g, imu.bump_g, applied[0], applied[1],
            int(bool(m and m.get("moving"))), int(bool(m and m.get("settled_since") is not None))))

    def _pitch_peak(self, t0, t1):
        """Largest |pitch rate| seen between t0 and t1 (0 if no data)."""
        return max((abs(row[5]) for row in self.history if t0 <= row[0] <= t1), default=0.0)

    def _dump(self, reasons):
        if self.blackbox_dir is None or not self.history:
            return
        try:
            self.blackbox_dir.mkdir(parents=True, exist_ok=True)
            path = self.blackbox_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{'-'.join(reasons)}.csv"
            t_end = self.history[-1][0]
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(BOX_FIELDS)
                for row in self.history:
                    if row[0] >= t_end - config.BLACKBOX_S:
                        w.writerow([f"{row[0] - t_end:.3f}"] + [f"{v:.4f}" if isinstance(v, float) else v
                                                                for v in row[1:]])
            log.warning("black box: %s", path)
            return path
        except OSError as e:
            log.error("black box not written: %s", e)

    def run(self, stop):
        nxt = self.clock()
        while not stop.is_set():
            self.step()
            nxt += self.period
            delay = nxt - self.clock()
            if delay > 0:
                stop.wait(delay)
            else:
                nxt = self.clock()
