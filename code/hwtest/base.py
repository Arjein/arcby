#!/usr/bin/env python3
"""
Hardware test: the whole lower deck ON THE FLOOR - motors + MPU6050 together.
The IMU measures what the motors really did, so no guessing by eye.

Put the robot on the floor with ~1 x 1 m free around it, Pi fixed on the robot,
power cable long enough (or Pi on the battery buck, LiPo alarm on).

  python3 code/hwtest/base.py                  # all tests (~3 min)
  python3 code/hwtest/base.py --only turn90    # one or more tests, comma separated
  python3 code/hwtest/base.py --sim            # simulated robot: no motors, no IMU needed
  python3 code/hwtest/base.py --distance       # also stop forward moves at an obstacle (HC-SR04 on 3.3 V!)

Tests (in order):
  pivot     one wheel at a time, speed rising -> real start speed of each wheel and
            direction on the floor; also checks that the IMU sees the expected turn
  spin      spin left / right on the spot at two speeds -> turn rate per speed
  straight  forward then backward 1.5 s -> how much it drifts; suggests a wheel trim
  turn90    turn exactly +90 deg using the IMU, then back -> final error, overshoot
  spinramp  (only with --only spinramp) both wheels opposite, power rising in small steps
            -> real turn rate per power: how slowly it can turn on the spot

Safety: every move is short (<= 4 s) and stops on Ctrl+C, tipped over, bump,
IMU data missing for 0.2 s, too much turning, or (with --distance) an obstacle.

Everything is logged for tuning in ~/scratch/hwtest/base-<date>-<time>/:
  imu.csv       every IMU sample (100 Hz) with the motor command at that moment
  events.jsonl  every move: start, end, why it stopped, results
  summary.json  all results + the parameters used (feed this back for tuning)
  report.txt    the printed report
"""
import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import common
from common import FAIL, INFO, PASS, SKIP, Report, check_power, heading, release_pins
from robot import config, imu
from robot.state import State

MOTOR_PINS = [17, 27, 22, 23, 24, 25]
LEFT = dict(forward=27, backward=22, enable=17)     # docs/pinout.md
RIGHT = dict(forward=23, backward=24, enable=25)

# Defaults - all can be changed on the command line (see --help) for tuning.
DEFAULTS = dict(
    max_speed=0.8,          # no command above this
    ramp_start=0.15, ramp_step=0.05, ramp_hold=0.5,   # pivot ramp
    moving_dps=10.0,        # turn rate that counts as "the robot moves"
    pivot_yaw_limit=45.0,   # stop a pivot after this much turning
    spin_s=1.0, straight_s=1.5,
    speed=0.6,              # drive speed if the pivot test did not run
    turn_min=0.45, turn_max=0.55, turn_kp=0.004, turn_tol=2.0, turn_timeout=4.0,
    turn_lead_s=0.125,      # stop this early: remaining angle - turn rate x lead (coast);
                            # 0.117-0.129 measured on carpet 2026-10-29 at ~460 deg/s
    accel=2.0,              # soft start: max speed change per second (0 = off). Without it a
                            # 0 -> 0.5 jump lifted the nose ~15 deg (2026-10-29)
    settle_s=1.0,           # logging after each move while it rolls out
    stale_s=0.2, obstacle_cm=15.0,
    bump_blank_s=0.3,       # ignore bumps this long after a motor command change: starting and
                            # stopping jerk the robot (measured 2026-10-29: up to 1.1 g, steady
                            # driving <= 0.26 g, IMU_BUMP_G 0.6)
)


# ---------- logging ----------

class Logger:
    FIELDS = ["t", "test", "phase", "cmd_l", "cmd_r", "fresh", "yaw", "yaw_rate", "roll", "pitch",
              "tilt", "ax", "ay", "az", "gx", "gy", "gz", "bump_g", "dist_cm"]

    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        self._csv_f = open(folder / "imu.csv", "w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(self.FIELDS)
        self._ev = open(folder / "events.jsonl", "w")

    def row(self, test, phase, cmd, r, fresh, dist):
        if r is None:
            return
        self._csv.writerow([
            f"{time.monotonic() - self.t0:.3f}", test, phase, f"{cmd[0]:.2f}", f"{cmd[1]:.2f}", int(fresh),
            f"{r.yaw_deg:.3f}", f"{r.yaw_rate_dps:.3f}", f"{r.roll_deg:.2f}", f"{r.pitch_deg:.2f}",
            f"{r.tilt_deg:.2f}", *(f"{v:.4f}" for v in r.accel_g), *(f"{v:.3f}" for v in r.gyro_dps),
            f"{r.bump_g:.3f}", "" if dist is None else f"{dist:.1f}"])

    def event(self, test, kind, **data):
        rec = {"t": round(time.monotonic() - self.t0, 3), "test": test, "event": kind, **data}
        self._ev.write(json.dumps(rec) + "\n")
        self._ev.flush()

    def close(self):
        self._csv_f.close()
        self._ev.close()


# ---------- drive (real and simulated) ----------

class Drive:
    """Both wheels as signed speeds -1..1 (+ = forward), PWM like robot_drive.py."""

    def __init__(self, max_speed):
        from gpiozero import Motor
        from common import require_battery
        require_battery("the motors")           # no battery: never touch the motor pins
        self.max_speed = max_speed
        self.motors = (Motor(**LEFT), Motor(**RIGHT))

    def set(self, left, right):
        for m, v in zip(self.motors, (left, right)):
            v = max(-self.max_speed, min(self.max_speed, v))
            if v > 0:
                m.forward(v)
            elif v < 0:
                m.backward(-v)
            else:
                m.stop()

    def stop(self):
        for m in self.motors:
            m.stop()

    def close(self):
        self.stop()
        for m in self.motors:
            m.close()
        release_pins(MOTOR_PINS)


class SimRobot:
    """Tiny model: dead band per wheel/direction, lag, weaker right wheel, coasting."""
    START = {("L", 1): 0.30, ("L", -1): 0.30, ("R", 1): 0.35, ("R", -1): 0.40}
    GAIN = {"L": 1.0, "R": 0.93}
    TAU = 0.12
    CM_S = 70.0             # forward speed at command 1.0 (for route.py)

    def __init__(self):
        self.cmd = (0.0, 0.0)
        self.vel = {"L": 0.0, "R": 0.0}
        self.t = time.monotonic()
        self.rnd = random.Random(7)
        self.x = self.y = self.heading = 0.0    # cm forward, cm left, deg (true position)

    def update(self):
        now = time.monotonic()
        dt, self.t = now - self.t, now
        for side, c in zip("LR", self.cmd):
            sign = 1 if c > 0 else -1
            target = 0.0 if abs(c) < self.START[(side, sign)] else c * self.GAIN[side]
            self.vel[side] += (target - self.vel[side]) * min(1.0, dt / self.TAU)
        rate = 180.0 * (self.vel["R"] - self.vel["L"])     # yaw rate deg/s, + = left
        v = self.CM_S * (self.vel["L"] + self.vel["R"]) / 2
        h = math.radians(self.heading)
        self.x += v * math.cos(h) * dt
        self.y += v * math.sin(h) * dt
        self.heading += rate * dt
        return rate

    def imu_device(self, mapping):
        from tests.fakes import FakeMpu6050
        sim = self

        class SimImu(FakeMpu6050):
            def _sample(self):
                yaw_rate = sim.update()
                robot_g = (sim.rnd.gauss(0, 0.05), sim.rnd.gauss(0, 0.05), yaw_rate + sim.rnd.gauss(0, 0.05))
                robot_a = (sim.rnd.gauss(0, 0.003), sim.rnd.gauss(0, 0.003), 1.0 + sim.rnd.gauss(0, 0.003))
                self.gyro, self.accel = to_sensor(mapping, robot_g), to_sensor(mapping, robot_a)
                return super()._sample()
        dev = SimImu()
        dev.close = lambda: None
        return dev


class SimDrive:
    def __init__(self, sim, max_speed):
        self.sim, self.max_speed = sim, max_speed

    def set(self, left, right):
        cap = self.max_speed
        self.sim.cmd = (max(-cap, min(cap, left)), max(-cap, min(cap, right)))

    def stop(self):
        self.sim.cmd = (0.0, 0.0)

    def close(self):
        self.stop()


def to_sensor(mapping, v):
    out = [0.0, 0.0, 0.0]
    for row, (i, sign) in enumerate(mapping):
        out[i] = sign * v[row]
    return tuple(out)


class DistanceThread:
    """Optional HC-SR04 pinger (10 Hz) for the obstacle stop."""

    def __init__(self):
        from distance import Sensor
        self.sensor = Sensor()
        self.cm = None
        self.answered = 0           # pings that got any answer (sensor present)
        self.stop_ev = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.stop_ev.is_set():
            r = self.sensor.ping()
            if r != "noanswer":
                self.answered += 1
            self.cm = r if isinstance(r, float) else None
            self.stop_ev.wait(0.1)

    def close(self):
        self.stop_ev.set()
        self.th.join(timeout=1)
        self.sensor.close()
        release_pins([5, 6])


# ---------- shared context and the move loop ----------

class Ctx:
    def __init__(self, args, rep, log, imu_mod, drive, dist):
        self.args, self.rep, self.log, self.imu, self.drive, self.dist = args, rep, log, imu_mod, drive, dist
        self.last = None            # latest ImuReading
        self.last_t = 0.0
        self.cmd_change_t = 0.0     # when the motor command last changed (bump blanking)
        self.sim = None             # SimRobot in --sim mode (true position for route.py)
        self.results = {}

    def step(self):
        r = self.imu.step()
        if r is not None:
            self.last, self.last_t = r, time.monotonic()
        return r

    def dist_cm(self):
        return self.dist.cm if self.dist else None

    def ask(self, prompt, choices="yn"):
        if self.args.yes:
            print(f"{prompt} [auto: {choices[0]}]")
            return choices[0]
        return common.getkey(prompt, choices)

    def ask_text(self, prompt):
        return "" if self.args.yes else common.ask_text(prompt)


def idle(ctx, test, phase, seconds):
    """Motors off, keep reading + logging the IMU."""
    end = time.monotonic() + seconds
    nxt = time.monotonic()
    while time.monotonic() < end:
        r = ctx.step()
        ctx.log.row(test, phase, (0.0, 0.0), ctx.last, r is not None, ctx.dist_cm())
        nxt += ctx.imu.period
        time.sleep(max(0.0, nxt - time.monotonic()))


def move(ctx, test, phase, cmd_fn, max_s, yaw_limit=None, forward=False):
    """Run cmd_fn(elapsed, reading) -> (left, right) or None (= done) at the IMU rate,
    with the safety checks. Returns (readings while driving, why it stopped)."""
    a = ctx.args
    start = nxt = time.monotonic()
    yaw0 = ctx.last.yaw_deg
    readings, why, cmd = [], "time", (0.0, 0.0)
    ctx.cmd_change_t = start
    last_t = start
    ctx.log.event(test, "move_start", phase=phase, yaw=round(yaw0, 2))
    try:
        while True:
            r = ctx.step()
            now = time.monotonic()
            if r is not None:
                readings.append(r)
            why = safety(ctx, now, yaw0, yaw_limit, forward)
            if why:
                break
            if now - start >= max_s:
                why = "time"
                break
            nxt_cmd = cmd_fn(now - start, ctx.last)
            if nxt_cmd is None:
                why = "done"
                break
            if a.accel > 0:                 # soft start: approach the target step by step
                lim = a.accel * (now - last_t)
                nxt_cmd = tuple(c + max(-lim, min(lim, n - c)) for c, n in zip(cmd, nxt_cmd))
            last_t = now
            if nxt_cmd != cmd:
                ctx.cmd_change_t = now
            cmd = nxt_cmd
            ctx.drive.set(*cmd)
            ctx.log.row(test, phase, cmd, ctx.last, r is not None, ctx.dist_cm())
            nxt += ctx.imu.period
            time.sleep(max(0.0, nxt - time.monotonic()))
    finally:
        ctx.drive.stop()
    ctx.log.event(test, "move_end", phase=phase, why=why, seconds=round(time.monotonic() - start, 2),
                  yaw_change=round(ctx.last.yaw_deg - yaw0, 2), last_cmd=[round(c, 2) for c in cmd])
    idle(ctx, test, phase + " settle", a.settle_s)
    if why not in ("time", "done"):
        ctx.rep.add(f"Safety stop in {test} / {phase}", FAIL if why in ("tipped", "imu stale") else INFO, why)
        if why == "bump":
            ctx.rep.hint("A bump stopped a move. If nothing was hit, the start jerk was taken for a bump: "
                         "look at bump_g in imu.csv and raise IMU_BUMP_G in config.py.")
    return readings, why


def safety(ctx, now, yaw0, yaw_limit, forward):
    r = ctx.last
    if now - ctx.last_t > ctx.args.stale_s:
        return "imu stale"
    if r.tipped:
        return "tipped"
    if r.last_bump_t is not None and r.last_bump_t > ctx.cmd_change_t + ctx.args.bump_blank_s:
        return "bump"                       # only while driving steadily (see bump_blank_s)
    if yaw_limit is not None and abs(r.yaw_deg - yaw0) > yaw_limit:
        return "yaw limit"
    d = ctx.dist_cm()
    if forward and d is not None and d < ctx.args.obstacle_cm:
        return "obstacle"
    return None


def mean_rate(readings, last_s):
    if not readings:
        return 0.0
    t_end = readings[-1].t
    xs = [r.yaw_rate_dps for r in readings if r.t >= t_end - last_s]
    return statistics.fmean(xs) if xs else 0.0


def power_check(ctx, test):
    ok, text = check_power()
    ctx.log.event(test, "power", ok=ok, text=text)
    if ok is False:
        ctx.rep.add(f"Pi power after {test}", FAIL, text)
        ctx.rep.hint("Under-voltage while driving: the motors pull the battery down and the Pi buck "
                     "sags. Charge the battery; check the buck input voltage while driving.")
    return ok is not False


def base_speed(ctx):
    starts = ctx.results.get("pivot", {}).get("start_speed", {})
    vals = [v for v in starts.values() if v is not None]
    if vals:
        return round(min(ctx.args.max_speed, max(vals) + 0.15), 2)
    return ctx.args.speed


# ---------- tests ----------

def test_preflight(ctx):
    heading("Preflight")
    ok, text = check_power()
    ctx.rep.add("Pi power (get_throttled)", SKIP if ok is None else PASS if ok else FAIL, text)
    ctx.log.event("preflight", "power", ok=ok, text=text)
    if ok is False and not ctx.args.sim:
        ctx.rep.hint("Fix the Pi power first (under-voltage).")
        return False
    if ctx.dist is not None:
        time.sleep(1.0)                     # ~10 pings
        if ctx.dist.answered == 0:
            ctx.rep.add("HC-SR04 (--distance)", FAIL, "no answer - is it plugged in?")
            ctx.rep.hint("--distance was given but the HC-SR04 does not answer, so there would be "
                         "NO obstacle stop. Plug it in (3.3 V!) or run without --distance.")
            return False
        ctx.rep.add("HC-SR04 (--distance)", PASS, f"answering, {ctx.dist.cm} cm")
    print("\nRobot ON THE FLOOR, about 1 x 1 m free around it, Pi fixed on the robot,\n"
          "cables long enough to follow 30 cm? Nothing touching the robot?")
    if ctx.ask("Ready?") != "y":
        ctx.rep.add("Operator setup", SKIP, "not ready - aborted")
        return False
    battery = ctx.ask_text("Battery voltage if measured (e.g. 8.1), Enter to skip:")
    surface = ctx.ask_text("Floor type (e.g. laminate, tile, carpet), Enter to skip:")
    ctx.results["setup"] = {"battery_v": battery or None, "surface": surface or None}
    ctx.log.event("preflight", "setup", battery_v=battery or None, surface=surface or None)

    print("   keep the robot still - IMU calibration ...")
    end = time.monotonic() + 4 * config.IMU_CALIB_S + 2
    while ctx.imu.filter.calibrating and time.monotonic() < end:
        ctx.step()
        time.sleep(ctx.imu.period)
    if ctx.imu.filter.calibrating:
        ctx.rep.add("IMU calibration", FAIL, f"robot not still ({ctx.imu.filter.calib_failures} tries)")
        return False
    idle(ctx, "preflight", "still", 2.0)
    bias = [round(b, 3) for b in ctx.imu.filter.bias]
    ctx.rep.add("IMU calibration", PASS, f"gyro bias {bias} deg/s")
    ctx.results["imu"] = {"bias_dps": bias, "who_am_i": getattr(ctx, "who", None)}
    ctx.imu.filter.reset_yaw()
    return True


PIVOTS = [  # wheel, direction, expected yaw sign (+ = left turn)
    ("LEFT", "forward", -1), ("LEFT", "backward", +1),
    ("RIGHT", "forward", +1), ("RIGHT", "backward", -1),
]


def test_pivot(ctx):
    heading("Pivot: one wheel at a time, speed rising")
    a = ctx.args
    print("Each wheel turns the robot around the OTHER wheel, slowly faster until the IMU\n"
          f"sees it turn, then stops (at most {a.pivot_yaw_limit:.0f} deg per run).")
    if ctx.ask("Start?", "yn") != "y":
        ctx.rep.add("Pivot", SKIP, "operator")
        return
    out = {"start_speed": {}, "sign_ok": {}, "rate_at_start": {}}
    for wheel, direction, expect in PIVOTS:
        key = f"{wheel} {direction}"
        sign = 1 if direction == "forward" else -1
        found = {"speed": None}

        def cmd(el, r, wheel=wheel, sign=sign, found=found):
            s = round(a.ramp_start + a.ramp_step * int(el / a.ramp_hold), 2)
            if found["speed"] is not None or s > a.max_speed + 1e-9:
                return None
            if len(recent) >= 20 and abs(statistics.fmean(recent[-20:])) > a.moving_dps:
                found["speed"] = cur["s"]
                found["rate"] = statistics.fmean(recent[-20:])     # measured while moving
                return None
            cur["s"] = s
            return (sign * s, 0.0) if wheel == "LEFT" else (0.0, sign * s)

        recent, cur = [], {"s": None}
        orig_step = ctx.step

        def step_and_track():
            r = orig_step()
            if r is not None:
                recent.append(r.yaw_rate_dps)
            return r
        ctx.step = step_and_track
        try:
            readings, why = move(ctx, "pivot", key, cmd, max_s=20, yaw_limit=a.pivot_yaw_limit)
        finally:
            ctx.step = orig_step
        rate = found.get("rate", 0.0)
        speed = found["speed"]
        ok_sign = speed is not None and (rate > 0) == (expect > 0)
        out["start_speed"][key] = speed
        out["sign_ok"][key] = ok_sign
        out["rate_at_start"][key] = round(rate, 1)
        ctx.log.event("pivot", "result", wheel=key, start_speed=speed, rate_dps=round(rate, 1),
                      expected_sign=expect, sign_ok=ok_sign, why=why)
        if speed is None:
            ctx.rep.add(f"Pivot {key}", FAIL, f"did not move up to {a.max_speed} ({why})")
            ctx.rep.hint(f"{key} never moved the robot: battery low, wheel slipping, or the max speed "
                         "is too low for the floor (try --max-speed 1.0).")
        elif not ok_sign:
            ctx.rep.add(f"Pivot {key}", FAIL, f"turned the WRONG way ({rate:+.1f} deg/s)")
            ctx.rep.hint("Wrong turn direction: either that motor's wires are swapped (but the box test "
                         "said they are right) or IMU_AXES is wrong - run hwtest/imu.py --only axes.")
        else:
            ctx.rep.add(f"Pivot {key}", PASS, f"starts at {speed:.2f}, {rate:+.1f} deg/s")
        idle(ctx, "pivot", key + " pause", 1.0)
    ctx.results["pivot"] = out
    vals = [v for v in out["start_speed"].values() if v is not None]
    if vals:
        need = max(vals)
        ctx.rep.fact(f"floor start speeds: {out['start_speed']}; highest {need:.2f}. "
                     f"robot_drive.py starts at 0.5 and goes down to 0.2.")
        ctx.rep.add("Start margin vs robot_drive 0.5", PASS if need + 0.1 <= 0.5 else FAIL,
                    f"needs >= {need + 0.1:.2f}")
    power_check(ctx, "pivot")


def test_spin(ctx):
    heading("Spin on the spot")
    a = ctx.args
    s1 = base_speed(ctx)
    speeds = sorted({s1, round(min(a.max_speed, s1 + 0.2), 2)})
    print(f"Spins LEFT then RIGHT for {a.spin_s:.1f} s at speeds {speeds}.")
    if ctx.ask("Start?") != "y":
        ctx.rep.add("Spin", SKIP, "operator")
        return
    table = []
    for s in speeds:
        for name, cmd, expect in (("left", (-s, s), 1), ("right", (s, -s), -1)):
            readings, why = move(ctx, "spin", f"{name} {s:.2f}", lambda el, r, c=cmd: c,
                                 max_s=a.spin_s, yaw_limit=360)
            rate = mean_rate(readings, 0.5)
            ok = (rate > 0) == (expect > 0) and abs(rate) > a.moving_dps
            table.append({"dir": name, "speed": s, "rate_dps": round(rate, 1), "ok": ok, "why": why})
            ctx.log.event("spin", "result", **table[-1])
            ctx.rep.add(f"Spin {name} at {s:.2f}", PASS if ok else FAIL, f"{rate:+.1f} deg/s")
            idle(ctx, "spin", "pause", 0.5)
    ctx.results["spin"] = table
    power_check(ctx, "spin")


def test_straight(ctx):
    heading("Straight line")
    a = ctx.args
    s = base_speed(ctx)
    print(f"Forward {a.straight_s:.1f} s, then backward {a.straight_s:.1f} s, at {s:.2f}.")
    if ctx.ask("Start?") != "y":
        ctx.rep.add("Straight", SKIP, "operator")
        return
    spins = [x for x in ctx.results.get("spin", []) if x["ok"]]
    k = statistics.fmean(abs(x["rate_dps"]) / (2 * x["speed"]) for x in spins) if spins else None
    out = {"speed": s, "k_dps_per_speed_diff": None if k is None else round(k, 1)}
    for name, sign in (("forward", 1), ("backward", -1)):
        y0 = ctx.last.yaw_deg
        p0 = ctx.last.pitch_deg
        readings, why = move(ctx, "straight", name, lambda el, r, v=sign * s: (v, v),
                             max_s=a.straight_s, yaw_limit=60, forward=sign > 0)
        drift = ctx.last.yaw_deg - y0
        rate = mean_rate(readings, a.straight_s / 2)
        lift = max((r.pitch_deg - p0 for r in readings), default=0.0)
        res = {"drift_deg": round(drift, 1), "rate_dps": round(rate, 2), "why": why,
               "nose_lift_deg": round(lift, 1), "accel": a.accel}
        ctx.rep.add(f"Nose lift {name}", PASS if lift <= 8 else INFO,
                    f"{lift:+.1f} deg at start (soft start {a.accel}/s; 0 = off)")
        if lift > 8:
            ctx.rep.hint("The front lifts at the start: lower --accel (slower soft start), or more "
                         "weight in front of the wheel axle (upper deck parts).")
        if k:
            # yaw_rate = k * (right - left)  ->  right - left needed to cancel = -rate / k
            diff = -rate / k                    # signed commands, so it works backward too
            res["suggest_left"] = round(sign * s - diff / 2, 3)
            res["suggest_right"] = round(sign * s + diff / 2, 3)
        out[name] = res
        ctx.log.event("straight", "result", dir=name, **res)
        side = "left" if drift > 0 else "right"
        ctx.rep.add(f"Straight {name}", PASS if abs(drift) <= 15 else FAIL,
                    f"drifted {abs(drift):.1f} deg to the {side} in {a.straight_s:.1f} s")
        idle(ctx, "straight", "pause", 0.5)
    if k and "forward" in out:
        f = out["forward"]
        ctx.rep.fact(f"straight trim suggestion at {s:.2f}: left {f['suggest_left']}, right "
                     f"{f['suggest_right']} (from {f['rate_dps']:+.2f} deg/s drift, k = {k:.0f} deg/s "
                     "per unit speed difference)")
    ctx.results["straight"] = out
    power_check(ctx, "straight")


def turn_to(ctx, target, phase, test="turn90"):
    a = ctx.args
    tmin = a.turn_min
    starts = ctx.results.get("pivot", {}).get("start_speed", {})
    if all(starts.get(f"{w} {d}") for w, d, _ in PIVOTS):
        tmin = round(min(a.turn_max, max(starts.values()) + 0.05), 2)
    first = {"dir": None}

    def cmd(el, r):
        err = target - r.yaw_deg
        d = 1 if err > 0 else -1
        if first["dir"] is None:
            first["dir"] = d
        left = err - r.yaw_rate_dps * a.turn_lead_s     # where it would end if we stopped now
        if abs(left) <= a.turn_tol or d != first["dir"] or (left > 0) != (err > 0):
            first["rate_at_stop"] = r.yaw_rate_dps
            return None                     # close enough (with coast), or already past
        s = min(a.turn_max, tmin + a.turn_kp * abs(err))
        return (-d * s, d * s)              # d = +1: spin left

    t0 = time.monotonic()
    readings, why = move(ctx, test, phase, cmd, max_s=a.turn_timeout, yaw_limit=200)
    stop_yaw = readings[-1].yaw_deg if readings else ctx.last.yaw_deg
    final = ctx.last.yaw_deg                # after the settle time
    over = (final - target) * (first["dir"] or 1)
    res = {"target": target, "final_yaw": round(final, 2), "error": round(final - target, 2),
           "yaw_at_stop": round(stop_yaw, 2), "coast_deg": round(final - stop_yaw, 2),
           "overshoot": round(max(0.0, over), 2), "seconds": round(time.monotonic() - t0, 2),
           "turn_min": tmin, "turn_lead_s": a.turn_lead_s, "why": why}
    rate = first.get("rate_at_stop")
    if rate and abs(rate) > 5:
        res["rate_at_stop"] = round(rate, 1)
        # coast = rate x real lead  ->  the lead that would have hit the target exactly
        res["suggest_turn_lead_s"] = round(max(0.0, a.turn_lead_s + (final - target) / rate), 3)
    ctx.log.event(test, "result", phase=phase, **res)
    return res


def test_turn90(ctx):
    heading("Turn 90 deg with the IMU")
    print("Turns LEFT to +90 deg, stops by itself, then turns back to 0.\n"
          "Put a mark on the floor to compare with your eye.")
    if ctx.ask("Start?") != "y":
        ctx.rep.add("Turn 90", SKIP, "operator")
        return
    ctx.imu.filter.reset_yaw()
    idle(ctx, "turn90", "still", 0.5)
    out = []
    for target, phase in ((90.0, "left to +90"), (0.0, "back to 0")):
        res = turn_to(ctx, target, phase)
        out.append(res)
        ok = abs(res["error"]) <= 5
        ctx.rep.add(f"Turn {phase}", PASS if ok else FAIL,
                    f"ended at {res['final_yaw']:+.1f} deg (error {res['error']:+.1f}, coast "
                    f"{res['coast_deg']:+.1f}, {res['seconds']:.1f} s)")
        idle(ctx, "turn90", "pause", 0.5)
    leads = [r["suggest_turn_lead_s"] for r in out if "suggest_turn_lead_s" in r]
    if leads:
        ctx.rep.fact(f"turn coast: suggested --turn-lead-s {statistics.fmean(leads):.3f} "
                     f"(used {ctx.args.turn_lead_s}); measured coast {[r['coast_deg'] for r in out]} deg")
    if out and any(abs(r["error"]) > 5 for r in out) and leads:
        ctx.rep.hint(f"Turn missed the target: re-run with --only turn90 --turn-lead-s "
                     f"{statistics.fmean(leads):.3f} (value from this run's coast).")
    if ctx.ask("Did it look like 90 deg and back to the start?") == "y":
        ctx.rep.fact("operator: the turns looked right")
    else:
        seen = ctx.ask_text("What did you see? (e.g. 'only ~70 deg', 'turned right')")
        ctx.rep.fact(f"operator: {seen}")
        ctx.log.event("turn90", "operator", text=seen)
    ctx.results["turn90"] = out
    power_check(ctx, "turn90")


def test_spinramp(ctx):
    """Both wheels opposite, PWM rising in small steps: the real turn rate at each power.
    Tells how slowly the robot can turn on the spot (SPIN_START in robot/config.py)."""
    heading("Spin ramp: turn rate vs power, both wheels")
    a = ctx.args
    p0, p1, step, hold = 0.18, 0.60, 0.02, 0.6
    steps = int(round((p1 - p0) / step)) + 1
    print(f"Spins on the spot, power {p0:.2f} -> {p1:.2f} in {step} steps ({hold} s each), LEFT then RIGHT.\n"
          "Stops early when it turns faster than 450 deg/s. About 30 s.")
    if ctx.ask("Robot on the floor, nothing within 30 cm?") != "y":
        ctx.rep.add("Spin ramp", SKIP, "operator")
        return
    out = {}
    for name, sign in (("left", 1), ("right", -1)):
        table, cur = [], {"p": None}
        fast = {"hit": False}

        def cmd(el, r, sign=sign):
            i = int(el / hold)
            if i >= steps or fast["hit"]:
                return None
            cur["p"] = round(p0 + step * i, 2)
            return (-sign * cur["p"], sign * cur["p"])
        saved_accel, a.accel = a.accel, 0.0        # exact power levels: no soft start here
        try:
            readings, why = move(ctx, "spinramp", name, cmd, max_s=steps * hold + 0.5, yaw_limit=1e9)
        finally:
            a.accel = saved_accel
        t0 = readings[0].t if readings else 0.0
        for i in range(steps):
            seg = [r.yaw_rate_dps for r in readings if t0 + i * hold + hold / 2 <= r.t < t0 + (i + 1) * hold]
            if not seg:
                break
            rate = statistics.fmean(seg)
            table.append({"pwm": round(p0 + step * i, 2), "rate_dps": round(rate, 1)})
            if abs(rate) > 450:
                break
        started = next((row for row in table if abs(row["rate_dps"]) > 15), None)
        out[name] = {"table": table, "start_pwm": started and started["pwm"],
                     "rate_at_start": started and started["rate_dps"], "why": why}
        ctx.log.event("spinramp", "result", dir=name, **out[name])
        if started:
            ctx.rep.add(f"Spin ramp {name}", INFO,
                        f"starts at {started['pwm']:.2f} with {started['rate_dps']:+.0f} deg/s")
        else:
            ctx.rep.add(f"Spin ramp {name}", FAIL, "never turned")
        ctx.rep.fact(f"spin {name}: " + "  ".join(f"{r['pwm']:.2f}:{r['rate_dps']:+.0f}" for r in table))
        idle(ctx, "spinramp", "pause", 1.5)
    starts = [v["start_pwm"] for v in out.values() if v["start_pwm"]]
    if starts:
        slow = min(abs(v["rate_at_start"]) for v in out.values() if v["start_pwm"])
        ctx.rep.fact(f"suggestion: SPIN_START = {max(starts) - 0.02:.2f} in robot/config.py; slowest spin "
                     f"seen ~{slow:.0f} deg/s (slower turns on the spot need TURN_ON_SPOT = 'pivot')")
    ctx.results["spinramp"] = out
    power_check(ctx, "spinramp")


TESTS = {"pivot": test_pivot, "spin": test_spin, "straight": test_straight, "turn90": test_turn90,
         "spinramp": test_spinramp}
DEFAULT_TESTS = ["pivot", "spin", "straight", "turn90"]     # spinramp only with --only


class open_robot:
    """Context manager: motors (or the simulation), IMU and optional distance sensor
    -> a Ctx. Motors are always stopped and the pins released on exit."""

    def __init__(self, args, rep, log):
        self.args, self.rep, self.log = args, rep, log
        self.drive = self.dist = self.dev = None

    def __enter__(self):
        args = self.args
        mapping = imu.parse_axes(config.IMU_AXES)
        sim = None
        try:
            if args.sim:
                sim = SimRobot()
                self.dev = sim.imu_device(mapping)
                self.drive = SimDrive(sim, args.max_speed)
            else:
                from robot.i2c import I2CDevice
                self.dev = I2CDevice(config.I2C_BUS, config.IMU_ADDR)
                self.drive = Drive(args.max_speed)
                if getattr(args, "distance", False):
                    self.dist = DistanceThread()
            m = imu.Imu(State(), dev=self.dev)
            m.chip = imu.Mpu6050(self.dev)
            who = m.chip.init()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        ctx = Ctx(args, self.rep, self.log, m, self.drive, self.dist)
        ctx.who, ctx.sim = who, sim
        self.log.event("session", "start", sim=args.sim, params=vars(args), git=git_version(),
                       imu_axes=list(config.IMU_AXES), imu_bump_g=config.IMU_BUMP_G,
                       imu_tipped_deg=config.IMU_TIPPED_DEG, who_am_i=who)
        return ctx

    def __exit__(self, *exc):
        if self.drive:
            self.drive.close()
        if self.dist:
            self.dist.close()
        if self.dev and not self.args.sim:
            self.dev.close()
        return False


def git_version():
    try:
        rev = subprocess.run(["git", "-C", str(CODE), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(CODE), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma separated: " + ",".join(TESTS))
    ap.add_argument("--sim", action="store_true", help="simulated robot (no motors, no IMU)")
    ap.add_argument("--yes", action="store_true", help="answer every prompt with yes (only with --sim)")
    ap.add_argument("--distance", action="store_true", help="HC-SR04 obstacle stop (sensor on 3.3 V)")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v, help=f"default {v}")
    args = ap.parse_args()
    if args.yes and not args.sim:
        ap.error("--yes is only allowed with --sim")
    names = args.only.split(",") if args.only else list(DEFAULT_TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        ap.error(f"unknown test(s): {', '.join(unknown)}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = common.LOG_DIR / f"base-{stamp}{'-sim' if args.sim else ''}"
    log = Logger(folder)
    rep = Report("base-sim" if args.sim else "base")
    ctx = None
    try:
        with open_robot(args, rep, log) as ctx:
            log.event("session", "tests", tests=names)
            if test_preflight(ctx):
                for n in names:
                    TESTS[n](ctx)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        rep.add("Run", SKIP, "aborted with Ctrl+C")
        log.event("session", "aborted")

    summary = {"started": stamp, "sim": args.sim, "git": git_version(), "params": vars(args),
               "imu_axes": list(config.IMU_AXES), "results": ctx.results if ctx else {},
               "failed": rep.failed}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = rep.text()
    (folder / "report.txt").write_text(text + "\n")
    log.event("session", "end", failed=rep.failed)
    log.close()
    print("\n" + text)
    print(f"\nLogs: {folder}/ (imu.csv, events.jsonl, summary.json, report.txt)")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
