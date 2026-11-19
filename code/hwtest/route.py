#!/usr/bin/env python3
"""
Floor test: drive a rectangle and come back to the start mark.

There are no wheel encoders yet, so distance comes from time x speed:
'calibrate' measures how many cm the robot moves per second (you measure two
runs with a tape), 'rectangle' then drives the route with IMU heading hold on
the straights and IMU turns at the corners. At the end you measure how far it
stopped from the start mark - the closure error.

Carpet plan (default): 200 x 100 cm, robot starts in the bottom-right corner,
its centre ~20 cm from both edges, facing along the long side, carpet on its LEFT.
Route: 100 cm forward, left 90, 50 cm, left 90, 100 cm, left 90, 50 cm, left 90.

  python3 code/hwtest/route.py                    # calibrate, then the rectangle
  python3 code/hwtest/route.py --only rectangle   # reuse the saved calibration
  python3 code/hwtest/route.py --sim --yes        # simulated robot, no hardware

Put a piece of tape on the carpet at the start: mark where the FRONT edge of the
robot is, and a short line along its side for the direction. Ctrl+C stops at once.

Logs: ~/scratch/hwtest/route-<date>-<time>/ (imu.csv, events.jsonl, summary.json,
report.txt). Calibration: ~/scratch/hwtest/route-calibration.json.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime

import base
import common
from base import idle, move, turn_to
from common import FAIL, INFO, PASS, SKIP, Report
from robot import config

ROUTE = dict(
    speed=0.45,             # drive speed on the straights
    turn_min=0.35,          # turn speed near the target (floor start ~0.30)
    heading_kp=0.01,        # steering per degree of heading error (speed units)
    max_steer=0.1,          # largest steering correction
    cal_short_s=1.0, cal_long_s=2.5,    # the two calibration runs
    long_cm=100.0, short_cm=50.0,       # rectangle sides
    carpet_long_cm=200.0, carpet_short_cm=100.0, margin_cm=20.0,
    closure_ok_cm=20.0,     # closure error that still counts as a pass
)
CAL_FILE = common.LOG_DIR / "route-calibration.json"


# ---------- driving straight with the IMU ----------

def straight(ctx, phase, heading, seconds, est):
    """Drive at args.speed for `seconds`, steering to keep `heading` (deg).
    Updates est (dead-reckoning position) from time, speed model and IMU heading."""
    a = ctx.args
    s = a.speed

    def cmd(el, r):
        err = heading - r.yaw_deg
        steer = max(-a.max_steer, min(a.max_steer, a.heading_kp * err))
        return (s - steer, s + steer)       # err > 0: too far right -> right wheel faster

    readings, why = move(ctx, "route", phase, cmd, max_s=seconds, yaw_limit=45)
    if est is not None and len(readings) > 1:
        cal = ctx.cal
        ramp = s / a.accel if a.accel > 0 else 0.0
        t0 = readings[0].t
        for prev, r in zip(readings, readings[1:]):
            el = r.t - t0
            v = cal["v_cm_s"] * (min(1.0, el / ramp) if ramp else 1.0)
            h = math.radians(r.yaw_deg)
            est["x"] += v * math.cos(h) * (r.t - prev.t)
            est["y"] += v * math.sin(h) * (r.t - prev.t)
        h = math.radians(ctx.last.yaw_deg)
        est["x"] += cal["coast_cm"] * math.cos(h)
        est["y"] += cal["coast_cm"] * math.sin(h)
    errs = [abs(heading - r.yaw_deg) for r in readings]
    return readings, why, max(errs, default=0.0)


def leg_seconds(cal, dist_cm, args):
    """Time to drive dist_cm: d = v * (T - ramp/2) + coast."""
    ramp = args.speed / args.accel if args.accel > 0 else 0.0
    return max(ramp, (dist_cm - cal["coast_cm"]) / cal["v_cm_s"] + ramp / 2)


def measure(ctx, prompt, sim_value):
    """Ask the operator for a number in cm; in --sim use the true simulated value."""
    if ctx.sim is not None:
        print(f"{prompt} [sim: {sim_value:.1f}]")
        return sim_value
    while True:
        txt = common.ask_text(prompt)
        try:
            return float(txt.replace(",", "."))
        except ValueError:
            print("   please type a number, e.g. 34 or -5.5")


# ---------- tests ----------

def test_calibrate(ctx):
    common.heading("Calibrate distance (two straight runs, you measure)")
    a = ctx.args
    print(f"The robot drives straight at {a.speed:.2f} for {a.cal_short_s:.1f} s, then (after you\n"
          f"put it back on the start mark) for {a.cal_long_s:.1f} s. After each run measure from the\n"
          "start tape to the robot's FRONT edge, in cm.")
    runs = []
    for name, secs in (("short", a.cal_short_s), ("long", a.cal_long_s)):
        if ctx.ask(f"Robot on the start mark, facing along the carpet? ({name} run, {secs:.1f} s)") != "y":
            ctx.rep.add("Calibration", SKIP, "operator")
            return False
        if ctx.sim:
            ctx.sim.x = ctx.sim.y = ctx.sim.heading = 0.0
        ctx.imu.filter.reset_yaw()
        idle(ctx, "calibrate", "still", 0.5)
        _, why, worst = straight(ctx, f"{name} {secs:.1f}s", 0.0, secs, None)
        d = measure(ctx, "Distance driven (cm):", ctx.sim.x if ctx.sim else 0.0)
        runs.append({"seconds": secs, "cm": d, "why": why, "heading_err_max": round(worst, 1),
                     "yaw_end": round(ctx.last.yaw_deg, 1)})
        ctx.log.event("calibrate", "run", **runs[-1])
        if why != "time":
            ctx.rep.add(f"Calibration {name} run", FAIL, f"stopped early: {why}")
            return False
    (s1, d1), (s2, d2) = [(r["seconds"], r["cm"]) for r in runs]
    if d2 <= d1:
        ctx.rep.add("Calibration", FAIL, f"long run ({d2} cm) not longer than short run ({d1} cm)")
        return False
    ramp = a.speed / a.accel if a.accel > 0 else 0.0
    v = (d2 - d1) / (s2 - s1)
    coast = d1 - v * (s1 - ramp / 2)
    cal = {"v_cm_s": round(v, 2), "coast_cm": round(coast, 2), "speed": a.speed, "accel": a.accel,
           "runs": runs, "surface": ctx.results.get("setup", {}).get("surface"),
           "date": datetime.now().isoformat(timespec="seconds"), "sim": bool(ctx.sim)}
    ctx.cal = cal
    ctx.results["calibration"] = cal
    if not ctx.sim:
        CAL_FILE.write_text(json.dumps(cal, indent=2) + "\n")
    ctx.log.event("calibrate", "result", **{k: cal[k] for k in ("v_cm_s", "coast_cm")})
    ctx.rep.add("Calibration", PASS, f"{v:.1f} cm/s at {a.speed:.2f}, coast/ramp offset {coast:+.1f} cm")
    return True


def test_rectangle(ctx):
    common.heading("Rectangle and back to the start")
    a = ctx.args
    if getattr(ctx, "cal", None) is None:
        try:
            ctx.cal = json.loads(CAL_FILE.read_text())
        except (OSError, ValueError):
            ctx.rep.add("Rectangle", FAIL, "no calibration - run with --only calibrate first")
            return
        if abs(ctx.cal["speed"] - a.speed) > 1e-6 or abs(ctx.cal["accel"] - a.accel) > 1e-6:
            ctx.rep.add("Rectangle", FAIL, f"calibration was made at speed {ctx.cal['speed']} / "
                        f"accel {ctx.cal['accel']} - re-calibrate for {a.speed} / {a.accel}")
            return
        ctx.rep.fact(f"using saved calibration from {ctx.cal['date']}: {ctx.cal['v_cm_s']} cm/s")
    cal = ctx.cal
    room_long = a.carpet_long_cm - 2 * a.margin_cm
    room_short = a.carpet_short_cm - 2 * a.margin_cm
    if a.long_cm > room_long or a.short_cm > room_short:
        ctx.rep.add("Rectangle fits the carpet", FAIL,
                    f"{a.long_cm:.0f} x {a.short_cm:.0f} cm needs <= {room_long:.0f} x {room_short:.0f}")
        return
    legs = [a.long_cm, a.short_cm, a.long_cm, a.short_cm]
    times = [leg_seconds(cal, d, a) for d in legs]
    print(f"Route: {a.long_cm:.0f} cm forward, LEFT, {a.short_cm:.0f} cm, LEFT, {a.long_cm:.0f} cm, LEFT, "
          f"{a.short_cm:.0f} cm, LEFT -> back on the start mark.\n"
          f"Leg times {[round(t, 2) for t in times]} s at {cal['v_cm_s']} cm/s. About {sum(times) + 12:.0f} s.")
    if ctx.ask("Robot on the start mark, facing along the carpet, carpet on its LEFT?") != "y":
        ctx.rep.add("Rectangle", SKIP, "operator")
        return
    if ctx.sim:
        ctx.sim.x = ctx.sim.y = ctx.sim.heading = 0.0
    ctx.imu.filter.reset_yaw()
    idle(ctx, "route", "still", 1.0)
    est = {"x": 0.0, "y": 0.0}
    legs_out = []
    heading = 0.0
    for i, (d, secs) in enumerate(zip(legs, times), 1):
        _, why, worst = straight(ctx, f"leg {i} {d:.0f}cm", heading, secs, est)
        leg = {"leg": i, "cm": d, "seconds": round(secs, 2), "why": why,
               "heading_err_max": round(worst, 1), "est_x": round(est["x"], 1), "est_y": round(est["y"], 1)}
        if why != "time":
            legs_out.append(leg)
            ctx.rep.add(f"Leg {i}", FAIL, f"stopped early: {why}")
            break
        heading += 90.0
        turn = turn_to(ctx, heading, f"corner {i}", test="route")
        leg["turn_error"] = turn["error"]
        legs_out.append(leg)
        ctx.log.event("route", "leg", **leg)
        ctx.rep.add(f"Leg {i}: {d:.0f} cm + left turn", PASS if abs(turn["error"]) <= 5 else FAIL,
                    f"heading held within {worst:.1f} deg, corner error {turn['error']:+.1f} deg")
    idle(ctx, "route", "end", 1.0)
    yaw_err = ctx.last.yaw_deg - heading
    out = {"legs": legs_out, "final_yaw": round(ctx.last.yaw_deg, 1), "heading_error": round(yaw_err, 1),
           "est_end": {"x": round(est["x"], 1), "y": round(est["y"], 1)}}
    if len(legs_out) == 4 and all(l["why"] == "time" for l in legs_out):
        print("\nMeasure where the robot stopped, compared to the start tape (front edge):")
        fx = measure(ctx, "  FORWARD/back offset (cm, + = in front of the tape, - = behind):",
                     ctx.sim.x if ctx.sim else 0.0)
        ly = measure(ctx, "  LEFT/right offset (cm, + = to the left, - = to the right):",
                     ctx.sim.y if ctx.sim else 0.0)
        closure = math.hypot(fx, ly)
        out.update(measured_end={"x": fx, "y": ly}, closure_cm=round(closure, 1))
        ctx.rep.add("Back at the start", PASS if closure <= a.closure_ok_cm else FAIL,
                    f"{closure:.1f} cm from the mark ({fx:+.1f} forward, {ly:+.1f} left), "
                    f"heading {yaw_err:+.1f} deg")
        ctx.rep.fact(f"dead-reckoning estimate of the end: {est['x']:+.1f} forward, {est['y']:+.1f} left "
                     "(time x speed + IMU heading; compare with the measured offset)")
        if closure > a.closure_ok_cm:
            ctx.rep.hint("Big closure error: re-run calibrate (battery voltage changes the speed), "
                         "check that every corner was ~90 deg, and compare measured vs estimated end.")
    ctx.results["rectangle"] = out
    ctx.log.event("route", "result", **{k: v for k, v in out.items() if k != "legs"})
    base.power_check(ctx, "route")


TESTS = {"calibrate": test_calibrate, "rectangle": test_rectangle}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma separated: " + ",".join(TESTS))
    ap.add_argument("--sim", action="store_true", help="simulated robot (no motors, no IMU)")
    ap.add_argument("--yes", action="store_true", help="answer every prompt with yes (only with --sim)")
    for k, v in {**base.DEFAULTS, **ROUTE}.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v, help=f"default {v}")
    args = ap.parse_args()
    if args.yes and not args.sim:
        ap.error("--yes is only allowed with --sim")
    names = args.only.split(",") if args.only else list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        ap.error(f"unknown test(s): {', '.join(unknown)}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = common.LOG_DIR / f"route-{stamp}{'-sim' if args.sim else ''}"
    log = base.Logger(folder)
    rep = Report("route-sim" if args.sim else "route")
    ctx = None
    try:
        with base.open_robot(args, rep, log) as ctx:
            ctx.cal = None
            log.event("session", "tests", tests=names)
            if base.test_preflight(ctx):
                for n in names:
                    if TESTS[n](ctx) is False:
                        break
    except KeyboardInterrupt:
        print("\nStopped by user.")
        rep.add("Run", SKIP, "aborted with Ctrl+C")
        log.event("session", "aborted")

    summary = {"started": stamp, "sim": args.sim, "git": base.git_version(), "params": vars(args),
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
