#!/usr/bin/env python3
"""
Hardware test: how fast navigation's two speeds really are (no wheel encoders: the sonar measures it).

Put the robot on the floor it will drive on, facing a wall or a big flat box squarely, ~150 cm away, nothing in
between. Each run drives straight at the wall for ~1.5 s while the sonar pings; the slope of distance over time =
the speed. It stops by itself 40 cm before the wall (and safety still stops on a bump). After each run put it back
by hand (it never reverses blind).

  python3 code/hwtest/navcal.py              # crawl and cruise, 2 runs each (~3 min)
  python3 code/hwtest/navcal.py --runs 1

Prints the measured cm/s and the config.py lines to change (NAV_CRAWL_CM_S, NAV_CRUISE_CM_S); report in
~/scratch/hwtest/navcal-<time>.json. Needs: alive.py stopped, battery on.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from robot import config        # noqa: E402

REPORT_DIR = Path.home() / "scratch" / "hwtest"
STOP_CM = 40.0
DRIVE_S = 1.5
SKIP_S = 0.4                    # the soft start: speed is not steady before this


def slope(samples, skip_s=SKIP_S):
    """cm/s from [(t, cm)] while driving at a wall (least squares after the soft start), None if too few."""
    if not samples:
        return None
    t0 = samples[0][0]
    pts = [(t - t0, d) for t, d in samples if t - t0 >= skip_s]
    if len(pts) < 5:
        return None
    mt = statistics.fmean(t for t, _ in pts)
    md = statistics.fmean(d for _, d in pts)
    den = sum((t - mt) ** 2 for t, _ in pts)
    if den <= 0:
        return None
    return -sum((t - mt) * (d - md) for t, d in pts) / den


def run_once(robot, sonar, v):
    start = sonar.measure(5)
    if start is None or start < STOP_CM + 50:
        return {"error": f"wall at {start} cm - put the robot ~150 cm from a wall"}
    samples, why = [], "time"
    t0 = time.monotonic()
    while time.monotonic() - t0 < DRIVE_S:
        robot.motors.request_velocity(v, 0.0, "navcal")
        s = robot.state.get("safety") or {}
        if s.get("stop"):
            why = "safety: " + ", ".join(s.get("reasons", []))
            break
        r = sonar.ping()
        if isinstance(r, float):
            samples.append((time.monotonic(), r))
            if r < STOP_CM:
                why = "near the wall"
                break
        time.sleep(1.0 / config.NAV_SONAR_HZ)
    robot.motors.request_velocity(0.0, 0.0, "navcal")
    time.sleep(0.6)
    return {"v": v, "start_cm": start, "speed_cm_s": slope(samples), "pings": len(samples), "ended": why,
            "samples": [[round(t - t0, 3), d] for t, d in samples]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    from robot.main import Robot
    from robot.power import require_battery, require_no_alive
    from robot.sonar import Sonar
    require_no_alive()
    require_battery("the motors")
    report = {"time": datetime.now().isoformat(timespec="seconds"), "runs": []}
    sonar = Sonar()
    try:
        with Robot() as robot:
            print("IMU calibrating - keep the robot still...")
            time.sleep(3.0)
            for name, v in (("crawl", config.NAV_V_CRAWL), ("cruise", config.NAV_V_CRUISE)):
                for i in range(args.runs):
                    input(f"\n{name} run {i + 1}: robot ~150 cm from the wall, facing it. Enter = go ")
                    r = run_once(robot, sonar, v)
                    r["name"] = name
                    report["runs"].append(r)
                    print(f"  {r.get('error') or '%.1f cm/s (%d pings, %s)' % (r['speed_cm_s'] or 0, r['pings'], r['ended'])}")
    finally:
        sonar.close()
    got = {}
    for name in ("crawl", "cruise"):
        vals = [r["speed_cm_s"] for r in report["runs"] if r["name"] == name and r.get("speed_cm_s")]
        if vals:
            got[name] = round(statistics.median(vals), 1)
    report["result"] = got
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"navcal-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=1))
    print("\nResult:", got or "nothing measured")
    if "crawl" in got:
        print(f"  config.py: NAV_CRAWL_CM_S = {got['crawl']}   (now {config.NAV_CRAWL_CM_S})")
    if "cruise" in got:
        print(f"  config.py: NAV_CRUISE_CM_S = {got['cruise']}   (now {config.NAV_CRUISE_CM_S})")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
