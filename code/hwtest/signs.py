#!/usr/bin/env python3
"""
Hand signs (robot/gestures.py) replayed on the robot's own recordings: every pose-model result alive.py kept (the
"frames" stream, robot/recorder.py) goes through the body tracker and the sign detectors again, with the thresholds
in config now. Before a "gel gel" may start a drive, a long stretch of normal life must give NO beckon (each one would
be a drive nobody asked for) - docs/navigation.md part 3.

    python3 code/hwtest/signs.py replay                 # every session in DATA_DIR
    python3 code/hwtest/signs.py replay 2026-11-08/154322 2026-11-08/160010
    python3 code/hwtest/signs.py replay --last 5 --list # the 5 newest, every sign with its time

Prints per session how long bodies were in view and the signs found; the whole result goes to
~/scratch/hwtest/signs-replay.json. No pictures, no names: keypoints only.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np                                  # noqa: E402

from robot import config                            # noqa: E402
from robot.body import Body, BodyTracker           # noqa: E402

OUT = Path.home() / "scratch" / "hwtest" / "signs-replay.json"


def replay_records(records, size=config.CAMERA_LORES):
    """Pose results of one session -> {"bodies_s": seconds with a body in view, "signs": [(t, kind, body id)]}."""
    tracker = BodyTracker()
    w, h = size
    signs, said = [], {}
    in_view, last_t = 0.0, None
    for r in records:
        found = r.get("bodies")
        if found is None:
            continue
        t = float(r["t"])
        pose = tuple(r.get("head") or (0.0, 0.0))
        bodies = [Body(np.array(b["box"], np.float32), float(b["score"]), np.array(b["kps"], np.float32))
                  for b in found if len(b.get("kps", [])) == 17]
        tracker.update(bodies, t, w, h, pose)
        if bodies and last_t is not None and t - last_t < 2.0:
            in_view += t - last_t
        last_t = t
        for tr in tracker.visible(t):
            for kind, at, again in (("beckoned", tr.beckoned_t, config.GESTURE_BECKON_AGAIN_S),
                                    ("stop_sign", tr.stop_t, config.GESTURE_STOP_AGAIN_S)):
                if at == t and t - said.get((tr.id, kind), -math.inf) >= again:
                    said[(tr.id, kind)] = t
                    signs.append((round(t, 2), kind, tr.id))
    return {"bodies_s": round(in_view, 1), "signs": signs}


def cmd_replay(args):
    import data_cli
    sessions = data_cli.sessions()
    if args.sessions:
        sessions = [data_cli.find(s) for s in args.sessions]
    elif args.last:
        sessions = sessions[-args.last:]
    if not sessions:
        sys.exit(f"no sessions in {data_cli.ROOT}")
    report, total_s, total = {}, 0.0, {"beckoned": 0, "stop_sign": 0}
    for s in sessions:
        path = s / "frames.jsonl.gz"
        if not path.exists():
            continue
        res = replay_records(data_cli.records(path))
        name = f"{s.parent.name}/{s.name}"
        report[name] = res
        total_s += res["bodies_s"]
        kinds = [k for _, k, _ in res["signs"]]
        for k in total:
            total[k] += kinds.count(k)
        print(f"{name}: bodies in view {res['bodies_s'] / 60:6.1f} min - beckoned {kinds.count('beckoned')}, "
              f"stop sign {kinds.count('stop_sign')}")
        if args.list:
            for t, k, bid in res["signs"]:
                print(f"    t {t:10.2f}  {k:9s}  body {bid}")
    hours = total_s / 3600
    print(f"\nall: bodies in view {hours:.2f} h - beckoned {total['beckoned']} "
          f"({total['beckoned'] / hours if hours else 0:.1f} an hour), stop sign {total['stop_sign']}")
    if total["beckoned"]:
        print("  every beckon above is a drive nobody asked for (if nobody beckoned then): tighten GESTURE_BECKON_*")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {
        k: getattr(config, k) for k in dir(config) if k.startswith("GESTURE_")}, "sessions": report,
        "bodies_h": round(hours, 2), "total": total}, indent=1))
    print(f"report: {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="the sign detectors on recorded sessions")
    rp.add_argument("sessions", nargs="*", help="YYYY-MM-DD/HHMMSS (default: all)")
    rp.add_argument("--last", type=int, help="only the newest N sessions")
    rp.add_argument("--list", action="store_true", help="every sign found, with its time")
    args = ap.parse_args()
    cmd_replay(args)


if __name__ == "__main__":
    main()
