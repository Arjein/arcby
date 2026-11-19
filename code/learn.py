#!/usr/bin/env python3
"""
Learning mode: the robot improves itself from its own recordings (docs/vision.md, part L).
alive.py starts it when the Pi runs on the wall adapter (battery off) and stops it when the battery is back; it can
also be run by hand (alive.py may keep running - it only reads the recordings).

    python3 code/learn.py                    # run every job (Ctrl+C / SIGTERM stops cleanly; the next run goes on)
    python3 code/learn.py --only faces       # one job
    python3 code/learn.py show               # learnt values, proposals waiting, the last results
    python3 code/learn.py apply ID           # a proposal: take it ...
    python3 code/learn.py reject ID          # ... or not
    python3 code/learn.py undo KEY           # back to the value before the last change (e.g. FACE_MATCH_SIM)

Progress + time left: ~/scratch/learn/state.json (the dashboard's "Learning" card). Learnt values: only in
config.TUNED_FILE, only config.TUNABLE keys within their limits; alive.py reads them at its next start.
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import config                                  # noqa: E402
from robot.learning import Runner, State, Tuned           # noqa: E402

LOG = os.path.expanduser("~/scratch/learn/learn.log")


def jobs(only=None):
    from robot.learn_faces import FaceThresholdJob
    every = [FaceThresholdJob()]
    return [j for j in every if only is None or j.name == only]


def show(tuned, state_path):
    print("learnt values (tuned.json):")
    for k, (lo, hi) in config.TUNABLE.items():
        v = tuned.data["values"].get(k)
        print(f"  {k:22} {v if v is not None else '-':>8}   (default {tuned.defaults.get(k)}, limits {lo} - {hi})")
    print("proposals waiting:" if tuned.data["proposals"] else "proposals waiting: none")
    for p in tuned.data["proposals"]:
        print(f"  {p['id']}: {p['key']} {p['old']} -> {p['new']}  ({p['reason']})")
    try:
        st = json.loads(open(os.path.expanduser(state_path), encoding="utf-8").read())
    except (OSError, ValueError):
        st = {}
    for r in st.get("results", [])[-5:]:
        print(f"{r['job']} ({r['how']}):")
        for line in r.get("findings", []):
            print("  " + line)
    for h in tuned.data["history"][-5:]:
        print(f"history: {h['key']} {h['old']} -> {h['new']} ({h['how']}): {h['reason']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="run", choices=("run", "show", "apply", "reject", "undo"))
    ap.add_argument("arg", nargs="?", help="proposal ID (apply / reject) or KEY (undo)")
    ap.add_argument("--only", help="run only this job (faces)")
    args = ap.parse_args()
    tuned = Tuned()
    if args.what == "show":
        show(tuned, config.LEARN_STATE_FILE)
        return
    if args.what in ("apply", "reject"):
        ok = tuned.decide(args.arg, accept=args.what == "apply")
        sys.exit(0 if ok else f"no proposal {args.arg} (see: learn.py show)")
    if args.what == "undo":
        sys.exit(0 if tuned.undo(args.arg) else f"nothing to undo for {args.arg}")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
    try:
        os.nice(19)                       # alive.py first, always
    except OSError:
        pass
    todo = jobs(args.only)
    if not todo:
        sys.exit(f"no job {args.only!r}")
    runner = Runner(todo, state=State(), tuned=tuned)
    runner.install_signals()
    for name, how in runner.run():
        logging.info("learn: %s: %s", name, how)


if __name__ == "__main__":
    main()
