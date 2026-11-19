"""Run the scenarios: python3 -m sim.run [names...] [--seed N] [--real-stt] [-v]   (from code/)
                  or: python3 code/sim/run.py ...
Exit code 1 when a scenario FAILs (XFAIL = a known gap, still failing; SKIP = the local recording is missing)."""
import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):                       # python3 code/sim/run.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "sim"

from sim.scenarios import run_all  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="only scenarios whose name contains one of these")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-stt", action="store_true",
                    help="real Gemini speech-to-text on the simulated audio (COSTS MONEY; off by default)")
    ap.add_argument("-v", "--verbose", action="store_true", help="the robot's own log lines too")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.CRITICAL, format="%(name)s: %(message)s")
    real = None
    if a.real_stt:
        from robot.stt import load_stt
        real = load_stt()
        print("real STT:", getattr(real, "name", real), "- every utterance sent costs a request", file=sys.stderr)
    results = run_all(a.names, seed=a.seed, real_stt=real)
    w = max((len(r.name) for r in results), default=10)
    for r in results:
        print(f"{r.status:5}  {r.name:{w}}  {r.seconds:5.2f} s  {r.title}")
        if r.status != "PASS":
            print(f"{'':5}  {'':{w}}           -> {r.reason()}")
        if r.metrics:
            print(f"{'':5}  {'':{w}}              " + ", ".join(f"{k}={v}" for k, v in r.metrics.items()))
    counts = {s: sum(1 for r in results if r.status == s) for s in ("PASS", "FAIL", "XFAIL", "XPASS", "SKIP")}
    print(" ".join(f"{k} {v}" for k, v in counts.items() if v), f"- {sum(r.seconds for r in results):.1f} s")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
