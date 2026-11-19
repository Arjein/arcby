#!/usr/bin/env python3
"""
The robot's recorded data (robot/recorder.py): what is there, how much, how good.

    python3 code/data_cli.py list                      # every session: size, length, records per stream
    python3 code/data_cli.py show [SESSION]            # one session (default: the latest): code, rates, gaps
    python3 code/data_cli.py head STREAM [N] [--session SESSION]   # the first N records of a stream (frames, imu...)

SESSION = "YYYY-MM-DD/HHMMSS" as in the list. Files: config.DATA_DIR/<date>/<time>/ (meta.json, *.jsonl.gz,
keyframes/, faces/). In Python: gzip.open(path, "rt"), one json.loads per line.
"""
import argparse
import gzip
import json
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import config                                  # noqa: E402
from robot.recorder import folder_bytes                   # noqa: E402

ROOT = Path(config.DATA_DIR).expanduser()


def sessions():
    if not ROOT.exists():
        return []
    return sorted((s for d in ROOT.iterdir() if d.is_dir() for s in d.iterdir() if s.is_dir()),
                  key=lambda p: (p.parent.name, p.name))


def records(path):
    """Every record of a .jsonl.gz stream - also of one still being written (stops at the last flush)."""
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    return                          # half-written last line
    except (EOFError, OSError, zlib.error):
        return


def summary(s):
    streams = {p.name.split(".")[0]: list(records(p)) for p in sorted(s.glob("*.jsonl.gz"))}
    ts = [r["t"] for rs in streams.values() for r in rs if "t" in r]
    return streams, (max(ts) - min(ts) if ts else 0.0)


def find(name):
    ss = sessions()
    if not ss:
        sys.exit(f"no recordings in {ROOT}")
    if not name:
        return ss[-1]
    hit = [s for s in ss if f"{s.parent.name}/{s.name}" == name or s.name == name]
    if not hit:
        sys.exit(f"no session {name} (see: data_cli.py list)")
    return hit[-1]


def cmd_list(args):
    total = 0
    for s in sessions():
        streams, length = summary(s)
        size = folder_bytes(s)
        total += size
        keys = len(list((s / "keyframes").glob("*.jpg"))) if (s / "keyframes").exists() else 0
        faces = len(list((s / "faces").rglob("*-face.png"))) if (s / "faces").exists() else 0
        counts = ", ".join(f"{k} {len(v)}" for k, v in streams.items())
        print(f"{s.parent.name}/{s.name}  {length / 60:6.1f} min  {size / 1e6:7.1f} MB  | {counts} | "
              f"keyframes {keys}, faces {faces}")
    print(f"total {total / 1e9:.2f} GB of {config.DATA_MAX_BYTES / 1e9:.1f} GB ({ROOT})")


def cmd_show(args):
    s = find(args.session)
    meta = json.loads((s / "meta.json").read_text())
    streams, length = summary(s)
    print(f"{s}\n  started {meta.get('started')} on {meta.get('host')}, code {meta.get('code')}, "
          f"{meta.get('program')} {' '.join(meta.get('command', [])[1:])}\n  length {length / 60:.1f} min, "
          f"{folder_bytes(s) / 1e6:.1f} MB")
    for name, rs in streams.items():
        ts = sorted(r["t"] for r in rs if "t" in r)
        if len(ts) < 2:
            print(f"  {name:8} {len(rs)} records")
            continue
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        print(f"  {name:8} {len(rs):7d} records  {len(ts) / max(1e-6, ts[-1] - ts[0]):6.1f} /s  "
              f"longest gap {max(gaps):.2f} s")
    off = [r for r in streams.get("system", []) if "off" in r]
    for r in off:
        print(f"  sensor {r['sensor']} off: {r['off']}")


def cmd_head(args):
    s = find(args.session)
    path = s / f"{args.stream}.jsonl.gz"
    if not path.exists():
        sys.exit(f"no stream {args.stream} in {s} (have: {', '.join(p.name.split('.')[0] for p in s.glob('*.jsonl.gz'))})")
    for i, r in enumerate(records(path)):
        if i >= args.n:
            break
        print(json.dumps(r, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sh = sub.add_parser("show")
    sh.add_argument("session", nargs="?")
    hd = sub.add_parser("head")
    hd.add_argument("stream")
    hd.add_argument("n", nargs="?", type=int, default=3)
    hd.add_argument("--session", help="default: the latest")
    args = ap.parse_args()
    {"list": cmd_list, "show": cmd_show, "head": cmd_head}[args.cmd](args)


if __name__ == "__main__":
    main()
