#!/usr/bin/env python3
"""
Showing session (docs/vision.md S3): frames of Selim holding things up to the robot - and doing normal
things that are NOT showing - to set robot/show.py's SHOW_* values from real data. alive.py stopped (the camera).

    python3 code/hwtest/showshots.py snapshot                  # one small picture: is the person in view?
    python3 code/hwtest/showshots.py record --seconds 420      # big-stream frames, 4 a second, onto the stick
    python3 code/hwtest/showshots.py analyse DIR               # the pose + object models over the recording

record writes DIR/frames/<n>.jpg (2304 x 1296, the big stream) + DIR/frames.jsonl (n, wall time, monotonic time);
the phases (what Selim was asked to do, start / end wall times) come from whoever guided him: DIR/phases.json
[{"name": "phone@1m", "show": true, "t0": .., "t1": ..}, ...]. Local only (faces) - delete the folder when done.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot import config                                  # noqa: E402

OUT = Path("/mnt/arcby-data/hwtest/showshots") if config.DATA_ON_DISK else Path.home() / "scratch/hwtest/showshots"


def camera():
    from robot.camera import Camera
    from robot.power import AliveRunning, require_no_alive
    try:
        require_no_alive()
    except AliveRunning as e:
        sys.exit(f"{e}\nstop it first: pkill -f \"code/[a]live.py\"")
    return Camera().start()


def full_main(f):
    w, h = f.lores_size
    crop, _ = f.crop_main((0, 0, w, h), pad=0.0)
    return crop


def snapshot(args):
    from PIL import Image
    cam = camera()
    try:
        for _ in range(15):                     # let exposure settle
            with cam.frame():
                pass
        with cam.frame() as f:
            img = Image.fromarray(full_main(f))
    finally:
        cam.close()
    img.thumbnail((960, 960))
    out = Path(args.out or OUT / "snapshot.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=85)
    print(out)


def record(args):
    from PIL import Image
    folder = Path(args.out or OUT / datetime.now().strftime("%Y%m%d-%H%M%S"))
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    cam = camera()
    every = 1.0 / args.fps
    n, next_t, end = 0, 0.0, time.monotonic() + args.seconds
    print(f"recording into {folder} for {args.seconds:.0f} s", flush=True)
    try:
        with open(folder / "frames.jsonl", "w") as idx:
            while time.monotonic() < end:
                with cam.frame() as f:
                    if f.t < next_t:
                        continue
                    next_t = f.t + every
                    img = Image.fromarray(full_main(f))
                    wall = time.time()
                img.save(folder / "frames" / f"{n:05d}.jpg", "JPEG", quality=85)
                idx.write(json.dumps({"n": n, "wall": round(wall, 3), "t": round(f.t, 3)}) + "\n")
                idx.flush()
                n += 1
    finally:
        cam.close()
    print(f"{n} frames", flush=True)


class FileFrame:
    """A recorded big-stream frame as robot/show.py and robot/body.py see a camera frame."""

    def __init__(self, path, t):
        from PIL import Image
        self.t = t
        self._main = np.asarray(Image.open(path).convert("RGB"))
        self._lores = np.asarray(Image.fromarray(self._main).resize(config.CAMERA_LORES, Image.BILINEAR))
        self.y_plane = (self._lores @ np.array([0.257, 0.504, 0.098]) + 16).astype(np.uint8)

    def lores_rgb(self):
        return self._lores

    def crop_main(self, box, pad=0.4):
        s = self._main.shape[1] / self._lores.shape[1]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        mx1, my1 = max(0, int((x1 - pad * w) * s)), max(0, int((y1 - pad * h) * s))
        mx2 = min(self._main.shape[1], int((x2 + pad * w) * s) + 1)
        my2 = min(self._main.shape[0], int((y2 + pad * h) * s) + 1)
        return np.ascontiguousarray(self._main[my1:my2, mx1:mx2]), (mx1, my1, s)


def analyse(args):
    """Every frame through the faces (looking), bodies and the show logic, inline; per phase: hands held up, object
    runs and what they found, 'showing' events. -> DIR/analysis.json + a table."""
    from robot import body as B, show as S
    from robot.vision import Vision
    folder = Path(args.dir)
    frames = [json.loads(x) for x in (folder / "frames.jsonl").read_text().splitlines()]
    phases = json.loads((folder / "phases.json").read_text()) if (folder / "phases.json").exists() else []
    pose_model, objects = B.load_pose(), S.load_objects()
    bodies = B.BodySense(pose_model, inline=True)
    show = S.ShowSense.inline_for_tests(objects)
    vision = Vision.for_replay() if hasattr(Vision, "for_replay") else None
    rows = []
    for fr in frames:
        f = FileFrame(folder / "frames" / f"{fr['n']:05d}.jpg", fr["t"])
        faces = vision.faces_now(f) if vision is not None else []
        bodies.next_t = -1
        bodies.process(f, lambda t: (0.0, 0.0), 0.0, faces)
        runs = show.stats["runs"]
        ev = show.process(f, lambda t: (0.0, 0.0), bodies.tracker.visible(f.t), faces)
        held = max((h.held_s(f.t) for h in show.hands.values()), default=0.0)
        thing = None
        for h in show.hands.values():
            if h.hits and h.hits[-1][0] == f.t and h.hits[-1][1] is not None:
                thing = (h.hits[-1][1].name, round(h.hits[-1][1].score, 2))
        rows.append(dict(n=fr["n"], wall=fr["wall"], held=round(held, 2), ran=show.stats["runs"] > runs,
                         thing=thing, events=[e["what"] for e in ev if e["kind"] == "showing"]))
    report = []
    for ph in phases:
        rs = [r for r in rows if ph["t0"] <= r["wall"] <= ph["t1"]]
        if not rs:
            continue
        things = [r["thing"][0] for r in rs if r["thing"]]
        report.append(dict(name=ph["name"], show=ph.get("show"), frames=len(rs),
                           held_share=round(sum(r["held"] >= config.SHOW_HOLD_S for r in rs) / len(rs), 2),
                           runs=sum(r["ran"] for r in rs),
                           things={t: things.count(t) for t in set(things)},
                           events=sum(len(r["events"]) for r in rs)))
    (folder / "analysis.json").write_text(json.dumps(dict(rows=rows, report=report), indent=1))
    print(f"{'phase':22} {'show':5} {'frames':>6} {'held':>5} {'runs':>5}  things found")
    for r in report:
        print(f"{r['name']:22} {str(r['show']):5} {r['frames']:6d} {r['held_share']:5.0%} {r['runs']:5d}  {r['things']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--out")
    r = sub.add_parser("record")
    r.add_argument("--seconds", type=float, default=420)
    r.add_argument("--fps", type=float, default=4.0)
    r.add_argument("--out")
    a = sub.add_parser("analyse")
    a.add_argument("dir")
    args = ap.parse_args()
    {"snapshot": snapshot, "record": record, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
