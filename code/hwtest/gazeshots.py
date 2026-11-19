#!/usr/bin/env python3
"""
Gaze test set: where is the person looking? The robot asks by voice (Turkish) and records every frame, so the
"looking at the robot" threshold (robot/gaze.py, config.GAZE_*) is set on real robot views, not guessed.

    python3 code/hwtest/gazeshots.py session          # the robot guides you (speaker); the head is NOT moved
    python3 code/hwtest/gazeshots.py report           # per instruction: facing values, best threshold

session: at 1 m and 2 m - look at me / at your phone / right / left / up / head turned but eyes on me / at me
again; 6 s each. Per frame: facing() from the small-stream landmarks (YuNet) and from the big-stream crop
(SCRFD at 128 and 192 px input); a face crop twice a second (for testing an eye-contact model later).
Folder: ~/scratch/hwtest/gazeshots/<time>/ frames.jsonl + crops/. Camera only: stop alive.py first.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot import camera as rcam, config, faces, gaze   # noqa: E402
from hwtest.common import LOG_DIR                         # noqa: E402

OUT = LOG_DIR / "gazeshots"
DISTANCES = (1, 2)
NUMBER = {1: "bir", 2: "iki"}
LOOKS = (("robot", "Bana bak."),
         ("phone", "Şimdi telefonuna bak."),
         ("right", "Şimdi sağına bak."),
         ("left", "Şimdi soluna bak."),
         ("up", "Şimdi tavana bak."),
         ("eyes_only", "Şimdi başını biraz sola çevir, ama gözlerin bende kalsın."),
         ("robot_again", "Tekrar bana bak."))
POSITIVE = ("robot", "robot_again")          # looking at the robot, head too
NEGATIVE = ("phone", "right", "left", "up")
RECORD_S, SETTLE_S = 6.0, 1.5


def best_face(found):
    return max(found, key=lambda f: f.width) if found else None


def centre_face(found, crop):
    if not found:
        return None
    mid = np.array([crop.shape[1] / 2, crop.shape[0] / 2])
    return min(found, key=lambda f: float(np.hypot(*(np.array(f.centre) - mid))))


def cmd_session(args):
    from robot.tts import Voice
    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / "crops").mkdir(parents=True)
    det = faces.load_detector()
    ref = {"big128": faces.load_detector("scrfd", size=(128, 128), threads=1, min_px=8, min_score=0.25),
           "big192": faces.load_detector("scrfd", size=(192, 192), threads=1, min_px=8, min_score=0.25)}
    voice = Voice()
    cam = rcam.Camera().start()
    frames = open(out / "frames.jsonl", "w")

    def say(text):
        print("says:", text, flush=True)
        voice.say_now(text)

    def record(distance, label, seconds):
        end, n, last_crop = time.monotonic() + seconds, 0, 0.0
        while time.monotonic() < end:
            with cam.frame() as f:
                rgb = f.lores_rgb()
                face = best_face(det.detect(rgb))
                rec = dict(t=round(f.t, 3), distance=distance, label=label, found=face is not None)
                if face is not None:
                    rec.update(width_lores=round(face.width, 1), lores=round(gaze.facing(face.kps), 3))
                    crop, _ = f.crop_main(face.box, pad=0.3)
                    for name, r in ref.items():
                        g = centre_face(r.detect(crop), crop)
                        rec[name] = None if g is None else round(gaze.facing(g.kps), 3)
                        if g is not None:
                            rec[name + "_turn"] = [round(v, 3) for v in gaze.head_turn(g.kps)]
                    if f.t - last_crop >= 0.5:
                        big, _ = f.crop_main(face.box, pad=0.5)
                        name = f"{distance}m-{label}-{n:03d}.jpg"
                        Image.fromarray(big).save(out / "crops" / name, quality=92)
                        rec["crop"], last_crop, n = name, f.t, n + 1
                frames.write(json.dumps(rec) + "\n")
        frames.flush()

    try:
        say("Merhaba Selim. Şimdi nereye baktığını öğreneceğim. Lütfen sesimi takip et.")
        for d in DISTANCES:
            say(f"Lütfen benden {NUMBER[d]} metre uzakta dur ya da otur, ve kameraya bak.")
            time.sleep(args.walk)
            for label, line in LOOKS:
                say(line)
                time.sleep(SETTLE_S)
                record(d, label, RECORD_S)
                print(f"  {d} m {label}: recorded", flush=True)
        say("Teşekkürler, bitti.")
    finally:
        frames.close()
        cam.close()
        voice.close()
    print(f"saved {out}")


# ---------- report ----------

def auc(pos, neg):
    """Probability that a 'looking at the robot' frame scores higher than a 'looking elsewhere' one."""
    if not pos or not neg:
        return float("nan")
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]      # every pair (a few hundred frames each)
    return float(((p > n) + 0.5 * (p == n)).mean())


def smoothed(recs, key, tau=config.GAZE_TAU_S):
    """Engagement as the tracker would have it (exponential smoothing over the frame times)."""
    e, last, out = 0.0, None, []
    for r in recs:
        v = r.get(key)
        if v is None:
            out.append(None)
            continue
        dt = 0.1 if last is None else min(1.0, r["t"] - last)
        last = r["t"]
        e += (v - e) * (1 - np.exp(-dt / tau))
        out.append(e)
    return out


def cmd_report(args):
    sessions = sorted(p for p in OUT.iterdir() if (p / "frames.jsonl").exists()) if OUT.exists() else []
    if not sessions:
        sys.exit("no session yet - run session first")
    folder = OUT / args.session if args.session else sessions[-1]
    recs = [json.loads(x) for x in (folder / "frames.jsonl").read_text().splitlines()]
    keys = ("lores", "big128", "big192")
    lines = [f"# Gaze shots {folder.name}", "",
             "facing() per instruction: median [25 % - 75 %] of the frames with a face; 'found' = frames with a "
             "face / with landmarks on the big crop (128).", "",
             "| distance | look | found | small stream | big crop 128 | big crop 192 | smoothed 128: frames >= ON |",
             "|---|---|---|---|---|---|---|"]
    for d in DISTANCES:
        sel = [r for r in recs if r["distance"] == d]
        sm = smoothed(sel, "big128")
        for label, _ in LOOKS:
            idx = [i for i, r in enumerate(sel) if r["label"] == label]
            rs = [sel[i] for i in idx]
            if not rs:
                continue
            cells = []
            for k in keys:
                v = [r[k] for r in rs if r.get(k) is not None]
                cells.append("-" if not v else
                             f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}]")
            on = [sm[i] for i in idx if sm[i] is not None]
            frac = f"{np.mean([x >= config.GAZE_LOOKING_ON for x in on]):.0%}" if on else "-"
            found = f"{sum(r['found'] for r in rs)}/{len(rs)} / {sum(r.get('big128') is not None for r in rs)}"
            lines.append(f"| {d} m | {label} | {found} | " + " | ".join(cells) + f" | {frac} |")
    lines += ["", "Separation, 'at me' vs phone / right / left / up (AUC: 1 = perfect, 0.5 = chance), and the "
              "threshold with the fewest mistakes:", "", "| distance | source | AUC | best threshold | right at it |",
              "|---|---|---|---|---|"]
    for d in DISTANCES:
        for k in keys:
            pos = [r[k] for r in recs if r["distance"] == d and r["label"] in POSITIVE and r.get(k) is not None]
            neg = [r[k] for r in recs if r["distance"] == d and r["label"] in NEGATIVE and r.get(k) is not None]
            if not pos or not neg:
                continue
            ths = np.linspace(0.05, 0.95, 19)
            acc = [(np.mean([p >= th for p in pos]) + np.mean([n < th for n in neg])) / 2 for th in ths]
            i = int(np.argmax(acc))
            lines.append(f"| {d} m | {k} | {auc(pos, neg):.2f} | {ths[i]:.2f} | {acc[i]:.0%} |")
    lines += ["", f"eyes_only (head turned, eyes on the robot) is what landmarks cannot see - an eye-contact model "
              f"would have to call those frames 'looking'. Crops for that: {folder / 'crops'}"]
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("session")
    s.add_argument("--walk", type=float, default=8.0, help="seconds to get to each distance")
    r = sub.add_parser("report")
    r.add_argument("--session", help="session folder name (default: the latest)")
    args = ap.parse_args()
    {"session": cmd_session, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
