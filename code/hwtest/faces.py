#!/usr/bin/env python3
"""
Face model bake-off: which detector / embedder (robot/faces.py) is best on this robot?

    python3 code/hwtest/faces.py capture selim-2m --count 5     # robot camera photos for the tests below
    python3 code/hwtest/faces.py speed                         # ms per frame / per face on this machine
    python3 code/hwtest/faces.py range                         # which detector finds the faces in the captures
    python3 code/hwtest/faces.py recog                         # recognition accuracy (LFW + captures)

Captures go to ~/scratch/hwtest/faces/captures/LABEL-NN.jpg; name labels "<person>-<distance>m"
(e.g. selim-1m, selim-3m). Reports: ~/scratch/hwtest/faces/<test>-<time>.md (+ .json).

recog uses LFW (Labeled Faces in the Wild, public: ~/scratch/datasets/lfw/<Name>/<Name>_0001.jpg).
LFW photos are sharp and well lit, so every model looks great on them; the test therefore also makes them
look like the robot camera's (CONDS: small = far away, sensor noise, dark) and enrols from clean photos, as
the robot will: meet someone close up, recognise them later further away or in worse light.
Captures labelled "<person>-..." join the "household" (closest photos with a face = enrolment).
recog only needs files, so it can run on the Mac too (--lfw / --captures / --models; speed only on the Pi).
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot import faces                                   # noqa: E402
from hwtest.common import LOG_DIR                         # noqa: E402

OUT = LOG_DIR / "faces"
CAPTURES = OUT / "captures"
LFW = Path.home() / "scratch" / "datasets" / "lfw"
SIZES = ((320, 192), (480, 288), (640, 384))
REF_SIZE = (640, 640)           # detector input for the recognition test (landmarks; same faces for every embedder)
MODELS_DIR = None               # None = config.MODELS_DIR (--models)


def available(kind):
    return [n for n, m in faces.MODELS.items() if m["kind"] == kind and os.path.exists(faces.model_path(n, MODELS_DIR))]


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def save_report(name, lines, data):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
    text = "\n".join(lines) + "\n"
    stem.with_suffix(".md").write_text(text)
    stem.with_suffix(".json").write_text(json.dumps(data, indent=1, default=float))
    print(text)
    print(f"report: {stem}.md")


def captures():
    """label -> sorted capture paths."""
    out = {}
    for p in sorted(CAPTURES.glob("*.jpg")):
        out.setdefault(p.stem.rsplit("-", 1)[0], []).append(p)
    return out


def distance_m(label):
    try:
        return float(label.rsplit("-", 1)[1].rstrip("m"))
    except (IndexError, ValueError):
        return 0.0


# ---------- capture ----------

def cmd_capture(args):
    from picamera2 import Picamera2
    CAPTURES.mkdir(parents=True, exist_ok=True)
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (1536, 864)}))   # full field of view
    cam.start()
    try:
        cam.set_controls({"AfMode": 2})         # continuous autofocus
    except Exception:  # noqa: BLE001 - fixed focus is fine too
        pass
    time.sleep(1.5)                             # exposure + focus settle
    n0 = len(list(CAPTURES.glob(f"{args.label}-*.jpg")))
    try:
        for i in range(args.count):
            path = CAPTURES / f"{args.label}-{n0 + i:02d}.jpg"
            cam.capture_image("main").convert("RGB").save(path, quality=92)
            print(path)
            if i + 1 < args.count:
                time.sleep(args.every)
    finally:
        cam.close()


# ---------- speed ----------

def median_ms(fn, inputs, repeat):
    fn(inputs[0])                               # warm-up (first run allocates)
    times = []
    for i in range(repeat):
        t = time.perf_counter()
        fn(inputs[i % len(inputs)])
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def cmd_speed(args):
    frames = [load_rgb(p) for ps in captures().values() for p in ps][:10]
    if not frames:
        frames = [np.random.default_rng(0).integers(0, 255, (864, 1536, 3), np.uint8)]
    lines = [f"# Face model speed ({os.uname().nodename}, {args.threads} threads, "
             f"{'captures' if len(frames) > 1 else 'random frame'} 1536x864)", "",
             "| detector | input | ms / frame | faces found (first frame) |", "|---|---|---|---|"]
    data = {"detectors": [], "embedders": []}
    for name in available("detector"):
        for size in SIZES:
            det = faces.load_detector(name, size=size, threads=args.threads, models_dir=MODELS_DIR)
            ms = median_ms(det.detect, frames, args.repeat)
            n = len(det.detect(frames[0]))
            lines.append(f"| {name} | {size[0]}x{size[1]} | {ms:.1f} | {n} |")
            data["detectors"].append(dict(name=name, size=size, ms=ms, found=n))
    lines += ["", "| embedder | ms / face | vector |", "|---|---|---|"]
    crop = np.random.default_rng(1).integers(0, 255, (112, 112, 3), np.uint8)
    for name in available("embedder"):
        emb = faces.load_embedder(name, threads=args.threads, models_dir=MODELS_DIR)
        ms = median_ms(emb.embed, [crop], args.repeat)
        lines.append(f"| {name} | {ms:.1f} | {len(emb.embed(crop))}-d |")
        data["embedders"].append(dict(name=name, ms=ms))
    save_report("speed", lines, data)


# ---------- range ----------

def cmd_range(args):
    caps = captures()
    if not caps:
        sys.exit(f"no captures in {CAPTURES} - run: faces.py capture selim-2m --count 3")
    labels = sorted(caps, key=lambda lb: (lb.rsplit("-", 1)[0], distance_m(lb)))
    frames = {lb: [load_rgb(p) for p in caps[lb]] for lb in labels}
    cols = [(n, s) for n in available("detector") for s in SIZES]
    lines = ["# Face detection range (one person per photo: found / photos, face width px, score)", "",
             "| capture | " + " | ".join(f"{n} {s[0]}x{s[1]}" for n, s in cols) + " |",
             "|---|" + "---|" * len(cols)]
    data = {}
    rows = {lb: [] for lb in labels}
    for n, s in cols:
        det = faces.load_detector(n, size=s, threads=args.threads, models_dir=MODELS_DIR)
        for lb in labels:
            found = [det.detect(f) for f in frames[lb]]
            best = [max(fs, key=lambda f: f.width) for fs in found if fs]
            hit = len(best)
            w = statistics.median(f.width for f in best) if best else 0
            sc = statistics.median(f.score for f in best) if best else 0
            extra = max(len(fs) for fs in found) > 1
            rows[lb].append(f"{hit}/{len(found)} {w:.0f}px {sc:.2f}" + (" +extra" if extra else ""))
            data.setdefault(lb, {})[f"{n} {s[0]}x{s[1]}"] = dict(found=hit, photos=len(found), width=w,
                                                                 score=sc, extra_faces=extra)
    for lb in labels:
        lines.append(f"| {lb} | " + " | ".join(rows[lb]) + " |")
    lines += ["", "+extra = some photo had more than one face (a false detection if only one person was there)"]
    save_report("range", lines, data)


# ---------- recognition ----------

def central_face(det, rgb):
    found = det.detect(rgb)
    if not found:
        return None
    h, w = rgb.shape[:2]
    return min(found, key=lambda f: (f.centre[0] - w / 2) ** 2 + (f.centre[1] - h / 2) ** 2 - f.width ** 2)


# Conditions for the recognition test: how the robot camera might see the face (LFW photos are clean).
#   px    shrink the photo so the face is this wide (distance), LANCZOS like a good camera scaler
#   dark  multiply the light (0.2 = a dim room / a face against a bright ceiling lamp)
#   noise sensor noise, standard deviation in 0-255 grey levels (more gain in the dark = more noise)
#   fix   robot/faces.even_light on the aligned crop before the embedder
CONDS = {
    "full": {},
    "48px": dict(px=48),
    "32px": dict(px=32),
    "40px+noise": dict(px=40, noise=6),
    "dark": dict(dark=0.2, noise=3),
    "dark+fix": dict(dark=0.2, noise=3, fix=True),
    "40px+dark": dict(px=40, dark=0.2, noise=3),
    "40px+dark+fix": dict(px=40, dark=0.2, noise=3, fix=True),
}
STRANGER_RATE = 0.01            # the robot may greet 1 stranger in 100 by someone's name - not more


def degraded_crop(rgb, face, cond, rng):
    """Aligned 112 x 112 face as the robot camera might see it under cond (see CONDS)."""
    kps = face.kps
    img = rgb
    if cond.get("px") and face.width > cond["px"]:
        f = cond["px"] / face.width
        img = np.asarray(Image.fromarray(rgb).resize((max(1, round(rgb.shape[1] * f)),
                                                      max(1, round(rgb.shape[0] * f))), Image.LANCZOS))
        kps = kps * f
    if cond.get("dark") or cond.get("noise"):
        x = img.astype(np.float32) * cond.get("dark", 1.0)
        if cond.get("noise"):
            x += rng.normal(0, cond["noise"], x.shape)
        img = np.clip(x, 0, 255).astype(np.uint8)
    crop = faces.align(img, kps)
    return faces.even_light(crop) if cond.get("fix") else crop


def pick_lfw(seed, n_house, n_pairs):
    people = {}
    for d in sorted(LFW.iterdir()):
        if d.is_dir():
            people[d.name] = sorted(d.glob("*.jpg"))
    rng = random.Random(seed)
    big = [p for p, imgs in people.items() if len(imgs) >= 8]
    house = rng.sample(big, n_house)
    pairs = rng.sample([p for p, imgs in people.items() if len(imgs) >= 2 and p not in house], n_pairs)
    return ({p: people[p][:8] for p in house}, {p: rng.sample(people[p], 2) for p in pairs})


def tar_at_far(gen, imp, far):
    thr = float(np.quantile(imp, 1 - far))
    return float(np.mean(gen > thr)), thr


def open_set(gallery, probes, strangers, thr):
    """Robot question 'who is this?' over the household: (right, wrong person, not recognised) shares of
    the probes and the share of strangers greeted by someone's name, at similarity threshold thr."""
    names = list(gallery)
    def best(v):
        sims = [float((gallery[n] @ v).max()) for n in names]
        i = int(np.argmax(sims))
        return names[i], sims[i]
    right = wrong = missed = 0
    for person, v in probes:
        who, sim = best(v)
        if sim < thr:
            missed += 1
        elif who == person:
            right += 1
        else:
            wrong += 1
    let_in = sum(1 for v in strangers if best(v)[1] >= thr)
    n = max(1, len(probes))
    return right / n, wrong / n, missed / n, let_in / max(1, len(strangers))


def cmd_recog(args):
    if not LFW.exists():
        sys.exit(f"LFW not found at {LFW} (download lfw.tgz from https://ndownloader.figshare.com/files/5976018)")
    house, pairs = pick_lfw(args.seed, args.house, args.pairs)
    for lb, ps in captures().items():           # captures join the household, per person
        person = "capture:" + lb.rsplit("-", 1)[0]
        house.setdefault(person, [])
        house[person] += ps
    for person in house:                        # enrol from the closest captures first
        if person.startswith("capture:"):
            house[person].sort(key=lambda p: (distance_m(p.stem.rsplit("-", 1)[0]), p.name))
    images = sorted({p for ps in house.values() for p in ps} | {p for ps in pairs.values() for p in ps})
    print(f"{len(images)} photos: household {len(house)} people, {len(pairs)} genuine pairs")

    det = faces.load_detector(args.det, size=REF_SIZE, threads=args.threads, models_dir=MODELS_DIR)
    found = {}
    for i, p in enumerate(images):
        rgb = load_rgb(p)
        f = central_face(det, rgb)
        if f is not None:
            found[p] = (rgb, f)
        if i % 200 == 0:
            print(f"  detecting {i}/{len(images)}")
    print(f"faces found in {len(found)}/{len(images)} photos")
    real = {p for p in found if str(p).startswith(str(CAPTURES))}

    lines = ["# Face recognition bake-off", "",
             f"LFW seed {args.seed}: household {len(house)} people (enrol 3 photos each, clean), "
             f"{len(pairs)} other people = strangers + same-person pairs; landmarks from {args.det} "
             f"{REF_SIZE[0]}x{REF_SIZE[1]}; faces found {len(found)}/{len(images)}.", "",
             f"One threshold per embedder, set on clean photos so that {STRANGER_RATE:.0%} of strangers get "
             "greeted by a household name; the same threshold is used in every condition (as on the robot).",
             "Verification column: same-person pairs accepted at 0.1 % false pairs (both photos degraded).", "",
             "| embedder | ms/face | condition | verif. TAR@0.1% | d' | household right | wrong person | "
             "not recognised | strangers greeted | real captures right |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    data = {}
    for name in available("embedder"):
        emb = faces.load_embedder(name, threads=args.threads, models_dir=MODELS_DIR)
        vecs, times = {}, []
        for cname, cond in CONDS.items():
            rng = np.random.default_rng(args.seed)   # same noise for every embedder
            for p, (rgb, f) in found.items():
                if p in real and cname != "full":
                    continue                    # real captures are used as they are
                crop = degraded_crop(rgb, f, cond, rng)
                t = time.perf_counter()
                vecs[p, cname] = emb.embed(crop)
                times.append((time.perf_counter() - t) * 1000)
        ms = statistics.median(times)
        gallery = {}
        for person, ps in house.items():
            ps = [p for p in ps if p in found]
            if ps:
                gallery[person] = np.array([vecs[p, "full"] for p in ps[:3]])

        def probes_for(cname):
            out = []
            for person, ps in house.items():
                for p in [p for p in ps if p in found][3:]:
                    out.append((person, vecs[p, "full" if p in real else cname], p in real))
            return out

        def strangers_for(cname):
            return [vecs[a, cname] for a, _ in pairs.values() if a in found]

        # threshold: the best-match similarity of the strangers on clean photos, top STRANGER_RATE
        names = list(gallery)
        clean_best = [max(float((gallery[n] @ v).max()) for n in names) for v in strangers_for("full")]
        thr = float(np.quantile(clean_best, 1 - STRANGER_RATE))
        for cname in CONDS:
            gen = np.array([float(vecs[a, cname] @ vecs[b, cname]) for a, b in pairs.values()
                            if a in found and b in found])
            m = np.array(strangers_for(cname))
            imp = (m @ m.T)[np.triu_indices(len(m), 1)]
            t01, _ = tar_at_far(gen, imp, 0.001)
            dprime = (gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2)
            probes = probes_for(cname)
            right, wrong, missed, let_in = open_set(gallery, [(q, v) for q, v, _ in probes],
                                                    strangers_for(cname), thr)
            rp = [(q, v) for q, v, is_real in probes if is_real]
            real_right = open_set(gallery, rp, [], thr)[0] if rp else None
            row = dict(ms=ms, thr=thr, tar01=t01, dprime=float(dprime), right=right, wrong=wrong, missed=missed,
                       let_in=let_in, probes=len(probes), real_right=real_right, real_probes=len(rp))
            data.setdefault(name, {})[cname] = row
            real_txt = f"{round(real_right * len(rp))}/{len(rp)}" if rp else "-"
            lines.append(f"| {name} | {ms:.0f} | {cname} | {t01:.1%} | {dprime:.2f} | {right:.0%} | {wrong:.1%} | "
                         f"{missed:.0%} | {let_in:.1%} | {real_txt} |")
        lines.append(f"| | | threshold {thr:.3f} | | | | | | | |")
        print(f"{name} done")
    caps = [p for p in house if p.startswith("capture:")]
    if caps:
        lines += ["", f"Real robot captures in the household: {', '.join(caps)} - enrolled from the closest "
                      "photos with a face, probed with the rest as they are (same in every row)."]
    save_report(f"recog-{args.det}", lines, data)


def main():
    global OUT, CAPTURES, LFW, MODELS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", help=f"reports folder (default {OUT})")
    ap.add_argument("--captures", help="robot photos folder (default OUT/captures)")
    ap.add_argument("--lfw", help=f"LFW folder (default {LFW})")
    ap.add_argument("--models", help="models folder (default config.MODELS_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("label")
    c.add_argument("--count", type=int, default=3)
    c.add_argument("--every", type=float, default=1.0, help="seconds between photos")
    s = sub.add_parser("speed")
    s.add_argument("--repeat", type=int, default=20)
    sub.add_parser("range")
    r = sub.add_parser("recog")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--det", default="scrfd", choices=sorted(faces.DETECTORS), help="landmarks from this detector")
    r.add_argument("--house", type=int, default=20, help="LFW people in the household")
    r.add_argument("--pairs", type=int, default=600, help="other LFW people (genuine pairs + strangers)")
    args = ap.parse_args()
    OUT = Path(args.out).expanduser() if args.out else OUT
    CAPTURES = Path(args.captures).expanduser() if args.captures else OUT / "captures"
    LFW = Path(args.lfw).expanduser() if args.lfw else LFW
    MODELS_DIR = args.models
    {"capture": cmd_capture, "speed": cmd_speed, "range": cmd_range, "recog": cmd_recog}[args.cmd](args)


if __name__ == "__main__":
    main()
