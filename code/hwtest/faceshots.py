#!/usr/bin/env python3
"""
Real robot face test set: does a sharp full-resolution photo recognise people better than the video stream?

    python3 code/hwtest/faceshots.py guided --level 0.3      # the robot tells you where to stand (speaker)
    python3 code/hwtest/faceshots.py guided --level 0.3 --spot --hdr   # light measured on the face, HDR
    python3 code/hwtest/faceshots.py report                  # similarities of the latest session
    python3 code/hwtest/faceshots.py compare --embedders mbf r50 r50_int8 --lfw DIR   # which embedder?
    python3 code/hwtest/faceshots.py session --level 0.3     # hands-free, in Turkish: waits for "hazırım",
                                                             # runs guided --spot --hdr, reads the results out

guided: at 1, 2 and 3 m the robot finds your face, centres its head on it, and takes shots (straight,
turned a little left, turned a little right). Each shot is two pictures of the same moment: the video stream
frame alive.py uses (2304x1296) and a full-resolution still (4608x2592) focused on the face (autofocus window).
--spot meters the exposure in the middle of the picture only - the head centres the face there, so the face
is exposed right even with ceiling lamps behind it (seen from the floor, people are usually backlit);
--hdr switches on the sensor's own HDR. Session folder: ~/scratch/hwtest/faceshots/<time>[-spot][-hdr]/ with <station>-<pose>-<n>-main.jpg / -full.jpg + shots.json.

report: embeds every face (the alive.py way: crop, SCRFD landmarks, align, MobileFaceNet), enrols from the
1 m straight shots, and prints the similarity per distance / pose for main vs full - plus against the person
alive.py already stored (config.PEOPLE_DIR). For scale (LFW): different people scored at most 0.29, the same
person >= 0.47 in 95 % of pairs; alive.py greets above FACE_MATCH_SIM.
compare: the same aligned faces through several embedders. Gallery = the straight shots of the latest -spot-hdr
session; probes = its turned shots + every shot of the other sessions; strangers = one LFW photo each of
--strangers people. Threshold per embedder = the score only STRANGER_RATE of strangers reach, then: how many of
the probes pass (scores of different models are not comparable - this is). Runs on the Mac too (--shots,
--models; copy the session folders over).
session: the robot speaks Turkish (espeak-ng) and listens (faster-whisper, hwtest.common.Listener): "hazırım"
starts a session, afterwards "tekrar" runs another one, "bitti" ends. Log: ~/scratch/hwtest/faceshots/session.log.
Needs the camera and the head: stop alive.py / drive_web.py / body_mcp.py first.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot import camera as rcam, config, faces           # noqa: E402
from hwtest.common import LOG_DIR, Listener, heard_any, say   # noqa: E402

OUT = LOG_DIR / "faceshots"
PENDING = Path.home() / "scratch" / "faceshots.pending"   # exists while a session should (re)start at boot
FULL = (4608, 2592)
STATIONS = (1, 2, 3)                                      # metres
POSES = (("straight", 3), ("left", 2), ("right", 2))
VOICE = {"en": ("en-gb", 150), "tr": ("tr", 140)}         # espeak-ng voice, words per minute
LINES = {
    "en": dict(hello="Hello. I am Arcby. Let us take some photos of your face. Please follow my voice.",
               stand="Please stand {d} {unit} in front of me, and look at my camera.",
               lost="I cannot see your face. Please step into my view.",
               skip="I still cannot see you. Let us skip this one.",
               straight="Found you. Hold still, and look at my camera.",
               left="Now turn your head a little to your left.",
               right="And now a little to your right.",
               done="All done. Thank you."),
    "tr": dict(hello="Merhaba, ben Arkbi. Şimdi yüzünün fotoğraflarını çekeceğim. Lütfen sesimi takip et.",
               stand="Lütfen benden {d} metre uzakta dur, ve kameraya bak.",
               lost="Yüzünü göremiyorum. Lütfen görüş alanıma gel.",
               skip="Seni hâlâ göremiyorum. Bunu atlıyoruz.",
               straight="Seni buldum. Kıpırdama, ve kameraya bak.",
               left="Şimdi başını biraz sola çevir.",
               right="Şimdi de biraz sağa çevir.",
               done="Fotoğraflar bitti. Teşekkürler."),
}
NUMBER = {"en": {1: "1", 2: "2", 3: "3"}, "tr": {1: "bir", 2: "iki", 3: "üç"}}


# ---------- guided session (on the robot) ----------

class Rig:
    """Camera (video mode + full-resolution stills) and head, for the session."""

    def __init__(self, level, spot=False, hdr=False, lang="en"):
        from libcamera import controls
        from picamera2 import Picamera2
        from robot.i2c import I2CDevice
        from robot.pan_tilt import PanTilt
        self.controls, self.level, self.hdr, self.lang = controls, level, hdr, lang
        self.head = PanTilt(I2CDevice(config.I2C_BUS, config.PTZ_ADDR))
        self.head.init()
        self.head.move(0, 15)
        if not rcam.set_sensor_hdr(hdr) and hdr:
            raise RuntimeError("could not switch the sensor HDR on")
        self.cam = Picamera2()
        self.video = self.cam.create_video_configuration(
            main={"size": config.CAMERA_MAIN, "format": "BGR888"},
            lores={"size": config.CAMERA_LORES, "format": "YUV420"},
            sensor={"output_size": config.CAMERA_MAIN}, buffer_count=4)
        self.still = self.cam.create_still_configuration(main={"size": FULL, "format": "BGR888"}, buffer_count=1)
        self.cam.configure(self.video)
        self.cam.start()
        if spot:
            self.cam.set_controls({"AeMeteringMode": controls.AeMeteringModeEnum.Spot})
        self._continuous_af()
        self.det = faces.load_detector()
        time.sleep(1.5)

    def say(self, key, **kw):
        text = LINES[self.lang][key].format(**kw)
        print("says:", text, flush=True)
        voice, speed = VOICE[self.lang]
        say(text, level=self.level, voice=voice, speed=speed)

    def _continuous_af(self):
        c = self.controls
        self.cam.set_controls({"AfMode": c.AfModeEnum.Continuous, "AfMetering": c.AfMeteringEnum.Auto})

    def look(self):
        """(lores RGB, main RGB) of one video frame."""
        req = self.cam.capture_request()
        try:
            lores = rcam.yuv420_to_rgb(req.make_array("lores"), *config.CAMERA_LORES)
            main = np.ascontiguousarray(req.make_array("main")[:, :, :3])
        finally:
            req.release()
        return lores, main

    def find_face(self, seconds):
        """The biggest face within `seconds` (lores pixels), or None."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            lores, main = self.look()
            found = self.det.detect(lores)
            if found:
                return max(found, key=lambda f: f.width), lores, main
        return None

    def centre_on(self, face):
        w, h = config.CAMERA_LORES
        pan, tilt = rcam.head_target(*face.centre, w, h, (self.head.pan, self.head.tilt))
        pan = min(max(pan, config.HEAD_PAN_SOFT[0]), config.HEAD_PAN_SOFT[1])
        tilt = min(max(tilt, config.HEAD_TILT_SOFT[0]), config.HEAD_TILT_SOFT[1])
        self.head.move(pan, tilt)
        time.sleep(0.4)                                     # servos settle, no blur

    def full_still(self, face):
        """A 4608x2592 still with autofocus on the face. Returns (RGB, info)."""
        c = self.controls
        s = FULL[0] / config.CAMERA_LORES[0]
        x1, y1, x2, y2 = face.box
        pad = 0.3 * (x2 - x1)
        win = (int(max(0, (x1 - pad) * s)), int(max(0, (y1 - pad) * s)),
               int((x2 - x1 + 2 * pad) * s), int((y2 - y1 + 2 * pad) * s))
        self.cam.switch_mode(self.still)
        try:
            self.cam.set_controls({"AfMode": c.AfModeEnum.Auto, "AfMetering": c.AfMeteringEnum.Windows,
                                   "AfWindows": [win]})
            focused = bool(self.cam.autofocus_cycle())
            req = self.cam.capture_request()
            try:
                rgb = np.ascontiguousarray(req.make_array("main")[:, :, :3])
                md = req.get_metadata()
            finally:
                req.release()
        finally:
            self.cam.switch_mode(self.video)
            self._continuous_af()
        return rgb, dict(af_window=win, focused=focused, lens=md.get("LensPosition"),
                         exposure_us=md.get("ExposureTime"), gain=md.get("AnalogueGain"))

    def close(self):
        try:
            self.cam.stop()
            self.cam.close()
        finally:
            if self.hdr:
                rcam.set_sensor_hdr(False)
            self.head.move(0, 0)
            self.head.dev.close()


def cmd_guided(args):
    run_guided(args.level, args.spot, args.hdr, args.walk, args.lang)


def run_guided(level, spot, hdr, walk, lang="en"):
    """One guided session. Returns its folder."""
    out = OUT / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-spot" * spot + "-hdr" * hdr)
    out.mkdir(parents=True)
    rig = Rig(level, spot=spot, hdr=hdr, lang=lang)
    shots = []
    try:
        rig.say("hello")
        for d in STATIONS:
            rig.say("stand", d=NUMBER[lang][d], unit="metre" if d == 1 else "metres")
            time.sleep(walk)
            hit = rig.find_face(12)
            if hit is None:
                rig.say("lost")
                hit = rig.find_face(12)
            if hit is None:
                rig.say("skip")
                continue
            rig.centre_on(hit[0])
            for pose, count in POSES:
                rig.say(pose)
                time.sleep(1.0)
                for n in range(count):
                    got = rig.find_face(3)
                    if got is None:
                        print(f"{d} m {pose} {n}: no face", flush=True)
                        continue
                    face, lores, main = got
                    full, info = rig.full_still(face)
                    name = f"{d}m-{pose}-{n}"
                    Image.fromarray(main).save(out / f"{name}-main.jpg", quality=95)
                    Image.fromarray(full).save(out / f"{name}-full.jpg", quality=95)
                    shots.append(dict(name=name, metres=d, pose=pose, box=[float(v) for v in face.box],
                                      width_lores=round(face.width, 1), head=[rig.head.pan, rig.head.tilt], **info))
                    print(f"{name}: face {face.width:.0f} px lores, focused {info['focused']}, "
                          f"lens {info['lens']}", flush=True)
                    (out / "shots.json").write_text(json.dumps(shots, indent=1))
        rig.say("done")
    finally:
        rig.close()
    print(f"{len(shots)} shots in {out}")
    return out


# ---------- report ----------

def embed_face(img, box_lores, refiner, embedder):
    """The alive.py way: crop around the box, SCRFD landmarks on the crop, align, embed. (vector, face px)"""
    aligned, px = align_face(img, box_lores, refiner)
    return (None if aligned is None else embedder.embed(aligned)), px


def align_face(img, box_lores, refiner):
    """Crop around the box, SCRFD landmarks on the crop, align: (112 x 112 face or None, face px)."""
    s = img.shape[1] / config.CAMERA_LORES[0]
    x1, y1, x2, y2 = box_lores
    w = x2 - x1
    cx1, cy1 = int(max(0, (x1 - 0.5 * w) * s)), int(max(0, (y1 - 0.5 * w) * s))
    crop = img[cy1:int((y2 + 0.5 * w) * s) + 1, cx1:int((x2 + 0.5 * w) * s) + 1]
    found = refiner.detect(crop)
    if not found:
        return None, w * s
    mid = np.array([crop.shape[1] / 2, crop.shape[0] / 2])
    f = min(found, key=lambda f: float(np.hypot(*(np.array(f.centre) - mid))))
    return faces.align(crop, f.kps), f.width


def cmd_report(args):
    sessions = sorted(p for p in OUT.iterdir() if (p / "shots.json").exists()) if OUT.exists() else []
    if not sessions:
        sys.exit("no session yet - run guided first")
    report(OUT / args.session if args.session else sessions[-1])


def report(folder):
    """Write folder/report.md; returns {metres: (mean main vs enrol main, mean full vs enrol full)}."""
    shots = json.loads((folder / "shots.json").read_text())
    refiner = faces.load_detector("scrfd", size=(192, 192), threads=2, min_px=8)
    embedder = faces.load_embedder()
    from robot.people import PeopleStore
    store = PeopleStore(config.PEOPLE_DIR)
    vec = {}
    for sh in shots:
        for src in ("main", "full"):
            img = np.asarray(Image.open(folder / f"{sh['name']}-{src}.jpg").convert("RGB"))
            v, px = embed_face(img, sh["box"], refiner, embedder)
            sh[f"{src}_px"] = round(float(px))
            if v is not None:
                vec[(sh["name"], src)] = v
    lines = [f"# Face shots {folder.name}", "",
             "Similarity to the 1 m straight shots (mean of the best 2) and to the person alive.py stored.",
             f"LFW: different people <= 0.29, same >= 0.47 (95 %); FACE_MATCH_SIM = {config.FACE_MATCH_SIM}.", "",
             "| shot | face px main / full | main vs enrol main | full vs enrol full | full vs enrol main | "
             "stored match main / full |", "|---|---|---|---|---|---|"]
    def straight(src, d):
        return [vec[(sh["name"], src)] for sh in shots if sh["metres"] == d and sh["pose"] == "straight"
                and (sh["name"], src) in vec]
    # enrol from the closest station with landmarks found (dark, backlit 1 m shots may have none)
    enrol_m = {src: next((d for d in STATIONS if straight(src, d)), None) for src in ("main", "full")}
    gallery = {src: straight(src, d) if d else [] for src, d in enrol_m.items()}
    lines.insert(3, f"Enrolled from the straight shots at main {enrol_m['main']} m / full {enrol_m['full']} m. "
                    "'no lm' = no landmarks found (face too dark / small).")

    def sim(v, gal):
        if v is None or not gal:
            return None
        s = sorted((float(v @ g) for g in gal), reverse=True)
        return sum(s[:2]) / len(s[:2])

    def fmt(x):
        return "-" if x is None else f"{x:.2f}"

    rows = []
    for sh in shots:
        m, f = vec.get((sh["name"], "main")), vec.get((sh["name"], "full"))
        def probe(src):
            return not (sh["metres"] == enrol_m[src] and sh["pose"] == "straight")
        r = dict(name=sh["name"], mm=sim(m, gallery["main"]) if probe("main") else None,
                 ff=sim(f, gallery["full"]) if probe("full") else None,
                 fm=sim(f, gallery["main"]) if probe("main") else None,
                 store_m=store.match(m) if m is not None else (None, 0.0),
                 store_f=store.match(f) if f is not None else (None, 0.0))
        rows.append(r)
        def sm(t, v):
            return "no lm" if v is None else f"{t[1]:.2f}{'' if t[0] else ' (no match)'}"
        lm = "" if m is not None and f is not None else f" (no lm: {'main ' * (m is None)}{'full' * (f is None)})"
        lines.append(f"| {sh['name']}{lm} | {sh['main_px']} / {sh['full_px']} | {fmt(r['mm'])} | {fmt(r['ff'])} | "
                     f"{fmt(r['fm'])} | {sm(r['store_m'], m)} / {sm(r['store_f'], f)} |")
    means = {}
    for d in STATIONS:
        sel = [r for r in rows if r["name"].startswith(f"{d}m-")]
        if sel:
            mean = lambda k: np.mean([r[k] for r in sel if r[k] is not None] or [np.nan])   # noqa: E731
            lines.append(f"| **{d} m mean** | | {mean('mm'):.2f} | {mean('ff'):.2f} | {mean('fm'):.2f} | |")
            means[d] = (float(mean("mm")), float(mean("ff")))
    report = folder / "report.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {report}")
    return means


# ---------- compare embedders ----------

STRANGER_RATE = 0.01


def lfw_faces(lfw, n, det, skip=0):
    """One aligned face from each of n LFW people (sorted by name, after `skip` people)."""
    out = []
    for person in sorted(p for p in lfw.iterdir() if p.is_dir())[skip:]:
        rgb = np.asarray(Image.open(sorted(person.glob("*.jpg"))[0]).convert("RGB"))
        found = det.detect(rgb)
        if found:
            f = min(found, key=lambda f: (f.centre[0] - 125) ** 2 + (f.centre[1] - 125) ** 2 - f.width ** 2)
            out.append(faces.align(rgb, f.kps))
        if len(out) == n:
            break
    return out


def best2(v, gallery):
    s = np.sort(gallery @ v)[::-1][:2]
    return float(s.mean())


def cmd_compare(args):
    root = Path(args.shots).expanduser() if args.shots else OUT
    sessions = sorted(p for p in root.iterdir() if (p / "shots.json").exists())
    main_s = [p for p in sessions if p.name.endswith("-spot-hdr")][-1]
    refiner = faces.load_detector("scrfd", size=(192, 192), threads=2, min_px=8, models_dir=args.models)
    det = faces.load_detector("scrfd", size=(256, 256), threads=2, models_dir=args.models)
    gal, probes = [], []                                  # (aligned, label)
    for folder in sessions:
        for sh in json.loads((folder / "shots.json").read_text()):
            for src in ("main", "full"):
                img = np.asarray(Image.open(folder / f"{sh['name']}-{src}.jpg").convert("RGB"))
                a, _ = align_face(img, sh["box"], refiner)
                if a is None:
                    continue
                if folder == main_s and sh["pose"] == "straight" and args.enrol_m in (None, sh["metres"]):
                    gal.append(a)
                else:
                    tag = "spot+hdr" if folder == main_s else "plain"
                    turn = "straight" if sh["pose"] == "straight" else "turned"
                    probes.append((a, f"{tag} {sh['metres']}m {turn} {src}"))
    strangers = lfw_faces(Path(args.lfw).expanduser(), args.strangers, det)
    print(f"gallery {len(gal)} faces ({main_s.name} straight), probes {len(probes)}, strangers {len(strangers)}")
    lines = [f"# Embedders on the robot's own face shots" + (f" (enrolled at {args.enrol_m} m only)" if args.enrol_m else ""), "",
             f"Gallery: {len(gal)} straight faces of {main_s.name}; probes: {len(probes)}; "
             f"strangers: {len(strangers)} LFW people. Threshold = score reached by {STRANGER_RATE:.0%} of strangers.",
             "", "| embedder | ms / face (here) | threshold | probes recognised | mean probe score | "
             "mean stranger score |", "|---|---|---|---|---|---|"]
    groups = sorted({lab.rsplit(" ", 1)[0] for _, lab in probes})
    per_group = {}
    for name in args.embedders:
        emb = faces.load_embedder(name, threads=args.threads, models_dir=args.models)
        emb.embed(gal[0])
        t = time.perf_counter()
        G = np.stack([emb.embed(a) for a in gal])
        ms = (time.perf_counter() - t) * 1000 / len(gal)
        P = [(best2(emb.embed(a), G), lab) for a, lab in probes]
        S = np.array([best2(emb.embed(a), G) for a in strangers])
        thr = float(np.quantile(S, 1 - STRANGER_RATE))
        ok = [p > thr for p, _ in P]
        lines.append(f"| {name} | {ms:.0f} | {thr:.2f} | {sum(ok)}/{len(ok)} = {np.mean(ok):.0%} | "
                     f"{np.mean([p for p, _ in P]):.2f} | {S.mean():.2f} |")
        per_group[name] = {g: np.mean([p > thr for p, lab in P if lab.rsplit(' ', 1)[0] == g]) for g in groups}
    lines += ["", "Recognised per kind of shot (main + full):", "",
              "| shots | " + " | ".join(args.embedders) + " |", "|---|" + "---|" * len(args.embedders)]
    for g in groups:
        n = sum(1 for _, lab in probes if lab.rsplit(" ", 1)[0] == g)
        lines.append(f"| {g} ({n}) | " + " | ".join(f"{per_group[e][g]:.0%}" for e in args.embedders) + " |")
    print("\n".join(lines))
    out = root / f"compare{f'-enrol{args.enrol_m}m' if args.enrol_m else ''}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out}")


# ---------- hands-free session in Turkish ----------

READY, AGAIN, DONE = ("hazirim", "hazir"), ("tekrar",), ("bitti", "bitir")
WORDS_PROMPT = "hazırım, tekrar, bitti"                   # nudges Whisper toward the words we expect


def spoken_results(means):
    """Turkish sentence with the results (similarities read as whole numbers: 0.31 -> 31)."""
    parts = []
    for d, (m, f) in sorted(means.items()):
        if d == min(means) and (np.isnan(m) or np.isnan(f)):
            continue                                       # the enrolment station has no score of its own
        num = lambda x: "yok" if np.isnan(x) else str(round(100 * x))   # noqa: E731
        parts.append(f"{NUMBER['tr'][d]} metrede, normal görüntü {num(m)}, keskin fotoğraf {num(f)}")
    if not parts:
        return "Sonuç çıkaramadım, yüzleri bulamadım."
    return "Sonuçlar. " + ". ".join(parts) + ". Kırk yedinin üstü, güvenle tanıyorum demek."


def cmd_session(args):
    log = open(OUT / "session.log", "a", buffering=1)

    def note(msg):
        log.write(f"{datetime.now():%H:%M:%S} {msg}\n")
        print(msg, flush=True)

    def speak(text):
        note(f"says: {text}")
        say(text, level=args.level, voice=VOICE["tr"][0], speed=VOICE["tr"][1])

    def wait_for(words_by_kind, remind, every_s, give_up_s):
        """Listen until one of the word groups is heard; returns its kind, or None after give_up_s."""
        t0 = last = time.monotonic()
        while time.monotonic() - t0 < give_up_s:
            text, rms = ear.hear(4, prompt=WORDS_PROMPT)
            if text:
                note(f"heard (rms {rms:.4f}): {text}")
            for kind, words in words_by_kind.items():
                if heard_any(text, words):
                    return kind
            if time.monotonic() - last >= every_s:
                speak(remind)
                last = time.monotonic()
        return None

    OUT.mkdir(parents=True, exist_ok=True)
    note(f"session start (pid {os.getpid()})")
    ear = Listener(args.model)
    try:
        speak("Merhaba, ben Arkbi. Beni başka odaya götürebilirsin. Hazır olduğunda, hazırım de.")
        go = wait_for({"ready": READY}, "Hazır olduğunda, hazırım de.", 45, args.wait_min * 60)
        while go:
            speak("Tamam, başlıyoruz.")
            folder = run_guided(args.level, True, True, args.walk, "tr")
            note(f"session folder {folder}")
            speak("Şimdi sonuçlara bakıyorum, biraz bekle.")
            means = report(folder)
            note(f"means {means}")
            speak(spoken_results(means))
            go = wait_for({"again": AGAIN, "done": DONE}, "Bir daha denemek istersen tekrar de. Bitirmek için bitti de.",
                          40, 10 * 60) == "again"
        speak("Tamam, görüşürüz.")
    finally:
        ear.close()
        PENDING.unlink(missing_ok=True)
        note("session end")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("guided")
    g.add_argument("--level", type=float, default=0.08, help="speech peak 0..1 (default quiet: 0.08)")
    g.add_argument("--walk", type=float, default=7.0, help="seconds to walk to each spot")
    g.add_argument("--spot", action="store_true", help="meter the exposure on the (centred) face")
    g.add_argument("--hdr", action="store_true", help="sensor HDR on")
    g.add_argument("--lang", choices=sorted(LINES), default="en")
    se = sub.add_parser("session")
    se.add_argument("--level", type=float, default=0.08, help="speech peak 0..1 (default quiet: 0.08)")
    se.add_argument("--walk", type=float, default=8.0, help="seconds to walk to each spot")
    se.add_argument("--model", default="tiny", help="faster-whisper model (tiny: 4/4 words, 2 s; base heard words in silence)")
    se.add_argument("--wait-min", type=float, default=30, help="give up waiting for 'hazırım' after this")
    c = sub.add_parser("compare")
    c.add_argument("--embedders", nargs="+", default=["mbf", "r50"])
    c.add_argument("--lfw", default=str(Path.home() / "scratch" / "datasets" / "lfw"))
    c.add_argument("--strangers", type=int, default=600)
    c.add_argument("--shots", help="folder with the session folders (default ~/scratch/hwtest/faceshots)")
    c.add_argument("--models", help="models folder (default config.MODELS_DIR)")
    c.add_argument("--threads", type=int, default=2)
    c.add_argument("--enrol-m", type=int, help="gallery from this distance only (a first meeting)")
    r = sub.add_parser("report")
    r.add_argument("--session", help="session folder name (default: the latest)")
    args = ap.parse_args()
    {"guided": cmd_guided, "report": cmd_report, "session": cmd_session, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
