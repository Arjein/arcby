#!/usr/bin/env python3
"""
Lip test: can the robot tell WHO is talking from the mouths it sees? (the simple way - no lip model: how much each
mouth moves while a voice is heard, against how much it moves in silence and whether it follows the loudness)

    python3 code/hwtest/mouthshots.py record --left Selim --right "Emre Bey"   # on the Pi, alive.py stopped: 4 min
    python3 code/hwtest/mouthshots.py analyse [DIR]      # on the Pi or the Mac (copy the folder): who won each sentence

record: two people sit ~1-2 m in front of the robot, side by side, both in the picture (--left = the one on the LEFT
as seen standing BEHIND the robot). The robot says who talks by name ("Şimdi Selim konuşsun") and switches every
PHASE_S seconds; the other one looks at the robot and stays quiet (the hard case: a face looking at it that does not
talk). Its own voice is in the recording too: those moments are left out of the analysis (meta.json "prompts"). Saved locally only, in
~/scratch/hwtest/mouthshots/<time>/: audio.wav (16 kHz), audio_t.npy (each 32 ms chunk's start, time.monotonic),
faces.npz (frame times, boxes, landmarks, 48x32 grey mouth crops from the big stream), meta.json (the schedule).
Nothing is sent anywhere. Faces + voices of the two people: delete the folder when the test is done.

analyse: speech = the mic's energy over the room's floor; per sentence each face gets
  ratio = its mouth movement during the sentence / its movement in silence (a talking mouth moves 2-5x more),
  corr  = how its movement follows the sentence's loudness (Pearson, -1..1);
the winner per score is compared with the schedule (the first 2 s after a switch are left out).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot import config                                  # noqa: E402

OUT = Path.home() / "scratch" / "hwtest" / "mouthshots"
PHASE_S = 30.0                  # who talks switches this often
SETTLE_S = 2.0                  # after a switch: not counted (the other may still finish a word)
MOUTH = (48, 32)                # mouth crop size (w, h), grey
CHUNK_S = 0.032                 # mic chunk (512 samples at 16 kHz)
MIN_FACE_PX = 16                # lores pixels: smaller faces are skipped (mouth ~5 px)


# ---------- recording (Pi) ----------

def mouth_box(kps, face_w):
    """Mouth corners (landmarks 3, 4) -> a box around the lips in the same pixels: 1.8 x the corner distance wide,
    1.2 x high (the lips open downwards: centred a little below the corners)."""
    a, b = np.asarray(kps[3], float), np.asarray(kps[4], float)
    d = max(float(np.linalg.norm(a - b)), 0.3 * face_w)
    cx, cy = (a + b) / 2
    return cx - 0.9 * d, cy - 0.45 * d, cx + 0.9 * d, cy + 0.75 * d


def grey_crop(rgb, size=MOUTH):
    from PIL import Image
    if rgb.size == 0:
        return np.zeros(size[::-1], np.uint8)
    return np.asarray(Image.fromarray(rgb).convert("L").resize(size, Image.BILINEAR), np.uint8)


class MicRecorder:
    """arecord like robot/mic.py Ears, every chunk kept with its start time (sample-count clock)."""

    def __init__(self, card=config.MIC_CARD):
        from robot.mic import Decimator
        self.card, self.down = card, Decimator()
        self.chunks, self.times = [], []
        self.proc = self.thread = None
        self.error = None

    def start(self, stop):
        self.proc = subprocess.Popen(["arecord", "-q", "-D", f"plughw:{self.card},0", "-c", "2", "-r", "48000",
                                      "-f", "S32_LE", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self._run, args=(stop,), name="mic", daemon=True)
        self.thread.start()

    def _run(self, stop):
        from robot.mic import to_16k_mono
        need, n, t0 = 1536 * 2 * 4, 0, None
        while not stop.is_set():
            try:
                raw = self.proc.stdout.read(need)
            except (OSError, ValueError):       # the pipe closed by close()
                raw = b""
            if not raw or len(raw) < need:
                if not stop.is_set():
                    self.error = "the microphone stream ended (card busy?)"
                return
            now = time.monotonic()
            if t0 is None or now - (t0 + n * CHUNK_S) > 0.5:       # (re)anchor: started / lost samples
                t0, n = now - CHUNK_S, 0
            self.times.append(t0 + n * CHUNK_S)
            n += 1
            self.chunks.append(to_16k_mono(raw, down=self.down))

    def close(self):
        """Stop arecord for sure: it can sit blocked writing into the pipe nobody reads any more and not react to
        SIGTERM (the first dry run, 2026-11-01) - the pipe is closed too, and it is killed if it still runs."""
        if self.proc is None:
            return
        self.proc.terminate()
        if self.thread is not None:
            self.thread.join(timeout=2)
        try:
            self.proc.stdout.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=3)


class Prompter:
    """Says who talks next with the robot's voice (robot/tts.py, cached first: plays at once), in its own thread so
    the camera loop never waits; keeps when the robot was speaking (its voice is in the mic: not a sentence)."""

    def __init__(self, phrases):
        from robot.tts import Voice
        self.voice = Voice()
        self.voice.prepare(list(phrases))
        self.spans, self._q = [], __import__("queue").Queue()
        self.thread = threading.Thread(target=self._run, name="prompter", daemon=True)
        self.thread.start()

    def say(self, text):
        self._q.put(text)

    def _run(self):
        while (text := self._q.get()) is not None:
            t0 = time.monotonic()
            try:
                self.voice.say_now(text)
            except Exception as e:  # noqa: BLE001 - the test goes on (the terminal says it too)
                print(f"  (speaker: {e})", flush=True)
            self.spans.append((t0, time.monotonic()))

    def close(self):
        self._q.put(None)
        self.thread.join(timeout=15)


def record(args):
    from robot.power import AliveRunning, require_no_alive
    try:
        require_no_alive()
    except AliveRunning as e:
        sys.exit(str(e))
    from robot.camera import Camera
    from robot.faces import load_detector
    from robot.vision import own_face
    det = load_detector()
    # the mouth corners from the BIG stream (SCRFD on the face's crop, as alive.py's gaze refiner): on the small
    # stream a face is ~26 px and its corners jump - the first test's mouth crops slid onto cheeks / chins
    refiner = load_detector("scrfd", size=(160, 160), threads=1, min_px=8, min_score=0.25)
    names = {"left": args.left, "right": args.right}
    lines = {side: f"Şimdi {name} konuşsun." for side, name in names.items()}
    start_line, end_line = "Başlıyoruz.", "Bitti, teşekkürler."
    print("preparing the robot's phrases...", flush=True)
    prompter = Prompter([start_line, end_line, *lines.values()])
    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    cam = Camera().start()
    mic = MicRecorder()
    mic.start(stop)
    frames, rows = [], []           # frame times; (frame index, box 4, kps 10, score)
    mouths = []
    print(f"\nRecording {args.seconds:.0f} s into {out}\n{args.left} on the LEFT, {args.right} on the RIGHT (as seen "
          f"standing BEHIND the robot), 1-2 m away.\nThe robot says who talks, switching every {PHASE_S:.0f} s; the "
          "other one looks at the robot and stays quiet.\n", flush=True)
    prompter.say(start_line)
    time.sleep(3)
    t_start = time.monotonic()
    first = args.first
    said = None
    try:
        while time.monotonic() - t_start < args.seconds and mic.error is None:
            with cam.frame() as f:
                phase = max(0, int((f.t - t_start) // PHASE_S))     # the first frame was taken just before
                who = first if phase % 2 == 0 else ("right" if first == "left" else "left")
                if who != said:
                    left = args.seconds - (f.t - t_start)
                    print(f"  >>> {names[who]} ({who}) talks now   [{left:4.0f} s left]", flush=True)
                    prompter.say(lines[who])
                    said = who
                faces = det.detect(f.lores_rgb())
                i = len(frames)
                frames.append(f.t)
                for face in faces:
                    if face.width < MIN_FACE_PX:
                        continue
                    big, (x0, y0, s) = f.crop_main(face.box, pad=0.3)
                    best = own_face(refiner.detect(big), np.asarray(face.box) * s - [x0, y0, x0, y0])
                    if best is not None:            # the mouth from the sharp crop
                        mx1, my1, mx2, my2 = (int(round(v)) for v in mouth_box(best.kps, best.width))
                        h, w = big.shape[:2]
                        crop = big[max(0, my1):min(h, my2), max(0, mx1):min(w, mx2)]
                    else:                           # not found there: the small stream's corners (marked)
                        crop, _ = f.crop_main(mouth_box(face.kps, face.width), pad=0.0)
                    rows.append([i, *face.box, *np.asarray(face.kps).ravel(), face.score, best is not None])
                    mouths.append(grey_crop(crop))
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        prompter.say(end_line)
        prompter.close()
        stop.set()
        cam.close()
        mic.close()
    if mic.error:
        print("MIC:", mic.error)
    pcm = np.concatenate(mic.chunks) if mic.chunks else np.zeros(0, np.float32)
    with wave.open(str(out / "audio.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    np.save(out / "audio_t.npy", np.asarray(mic.times))
    np.savez_compressed(out / "faces.npz", frame_t=np.asarray(frames), rows=np.asarray(rows, np.float32).reshape(-1, 17),
                        mouths=np.asarray(mouths, np.uint8).reshape(-1, MOUTH[1], MOUTH[0]))
    (out / "meta.json").write_text(json.dumps(dict(t_start=t_start, phase_s=PHASE_S, first=first,
                                                   lores=list(config.CAMERA_LORES), note=args.note,
                                                   names=names, prompts=prompter.spans)))
    fps = len(frames) / max(1e-6, frames[-1] - frames[0]) if len(frames) > 1 else 0
    print(f"\n{len(frames)} frames ({fps:.1f} fps), {len(rows)} faces, {len(pcm) / 16000:.0f} s of sound -> {out}\n"
          f"Next: python3 code/hwtest/mouthshots.py analyse {out}")


# ---------- analysis (anywhere) ----------

def tracks_of(frame_t, rows, max_gap_s=0.5):
    """Simple tracking: each face joins the track whose last box centre is nearest (within half a face width,
    seen within max_gap_s). -> track id per row."""
    ids, last, nxt = np.full(len(rows), -1), {}, 0      # track -> (t, cx, cy, w)
    for r in np.argsort(rows[:, 0], kind="stable"):
        t = frame_t[int(rows[r, 0])]
        x1, y1, x2, y2 = rows[r, 1:5]
        cx, cy, w = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1
        taken = {ids[k] for k in np.flatnonzero(rows[:, 0] == rows[r, 0]) if ids[k] >= 0}
        best, bd = None, 0.5 * w
        for tid, (lt, lx, ly, lw) in last.items():
            d = np.hypot(cx - lx, cy - ly)
            if tid not in taken and t - lt <= max_gap_s and d < bd:
                best, bd = tid, d
        if best is None:
            best, nxt = nxt, nxt + 1
        ids[r] = best
        last[best] = (t, cx, cy, w)
    return ids


def aligned_diff(a, b, shift=2):
    """mean |a - b| after the best shift of b by up to `shift` px (the crop jitters a pixel or two between frames:
    not lip movement), over the middle part both cover."""
    s = shift
    core = a[s:-s, s:-s]
    return min(float(np.abs(core - b[s + dy:b.shape[0] - s + dy, s + dx:b.shape[1] - s + dx]).mean())
               for dy in range(-s, s + 1) for dx in range(-s, s + 1))


def mouth_motion(frame_t, rows, mouths, ids, max_dt=0.15):
    """Per row: how much its mouth changed since the same track's previous frame (mean |difference| of the
    brightness-normalised crops, lined up first); NaN for a track's first frame, after a gap, or for a crop whose
    corners came only from the small stream (column 16 = 0, recordings with the big-stream refiner)."""
    m = np.full(len(rows), np.nan)
    prev = {}
    norm = mouths.astype(np.float32)
    norm = (norm - norm.mean(axis=(1, 2), keepdims=True)) / (norm.std(axis=(1, 2), keepdims=True) + 1e-3)
    for r in np.argsort(rows[:, 0], kind="stable"):
        t = frame_t[int(rows[r, 0])]
        p = prev.get(ids[r])
        refined = rows.shape[1] < 17 or (rows[r, 16] > 0 and (p is None or rows[p[1], 16] > 0))
        if p is not None and t - p[0] <= max_dt and refined:
            m[r] = aligned_diff(norm[r], norm[p[1]])
        prev[ids[r]] = (t, r)
    return m


def loudness(pcm, chunk_t):
    """log energy (dB) per 32 ms chunk, after the mic's high-pass (its sub-20 Hz drift is no speech)."""
    from robot.mic import HighPass
    x = HighPass()(pcm)
    n = min(len(chunk_t), len(x) // 512)
    e = (x[:n * 512].reshape(n, 512) ** 2).mean(1)
    return np.asarray(chunk_t[:n]), 10 * np.log10(e + 1e-12)


def sentences(t, db, over_db=12.0, gap_s=0.3, min_s=0.6):
    """Speech stretches: energy over the room's floor (20th percentile) + over_db, gaps < gap_s joined."""
    floor = np.percentile(db, 20)
    on = db > floor + over_db
    out, start, last = [], None, None
    for ti, o in zip(t, on):
        if o:
            if start is None:
                start = ti
            elif ti - last > gap_s:
                out.append((start, last + CHUNK_S))
                start = ti
            last = ti
    if start is not None:
        out.append((start, last + CHUNK_S))
    return [(a, b) for a, b in out if b - a >= min_s], floor


def truth(meta, a, b):
    """Who the schedule says talks over the whole stretch a..b (None: a switch inside it, or just after one)."""
    p0 = int((a - meta["t_start"]) // meta["phase_s"])
    p1 = int((b - meta["t_start"]) // meta["phase_s"])
    if p0 != p1 or a - (meta["t_start"] + p0 * meta["phase_s"]) < SETTLE_S or p0 < 0:
        return None
    other = "right" if meta["first"] == "left" else "left"
    return meta["first"] if p0 % 2 == 0 else other


def analyse_arrays(frame_t, rows, mouths, pcm, chunk_t, meta, log=print):
    ids = tracks_of(frame_t, rows)
    motion = mouth_motion(frame_t, rows, mouths, ids)
    rt = frame_t[rows[:, 0].astype(int)]
    at, db = loudness(pcm, chunk_t)
    sents, floor = sentences(at, db)
    speech_at = np.interp(rt, at, db) > floor + 12.0
    half = meta["lores"][0] / 2
    side = {tid: ("left" if np.median((rows[ids == tid, 1] + rows[ids == tid, 3]) / 2) < half else "right")
            for tid in np.unique(ids)}
    quiet = {tid: np.nanmedian(motion[(ids == tid) & ~speech_at]) for tid in side}
    results = []
    log(f"{len(sents)} sentences, {len(side)} tracks, room floor {floor:.0f} dB")
    log(f"{'time':>7} {'len':>4}  {'truth':5}  {'ratio winner':14} {'corr winner':14} scores (side: ratio / corr)")
    for a, b in sents:
        want = truth(meta, a, b)
        if any(a < p1 + 0.7 and b > p0 for p0, p1 in meta.get("prompts", [])):
            continue                        # the robot's own voice (and its echo) - nobody's mouth
        sel = (rt >= a) & (rt <= b) & ~np.isnan(motion)
        scores = {}
        for tid in side:
            s = sel & (ids == tid)
            if s.sum() < max(4, 0.4 * (b - a) * 12):       # not seen through most of it
                continue
            ratio = float(np.mean(motion[s]) / max(quiet[tid], 1e-6)) if np.isfinite(quiet[tid]) else np.nan
            loud = np.interp(rt[s], at, db)
            corr = float(np.corrcoef(motion[s], loud)[0, 1]) if np.std(motion[s]) > 0 and np.std(loud) > 0 else 0.0
            scores[tid] = (ratio, corr)
        if len(scores) < 2 or want is None:
            continue
        by_ratio = max(scores, key=lambda k: scores[k][0])
        by_corr = max(scores, key=lambda k: scores[k][1])
        results.append(dict(a=a, b=b, truth=want, ratio=side[by_ratio], corr=side[by_corr], scores={
            side[k]: v for k, v in scores.items()}))
        log(f"{a - meta['t_start']:7.1f} {b - a:4.1f}  {want:5}  {side[by_ratio]:5} {'ok' if side[by_ratio] == want else 'WRONG':8} "
            f"{side[by_corr]:5} {'ok' if side[by_corr] == want else 'WRONG':8} "
            + "  ".join(f"{side[k]} {v[0]:.2f}/{v[1]:+.2f}" for k, v in scores.items()))
    for key in ("ratio", "corr"):
        n = sum(r[key] == r["truth"] for r in results)
        log(f"{key:5}: {n} of {len(results)} sentences right" + (f" ({100 * n / len(results):.0f} %)" if results else ""))
    return results


def analyse(args):
    d = Path(args.dir) if args.dir else max(OUT.iterdir())
    z = np.load(d / "faces.npz")
    with wave.open(str(d / "audio.wav")) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    meta = json.loads((d / "meta.json").read_text())
    print(f"{d}: {len(z['frame_t'])} frames, {len(z['rows'])} faces, {len(pcm) / 16000:.0f} s of sound")
    analyse_arrays(z["frame_t"], z["rows"], z["mouths"], pcm, np.load(d / "audio_t.npy"), meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--left", required=True, help="the name the robot says for the person on the LEFT "
                   "(seen from behind the robot), e.g. Selim")
    r.add_argument("--right", required=True, help='the person on the RIGHT, e.g. "Emre Bey"')
    r.add_argument("--seconds", type=float, default=240.0)
    r.add_argument("--first", choices=("left", "right"), default="left", help="who talks first")
    r.add_argument("--note", default="", help="e.g. 'Selim left, Emre right, 1.5 m, lamp on'")
    a = sub.add_parser("analyse")
    a.add_argument("dir", nargs="?", help="a recording folder (default: the newest)")
    args = ap.parse_args(argv)
    (record if args.cmd == "record" else analyse)(args)


if __name__ == "__main__":
    main()
