#!/usr/bin/env python3
"""
Arcby, alive (phase A: eyes + head life). The camera finds faces, the robot learns who they are, attention
picks one to look at, and the head follows it softly with small living motions - no voice, no driving yet.

    python3 code/alive.py                       # on the robot (Ctrl+C stops; the head returns to the middle)
    python3 code/alive.py --collect             # ...and keep every face it sees (config.COLLECT_DIR, local only)
    python3 code/alive.py --no-head             # head without power (Pi on a wall adapter): it does not move
    python3 code/alive.py --quiet               # no voice (it greets people it knows by name otherwise)
    python3 code/alive.py --deaf                # no ears (otherwise: answers when called "Arkbi", asks new faces
                                                # their name - robot/conversation.py; Gemini speech-to-text)
    python3 code/alive.py --no-drive            # never open the motors (otherwise: with the battery on it can drive -
                                                # robot/legs.py; started from the dashboard's Drive card)
    python3 code/alive.py --sim PHOTOS_DIR      # no hardware: replay photos as the camera (Mac)
    python3 code/people_cli.py list             # who the robot knows; name / forget / merge them there

Records everything it senses into config.DATA_DIR (robot/recorder.py, robot/sensors.py: camera + vision per
frame, events, IMU, distance, Pi health, keyframes, face crops; no audio; `--no-record` turns it off; code/data_cli.py
lists it). Writes ~/scratch/alive/: alive.log (status every 5 s), events.jsonl (every event), view.jpg (what the
robot sees, boxes + names, every 2 s). People live in config.PEOPLE_DIR, models in config.MODELS_DIR.
Only one program may own the head and camera: stop drive_web.py / body_mcp.py / the dashboard camera first.
With the battery on alive.py also owns the motors, the IMU and the sonar (robot/legs.py): drive programs refuse to start
next to it. Every start of alive.py first logs how the previous run ended (the power breadcrumb, robot/breadcrumb.py).
"""
import argparse
import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import config, faces                           # noqa: E402
from robot.attention import Attention                     # noqa: E402
from robot.body import BodySense, body_summary, load_pose  # noqa: E402
from robot.camera import Background, FaceExposure         # noqa: E402
from robot.collect import Collector                       # noqa: E402
from robot.recorder import Recorder                       # noqa: E402
from robot.sensors import SensorLogger                    # noqa: E402
from robot.static_spots import StaticSpots                # noqa: E402
from robot.greeter import Greeter                         # noqa: E402
from robot.head_life import HeadLife                      # noqa: E402
from robot.people import PeopleStore                      # noqa: E402
from robot.vision import EmbedWorker, GazeWorker, Vision, track_summary   # noqa: E402

OUT = Path.home() / "scratch" / "alive"
EVENTS_MAX_BYTES = 20_000_000      # events.jsonl -> events.jsonl.1 above this (one old file kept)
EXIT_START_FAILED = 3              # could not start (camera busy, a model missing...): run_alive.sh waits longer
log = logging.getLogger("alive")


class Stopping(KeyboardInterrupt):
    """SIGTERM (shutdown, pkill) - stops as cleanly as Ctrl+C, also while still starting up."""


def quietly(what, fn, *a):
    """One clean-up step: a failure is logged, the others still run (and the exit stays clean)."""
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        log.warning("stopping: %s failed (%s)", what, e)


# ---------- a camera made of photos (--sim) ----------

class SimFrame:
    def __init__(self, img, t):
        self.t = t
        self._lores = np.asarray(img.resize(config.CAMERA_LORES, Image.BILINEAR))
        self._main = np.asarray(img.resize(config.CAMERA_MAIN, Image.BILINEAR))
        self.y_plane = (self._lores @ np.array([0.257, 0.504, 0.098]) + 16).astype(np.uint8)

    def lores_rgb(self):
        return self._lores

    def crop_main(self, box, pad=0.4):
        s = config.CAMERA_MAIN[0] / config.CAMERA_LORES[0]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        mx1, my1 = max(0, int((x1 - pad * w) * s)), max(0, int((y1 - pad * h) * s))
        mx2 = min(config.CAMERA_MAIN[0], int((x2 + pad * w) * s) + 1)
        my2 = min(config.CAMERA_MAIN[1], int((y2 + pad * h) * s) + 1)
        return np.ascontiguousarray(self._main[my1:my2, mx1:mx2]), (mx1, my1, s)


class SimCamera:
    """Each photo is 'seen' for frames_per_photo frames at CAMERA_FPS (real time), then the next one."""

    def __init__(self, paths, frames_per_photo):
        self.images = [Image.open(p).convert("RGB") for p in paths]
        self.frames_per_photo = frames_per_photo
        self.n = 0
        self.ev = 0.0
        self.next_t = time.monotonic()

    def start(self):
        return self

    @contextmanager
    def frame(self):
        self.next_t += 1.0 / config.CAMERA_FPS
        time.sleep(max(0.0, self.next_t - time.monotonic()))
        img = self.images[(self.n // self.frames_per_photo) % len(self.images)]
        self.n += 1
        yield SimFrame(img, time.monotonic())

    def set_ev(self, ev):
        self.ev = ev

    def close(self):
        pass


# ---------- output ----------

def save_view(frame, vision, store, path):
    img = Image.fromarray(frame.lores_rgb().copy())
    d = ImageDraw.Draw(img)
    for tr in vision.tracker.visible(frame.t):
        x1, y1, x2, y2 = tr.sighting.face.box
        if tr.person:
            name, col = store.name(tr.person) or f"#{tr.person[:4]}", "lime"
        elif not tr.is_face:
            name, col = f"not a face {tr.face_q:.2f}" if tr.face_q is not None else "unchecked", "grey"
        else:
            name, col = ("stranger", "orange") if tr.unknown else ("?", "yellow")
        d.rectangle((x1, y1, x2, y2), outline=col, width=2)
        label = f"{name} {tr.sim:.2f}" if tr.person else name
        d.text((x1, max(0, y1 - 11)), label + (" (o)" if tr.looking else ""), fill=col)
        d.text((x1, y2 + 2), f"eng {tr.engagement:.2f}", fill=col)
    tmp = str(path) + ".tmp"
    img.save(tmp, "JPEG", quality=80)
    os.replace(tmp, path)


def frame_record(f, vision, store, pose, head_dps, look, ev, target=None, bodies=None):
    """One line of the 'frames' stream: what the camera saw and what the robot made of it."""
    md = getattr(f, "metadata", {}) or {}
    pan, tilt, mode = look
    return dict(
        t=round(f.t, 4),
        cam={k: md.get(k) for k in ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux", "LensPosition",
                                     "ColourTemperature", "SensorTimestamp") if md.get(k) is not None},
        ev=ev, head=[round(pose[0], 2), round(pose[1], 2)], head_dps=round(head_dps, 1),
        attention=dict(mode=mode, pan=round(pan, 2), tilt=round(tilt, 2), target=target),
        detect_ms=round(vision.stats["detect_ms"], 1),
        faces=[dict(box=[round(v, 1) for v in d.box], score=round(float(d.score), 3),
                    kps=[[round(float(x), 1), round(float(y), 1)] for x, y in d.kps]) for d in vision.last_found],
        tracks=[dict(id=tr.id, person=tr.person, name=store.name(tr.person) if tr.person else None,
                     sim=round(tr.sim, 3), unknown=tr.unknown, is_face=tr.is_face,
                     face_q=None if tr.face_q is None else round(tr.face_q, 3),
                     engagement=round(tr.engagement, 3), gaze=tr.gaze_raw, looking=tr.looking, pan=round(tr.pan, 2),
                     tilt=round(tr.tilt, 2), size_deg=round(tr.sighting.size_deg, 2), hits=tr.hits)
                for tr in vision.tracker.visible(f.t)],
        **(body_record(bodies, f.t) if bodies is not None and bodies.enabled else {}))


def body_record(bodies, t):
    """The pose model's people when a new result came in since the last row (frame pixels: box, score, 17 keypoints
    x, y, visibility; else None) and the body tracks, for the 'frames' stream - the data to tune waving / face hints
    on."""
    fresh = bodies.stats["runs"] != getattr(bodies, "_recorded_runs", None)    # a new model result since the last row
    bodies._recorded_runs = bodies.stats["runs"]
    return dict(
        body_ms=round(bodies.stats["ms"], 1),
        bodies=None if not fresh else [dict(box=[round(float(v), 1) for v in b.box], score=round(b.score, 3),
                     kps=[[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in b.kps])
                for b in bodies.last_found],
        body_tracks=[dict(id=tr.id, pan=round(tr.pan, 2), tilt=round(tr.tilt, 2), h=round(tr.height_deg, 1),
                          hint=None if tr.hint is None else [round(tr.hint[0], 1), None if tr.hint[1] is None else
                                                             round(tr.hint[1], 1), tr.hint[2]],
                          face=tr.face, person=tr.person, still=tr.still, waved=tr.waved_t == tr.last_t,
                          given_up=tr.given_up)
                     for tr in bodies.tracker.visible(t)])


def _used(limit):
    try:
        return limit.used()[1]              # (day, requests)
    except Exception:  # noqa: BLE001
        return None


def live_state(t, fps, cam, vision, attention, mode_pan_tilt, pose, store, conv=None, voice=None, ears=None,
               mind=None, stt=None, sensors=None, rec=None, head=None, drive=None, drive_off=None):
    """Everything the robot knows right now, for the dashboard (~/scratch/alive/state.json, twice a second): who it
    sees (names, looking, where), whom its attention is on, the ears, the voice, the mind, the conversation,
    the sensors. Names only - never pictures or sound."""
    pan, tilt, mode = mode_pan_tilt
    tracks = vision.tracker.visible(t, faces_only=True)
    target = attention.target
    faces = []
    for tr in tracks:
        faces.append(dict(track=tr.id, person=tr.person, name=store.name(tr.person) if tr.person else None,
                          known=bool(tr.person), unknown=bool(tr.unknown), picture=bool(getattr(tr, "picture", False)),
                          looking=bool(tr.looking), engagement=round(tr.engagement, 2), sim=round(tr.sim, 2),
                          pan=round(tr.pan, 1), tilt=round(tr.tilt, 1), size_deg=round(tr.sighting.size_deg, 1),
                          target=tr.id == target))
    tgt = next((f for f in faces if f["target"]), None)
    partner = getattr(attention, "partner", None)
    out = dict(
        wall=round(time.time(), 2),
        camera=dict(fps=round(fps, 1), ev=round(getattr(cam, "ev", 0.0) or 0.0, 2),
                    detect_ms=round(vision.stats.get("detect_ms", 0)), embed_ms=round(vision.stats.get("embed_ms", 0))),
        head=dict(on=head is not None, pan=round(pose[0], 1), tilt=round(pose[1], 1),
                  dps=round(vision.stats.get("head_dps", 0), 1)),
        faces=faces,
        attention=dict(mode=mode, pan=round(pan, 1), tilt=round(tilt, 1), target=target,
                       target_name=(tgt["name"] or ("stranger" if tgt["unknown"] else "?")) if tgt else None,
                       partner=store.name(partner) if partner else None),
        people=[dict(id=pid, name=name, call=store.call(pid),
                     voice_prints=(conv.voices.count(pid) if conv is not None and conv.voices is not None else None))
                for pid, name, _ in store.people() if name],
    )
    if ears is not None:
        out["ears"] = dict(heard=ears.heard, barged=getattr(ears, "barged", 0),
                           level_db=round(20 * np.log10(ears.level + 1e-12), 1), gain=round(ears.agc.gain, 1),
                           muted=bool(voice.busy) if voice is not None else False,
                           hearing=ears.seg.buf is not None,
                           running=bool(ears.thread is not None and ears.thread.is_alive()),
                           stt=getattr(stt, "name", None), stt_today=_used(getattr(stt, "limit", None)),
                           live=getattr(getattr(conv, "live", None), "name", None))
    if voice is not None:
        day, n = voice.usage()
        out["voice"] = dict(busy=bool(voice.busy), speaking=bool(voice.speaking), model=voice.model,
                            voice=voice.voice, today=n, limit=voice.daily)
    if mind is not None:
        out["mind"] = dict(model=getattr(mind, "name", None), today=_used(getattr(mind, "limit", None)),
                           limit=config.MIND_DAILY_REQUESTS)
    if conv is not None:
        now = time.monotonic()
        sess, exp = conv.session, conv.expect
        out["conversation"] = dict(
            with_=store.name(sess["person"]) if sess and sess.get("person") and now <= sess["until"] else None,
            open_s=round(sess["until"] - now, 1) if sess and now <= sess["until"] else 0,
            waiting_for=exp.get("kind") if exp and now <= exp["until"] else None,
            dropped=conv.dropped)
    if sensors is not None:
        out["sensors"] = dict(status=dict(sensors.status), **{k: v for k, v in sensors.latest.items()})
    if rec is not None:
        out["recording"] = dict(dir=str(rec.dir), failed=bool(getattr(rec, "failed", False)))
    out["drive"] = drive.status() if drive is not None else dict(mode="off", why=drive_off or "no drive")
    return out


def write_state(path, state):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, default=str)
        os.replace(tmp, path)
    except OSError:
        pass


def make_hearing(ears, voice):
    """() -> None (nobody speaking, or the robot itself speaks) or seconds of quiet inside the sentence being heard,
    for the listening nods (attention mode "listen", robot/head_life.py). Uses Ears.pause_s when the ears have it,
    else reads their segmenter (an utterance under way = seg.buf; its quiet chunks = seg.quiet). None without ears."""
    if ears is None:
        return None
    from robot import mic

    def hearing():
        if voice is not None and getattr(voice, "busy", False):
            return None                         # never nod along to its own voice
        if hasattr(type(ears), "pause_s"):
            return ears.pause_s
        seg = getattr(ears, "seg", None)
        if seg is None or getattr(seg, "buf", None) is None:
            return None
        return getattr(seg, "quiet", 0) * getattr(mic, "CHUNK_S", 0.032)
    return hearing


def merge_voices(conv, people_dir, keep, drop):
    """vision merged two faces into one person: their voice prints too (else the voice still names the dropped id
    and the voice-vs-face check takes them for someone else). The running conversation's store when there is one
    (it would write its own copy back over the file)."""
    try:
        voices = conv.voices if conv is not None and conv.voices is not None else None
        if voices is None:
            from robot.voices import VoiceStore
            voices = VoiceStore(people_dir)
        voices.merge(keep, drop)
        log.info("voices: %s merged into %s", drop, keep)
    except Exception:  # noqa: BLE001 - the voice prints must never stop the robot
        log.exception("voices: merging %s into %s failed", drop, keep)


def open_legs(life, rec, stop, r):
    """The battery is on: motors + IMU + safety (robot.main.Robot), the sonar in its own thread, and the drive
    (robot/legs.py) stepping in its own thread. Each part goes into r as soon as it exists (shutdown closes it)."""
    from robot.drive_body import SonarThread
    from robot.legs import Legs
    from robot.main import Robot
    from robot.sonar import Sonar
    from robot import power
    power.require_battery("the drive (motors)", power.battery_on)     # (looked up now: tests replace it)
    robot = Robot(breadcrumb=False)                 # alive.py writes the breadcrumb itself (with the drive's PWM)
    r["drive_robot"] = robot.__enter__()            # the IMU calibrates for ~2 s from here: the robot stands still
    sonar = r["sonar"] = Sonar()
    st = r["sonar_thread_obj"] = SonarThread(sonar)

    def sonar_loop():
        try:
            st.run(stop)
        except Exception:  # noqa: BLE001 - no fresh echo = the drive does not move (sonar_gap 0)
            log.exception("drive: the sonar thread crashed - no driving until alive.py restarts")
    r["sonar_thread"] = threading.Thread(target=sonar_loop, name="sonar", daemon=True)
    r["sonar_thread"].start()
    legs = r["legs"] = Legs(robot, st, head_now=lambda: life.out, rec=rec)
    r["drive"] = legs                               # the breadcrumb's PWM
    threading.Thread(target=legs.run, args=(stop,), name="legs", daemon=True).start()
    return legs


def drive_signs(legs, follow, body_events, voice=None, store=None):
    """Hand signs (robot/gestures.py via robot/body.py) -> the drive: a stop sign stops it, "gel gel" starts following
    the one who beckoned (robot/legs.py follow mode). Returns the FollowTarget (or None)."""
    from robot.legs import FollowTarget
    for e in body_events:
        if e.get("kind") == "stop_sign" and legs.active:
            legs.stop("stop sign")
            follow = None
        elif e.get("kind") == "beckoned":
            who = (store.name(e["person"]) if store is not None and e.get("person") else None) or "someone"
            ok, why = legs.start("follow", why=f"gel gel ({who})")
            if ok:
                follow = FollowTarget(e.get("body"), e.get("person"), e.get("pan"))
                if voice is not None:
                    voice.say("Geliyorum.")
            else:
                log.info("drive: %s beckoned, but %s", who, why)
    return follow


def load_models(models_dir):
    det = faces.load_detector(models_dir=models_dir)
    emb = faces.load_embedder(models_dir=models_dir)
    # landmarks + face quality on the big crop: keep weak answers too (0.3-0.6 = may vote, not enrol)
    refiner = faces.load_detector("scrfd", size=(192, 192), threads=1, min_px=8, min_score=0.25,
                                  models_dir=models_dir)
    # big-stream landmarks for gaze, ~4x a second per face: SCRFD at a smaller input (cheap)
    gaze_refiner = faces.load_detector("scrfd", size=(128, 128), threads=1, min_px=8, min_score=0.25,
                                       models_dir=models_dir)
    return det, emb, refiner, gaze_refiner


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", metavar="PHOTOS_DIR", help="replay these photos as the camera (no hardware)")
    ap.add_argument("--frames-per-photo", type=int, default=45, help="--sim: frames each photo is shown")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    ap.add_argument("--people", default=config.PEOPLE_DIR, help="people database folder")
    ap.add_argument("--models", default=None, help="models folder (default config.MODELS_DIR)")
    ap.add_argument("--out", default=str(OUT), help="log / view folder")
    ap.add_argument("--no-head", action="store_true", help="the head has no power: do not drive it")
    ap.add_argument("--quiet", action="store_true", help="no voice (no greetings, no conversation)")
    ap.add_argument("--deaf", action="store_true", help="no ears (no conversation)")
    ap.add_argument("--mute", action="store_true", help="ears + mind on, voice off: what it would say only in the log "
                                                         "and on the dashboard (no speaker needed, no TTS requests)")
    ap.add_argument("--no-body", action="store_true", help="faces only: no pose model (robot/body.py)")
    ap.add_argument("--no-learn", action="store_true", help="never start learning (code/learn.py) on wall power")
    ap.add_argument("--no-drive", action="store_true", help="never open the motors (robot/legs.py)")
    ap.add_argument("--greet-away", type=float, default=config.GREET_AWAY_S, metavar="S",
                    help="greet someone again after this many seconds out of view (tests: e.g. 60)")
    ap.add_argument("--no-record", action="store_true", help="do not record (config.DATA_DIR)")
    ap.add_argument("--record", action="store_true", help="record also with --sim (off there by default)")
    ap.add_argument("--data", default=config.DATA_DIR, metavar="DIR", help="where to record (config.DATA_DIR)")
    ap.add_argument("--collect", nargs="?", const=config.COLLECT_DIR, metavar="DIR",
                    help="keep the faces it sees here (when recording they go into the session anyway)")
    args = ap.parse_args()

    terms = []

    def stop_on_term(signum, frame):
        terms.append(signum)
        if len(terms) == 1:                 # a second one while cleaning up: ignored (cleanup must finish)
            raise Stopping
    signal.signal(signal.SIGTERM, stop_on_term)

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    # alive.log: 4 files of 5 MB at most; the terminal only when there is one (in the background - cron,
    # nohup - stdout would be one more file that grows forever)
    handlers = [logging.handlers.RotatingFileHandler(out / "alive.log", maxBytes=5_000_000, backupCount=3)]
    if sys.stderr.isatty():
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers)
    events_path = out / "events.jsonl"
    events_file = open(events_path, "a")
    r = dict(cam=None, head=None, vision=None, bodies=None, ears=None, conv=None, stt=None, mind=None, voice=None, rec=None,
             sensors=None, collector=None, store=None, stop=threading.Event(), head_thread=None, events_file=events_file,
             learn=None, show=None, crumb=None, legs=None, drive_robot=None, sonar=None, sonar_thread=None)
    try:
        code = run(args, out, r)
    except Stopping:
        log.info("alive: stopped by SIGTERM while starting" if r["head_thread"] is None else "alive: SIGTERM")
        code = 0
    except KeyboardInterrupt:
        code = 0
    except Exception:  # noqa: BLE001 - logged; run_alive.sh tries again later
        log.exception("alive: could not start" if r["head_thread"] is None else "alive: crashed")
        code = EXIT_START_FAILED if r["head_thread"] is None else 1
    finally:
        try:
            shutdown(r)
        except KeyboardInterrupt:           # Ctrl+C twice: give up on the rest of the clean-up
            pass
    return code


def shutdown(r):
    """Stop everything that was started - each step on its own (one failing must not skip the others)."""
    r["stop"].set()
    if r.get("session") is not None:
        quietly("session recorder", r["session"].close)
    if r.get("commands") is not None:
        quietly("dashboard commands", r["commands"].close)
    if r.get("learn") is not None:
        quietly("learning", lambda: r["learn"].stop("alive.py stops", wait=True))
    if r["head_thread"] is not None:
        r["head_thread"].join(timeout=1.0)
    if r.get("legs") is not None:
        quietly("drive", r["legs"].stop, "alive.py stops")
    if r.get("drive_robot") is not None:              # motors off, pins released, IMU closed
        quietly("motors", r["drive_robot"].__exit__, None, None, None)
    if r.get("sonar_thread") is not None:
        r["sonar_thread"].join(timeout=0.5)
    if r.get("sonar") is not None:
        quietly("sonar", r["sonar"].close)
    if r["vision"] is not None:
        quietly("vision", r["vision"].close)
    if r["bodies"] is not None:
        quietly("bodies", r["bodies"].close)
    if r.get("show") is not None:
        quietly("show", r["show"].close)
    if r["ears"] is not None:
        quietly("ears", r["ears"].close)
    if r["conv"] is not None:
        quietly("conversation", r["conv"].close)
    if r["stt"] is not None:
        quietly("speech-to-text", r["stt"].close)
    if r["mind"] is not None:
        quietly("mind", r["mind"].close)
    if r["voice"] is not None:
        quietly("voice", r["voice"].close, 10)            # let a greeting under way finish
    head = r["head"]
    if head is not None:                                  # the battery may be off by now: never an error exit
        if quietly("head check", head.answers):
            quietly("head to the middle", head.move, 0, 0)
        quietly("head board", head.dev.close)
    if r["cam"] is not None:
        quietly("camera", r["cam"].close)
    if r.get("crumb") is not None:
        quietly("breadcrumb", r["crumb"].close)       # the clean-stop line (after the head and motors are still)
    quietly("events file", r["events_file"].close)
    rec, sensors, collector = r["rec"], r["sensors"], r["collector"]
    if rec is not None:
        if sensors is not None:
            quietly("sensors", sensors.join)
        quietly("recorder", rec.write, "events", {"kind": "stopped", "frames": rec.counts.get("frames", 0),
                                                  "sensors": sensors.status if sensors else {}})
        quietly("recorder", rec.close)
        log.info("alive: recorded %s (%s, %d keyframes kept, %d unchanged skipped; faces %s)", rec.dir, rec.counts,
                 rec.keyframes, rec.keyframes_skipped,
                 f"{collector.saved} kept, {collector.skipped} alike skipped" if collector else "none")
    if r["store"] is not None:
        log.info("alive: stopped; people known: %s", [(p, n, k) for p, n, k in r["store"].people()])
    else:
        log.info("alive: stopped")


def run(args, out, r):
    """Start everything (each part goes into r as soon as it exists, so shutdown() can stop it), then live."""
    events_path = out / "events.jsonl"
    events_file = r["events_file"]
    stop = r["stop"]
    status = r.setdefault("status", {"look": None})   # what the robot does now, for the power breadcrumb
    if not args.sim:                        # first of all: how did the previous run end (a power cut leaves no log)?
        from robot.breadcrumb import Breadcrumb, boot_id, describe
        crumb = r["crumb"] = Breadcrumb("alive.py", pwm=lambda: (r.get("drive") and r["drive"].applied) or None,
                                        state=lambda: status["look"])
        prev = crumb.start(stop)
        log.log(logging.WARNING if prev and not prev.get("exit") else logging.INFO, "%s", describe(prev, boot_id()))
    # the hardware first: if the camera is busy (the dashboard's live view) or the head fails, stop before any
    # session folder, speaker or model is opened
    head = None
    if args.sim:
        paths = sorted(p for p in Path(args.sim).expanduser().iterdir() if p.suffix.lower() in (".jpg", ".png"))
        if not paths:
            raise SystemExit(f"no photos in {args.sim}")
        cam = r["cam"] = SimCamera(paths, args.frames_per_photo).start()
    else:
        from robot.camera import Camera
        cam = r["cam"] = Camera().start()
        if not args.no_head:
            from robot.i2c import I2CDevice
            from robot.pan_tilt import PanTilt
            dev = I2CDevice(config.I2C_BUS, config.PTZ_ADDR)
            if not PanTilt(dev).answers():      # read only: is the servo board powered (battery on)?
                dev.close()
                log.warning("power: battery off (the PTZ board does not answer) - no head, no motors; camera, "
                            "ears and voice run on the Pi's own power. Restart with the battery on for the head.")
            else:
                head = r["head"] = PanTilt(dev)
                head.init()
                head.move(0, 0)

    store = r["store"] = PeopleStore(args.people)
    det, emb, refiner, gaze_refiner = load_models(args.models)
    record = not args.no_record and (not args.sim or args.record)
    rec = None
    if record:
        try:
            rec = Recorder(args.data, meta=dict(program="alive.py", sim=bool(args.sim), embedder=config.FACE_EMBEDDER))
        except OSError as e:                # disk full / SD error: the robot runs without recording
            log.warning("recorder: could not start (%s) - running without recording", e)
    r["rec"] = rec
    if rec is not None:                     # face crops go into the session (the recorder keeps the size limit)
        collector = Collector(rec.faces_dir, max_bytes=config.COLLECT_MAX_BYTES,
                              state_path=rec.root / "collect_state.json")   # the day's balance counts
    else:
        collector = Collector(args.collect) if args.collect else None
    r["collector"] = collector
    spots = StaticSpots(config.STATIC_FILE if not args.sim else None)
    vision = r["vision"] = Vision(det, EmbedWorker(emb), store, refiner=refiner, collector=collector,
                                  gaze_refiner=GazeWorker(gaze_refiner), spots=spots,
                                  background=Background())     # photos on the wall: still relative to the room
    vision.mode = "nohead" if head is None else "head"     # static spots / picture places per head mode
    log.info("static spots known: %s", spots.spots())
    attention = Attention(has_head=head is not None)    # no head: no turns, no turn events
    pose_model = None if args.no_body or not config.BODY_ENABLED else load_pose(args.models)
    bodies = r["bodies"] = BodySense(pose_model, inline=bool(args.sim))
    show = None
    if bodies.enabled and config.SHOW_ENABLED:          # things held up to it (robot/show.py): needs the bodies
        from robot.show import ShowSense, load_objects
        objects = load_objects(args.models)
        show = r["show"] = ShowSense.inline_for_tests(objects) if (args.sim and objects) else ShowSense(objects)
        log.info("show: %s", "object model loaded (things held up are seen)" if objects else
                 "no object model - things are looked at only when asked (bak / bu ne)")
    log.info("bodies: %s", f"{config.BODY_INPUT[0]}x{config.BODY_INPUT[1]} pose model" if bodies.enabled else "off")
    exposure = FaceExposure()
    voice = greeter = None
    if args.mute and not args.quiet:
        from robot.tts import MuteVoice
        voice = r["voice"] = MuteVoice()
        greeter = Greeter(voice, store, away_s=args.greet_away)
        log.info("voice: MUTE - ears + mind on, replies only in the log / on the dashboard")
    elif not args.quiet:
        from robot.tts import Voice
        voice = r["voice"] = Voice()
        greeter = Greeter(voice, store, away_s=args.greet_away)
        log.info("voice: %s %s (%s), %d phrases to prepare", voice.voice, voice.model,
                 "Gemini key found" if voice.key else "no key: silent", greeter.prepare_async())
    conv = ears = stt = mind = None
    if voice is not None and not args.deaf and not args.sim:
        from robot.conversation import Conversation
        from robot.mic import Ears
        from robot.stt import load_stt
        from robot.vad import Vad
        from robot.mind import load_mind
        from robot.voices import VoiceStore, load_voice_embedder
        stt = r["stt"] = load_stt()
        mind = r["mind"] = load_mind()
        voice_emb = load_voice_embedder(models_dir=args.models)
        from robot.live_stt import load_live_stt
        live = load_live_stt()
        conv = r["conv"] = Conversation(voice, stt, store, mind, VoiceStore(args.people), voice_emb,
                                         save_dir=config.CONV_SAVE_DIR, live=live)
        conv.show = show                            # "bak" / "bu ne": the hands + a picture for the mind
        ears = r["ears"] = Ears(Vad(), muted=lambda: voice.busy, on_utterance=conv.heard)
        ears.listener = conv                        # the sentence streams to the live STT while it is said
        if os.environ.get("ARCBY_EARS_DEBUG"):      # kill -USR1 <pid>: the last 15 s the ears got -> scratch (debug)
            import collections
            ears.debug = collections.deque(maxlen=int(15 / 0.032))
            signal.signal(signal.SIGUSR1, lambda *a: log.info("ears debug: %.1f s -> %s", ears.dump(
                out / "ears-debug.wav"), out / "ears-debug.wav"))
        log.info("ears: %s%s, %d requests a day at most", stt.name, f" + {live.name} while speaking" if live else "",
                 config.STT_DAILY_REQUESTS)
        log.info("mind: %s", f"{mind.name}, {config.MIND_DAILY_REQUESTS} requests a day at most" if mind
                 else "none (no ANTHROPIC_API_KEY) - rule-based answers")
        log.info("voices: %s", voice_emb.name if voice_emb else "none - who speaks only from faces")

    if voice is not None and not args.sim and not args.mute:
        from robot.tts import SpeakerError, open_speaker
        # opened last, after the heavy loading: its own process (robot/speakerd.py) keeps the card running from
        # here on - one amplifier switch now, none between sentences
        try:
            open_speaker(log_path=out / "speaker.log")
        except SpeakerError as e:               # card busy (say.py, a hwtest): the voice tries again per sentence
            log.warning("speaker: %s - no voice until the card is free", e)
        if ears is not None and conv is not None and config.BARGE_IN:
            from robot.bargein import BargeIn, EchoRef
            from robot.tts import set_echo_ref
            echo = EchoRef()
            set_echo_ref(echo)                  # what the speaker plays, and when...
            ears.barge = BargeIn(echo, tune_file=config.BARGE_TUNE_FILE)   # ...against what the mic hears
            ears.on_barge = conv.barge_in
            conv.on_false_barge = ears.barge.false_alarm    # its own words heard back: that echo is louder
            from robot.bargein import save_snippet
            ears.on_barge_saved = lambda ring: threading.Thread(target=save_snippet, args=(ring, echo),
                                                                daemon=True).start()
            log.info("barge-in: on (someone talking over the robot stops it, %.0f dB over its echo; echo model: %s)",
                     config.BARGE_THRESHOLD_DB,
                     f"{ears.barge.tuned}, {ears.barge.false_alarms} false alarms learnt" if ears.barge.tuned
                     else "defaults - run code/hwtest/bargecal.py once for this speaker")
    hearing = make_hearing(ears, voice)         # listening nods: is someone speaking (and where are the pauses)?
    attention.hearing = hearing
    life = HeadLife(head, hearing=hearing)
    # a replayed photo (or a head without power) does not move: frames are taken "straight ahead". On the
    # robot the servos arrive HEAD_SERVO_LAG_S after a command; speed is the fastest move within ~one exposure
    lag = config.HEAD_SERVO_LAG_S
    still = args.sim or head is None
    pose_at = (lambda t: (0.0, 0.0)) if still else (lambda t: life.pose_at(t - lag))
    speed_at = None if still else (lambda t: life.speed_at(t - lag - 0.04, t - lag + 0.04))

    head_thread = r["head_thread"] = threading.Thread(target=life.run, args=(stop,), name="head", daemon=True)
    head_thread.start()
    legs, drive_off = None, None
    if args.sim or args.no_drive or not config.DRIVE_IN_ALIVE:
        drive_off = "turned off"
    elif head is None:
        drive_off = "battery off (or no head)"
    else:
        try:
            legs = open_legs(life, rec, stop, r)
            log.info("drive: motors, IMU and sonar open - ready to drive (dashboard Drive card)")
        except Exception as e:  # noqa: BLE001 - alive without legs, as before
            drive_off = f"could not open the motors: {e}"
            log.warning("drive: off - %s", drive_off)
    if legs is not None and conv is not None:
        conv.drive = legs                           # "dur" stops it, "Arcby gez" starts a wander
    sensor_kw = {}
    if legs is not None:                            # one reader per sensor: the drive owns the IMU and the sonar
        from robot.drive_body import SonarReader, StateImu
        sensor_kw = dict(imu=StateImu(r["drive_robot"].state), sonar=SonarReader(r["sonar_thread_obj"]))
    sensors = r["sensors"] = SensorLogger(rec, **sensor_kw).start(stop) if rec is not None else None
    if ears is not None:
        ears.start(stop)
    log.info("alive: %s, %d people known (%s), embedder %s%s", "sim" if args.sim else "robot", len(store.people()),
             args.people, config.FACE_EMBEDDER,
             f", recording into {rec.dir}" if rec else
             f", collecting into {collector.root} ({collector.used / 1e6:.0f} MB used)" if collector else "")
    if not args.sim:                                # the dashboard's People tab: name / merge / forget / relabel
        from robot.commands import CommandBox, PeopleCommands
        from robot.voices import VoiceStore as _VoiceStore
        voices_store = conv.voices if conv is not None and getattr(conv, "voices", None) is not None \
            else _VoiceStore(args.people)           # ONE voice store: two would overwrite each other's saves
        from robot.commands import DriveCommands
        # the Drive card's STOP must be quick: the box is looked at 4 times a second
        r["commands"] = CommandBox(DriveCommands(PeopleCommands(store, voices_store, emb), legs, drive_off),
                                   every_s=0.25).start()
    session = None
    if not args.sim:                                # guided test sessions (hwtest/showshots.py guide): frames + crops
        from robot.session_rec import SessionRecorder
        session = r["session"] = SessionRecorder()
        if show is not None:
            show.session = session
    learn = None
    if not args.sim and not args.no_learn and config.LEARN_ENABLED:
        from robot.learning import LearnSupervisor
        learn = r["learn"] = LearnSupervisor()      # learn.py while on the wall adapter (battery off)
    if config.TUNED:
        log.info("alive: learnt values in use: %s", config.TUNED)
    t_start = time.monotonic()
    next_status = next_view = next_state = t_start
    fps_now = 0.0
    last_frames_t = -1e9
    frames = ears_chunks = 0
    people_t = None
    follow = None                                   # robot/legs.FollowTarget while following someone who beckoned
    bored = None
    if legs is not None:
        from robot.legs import Boredom
        bored = Boredom(t=time.monotonic())         # wanders by day when nobody was around for BORED_EXPLORE_S
    while not stop.is_set():
        with cam.frame() as f:
            events = vision.process(f, pose_at, speed_at)
            faces_seen = vision.tracker.visible(f.t, faces_only=True)
            body_events = bodies.process(f, pose_at, vision.stats["head_dps"], faces_seen)
            for e in body_events:               # someone waved: look at them (the greeter answers below)
                attention.wave(f.t, e.get("track"), e.get("person"), e.get("pan"), e.get("tilt"))
            events += body_events
            if legs is not None:
                follow = drive_signs(legs, follow, body_events, voice, store)
                if bodies.last_t is not None and bodies.last_t != people_t:     # a new pose result: who is where
                    from robot.drive_body import people_in
                    from robot.legs import target_person
                    people_t = bodies.last_t
                    w, h = bodies.last_size
                    legs.see_people(people_in(bodies.last_found, w, h, *bodies.last_pose), people_t)
                    if follow is not None and legs.mode == "follow":
                        tr = follow.pick(bodies.tracker.visible(people_t), people_t)
                        if tr is not None and tr.last_t == people_t:
                            legs.follow_target(target_person(tr, (w, h), bodies.last_pose), people_t)
                if legs.active:                 # the floor check (head looking down at the floor while it drives)
                    legs.frame(f.lores_rgb(), f.t, *pose_at(f.t), speed_at(f.t) if speed_at else 0.0)
                drive_events = legs.drain()
                for e in drive_events:          # said out loud: why a drive it was asked for ended
                    if e["what"] == "stopped" and e.get("mode") == "follow" and e["why"] == "lost them" and voice:
                        voice.say("Seni kaybettim.")
                if legs.mode != "follow":
                    follow = None
                someone = any(not getattr(tr, "picture", False) for tr in faces_seen) or \
                    any(not tr.still for tr in bodies.tracker.visible(f.t))
                bored.update(f.t, someone, legs, drive_events)
                events += drive_events + legs.drain()      # a wander it just started / stopped
            for e in events:                    # two faces found to be one person: their voice prints too
                if e.get("kind") == "merged":
                    merge_voices(conv, args.people, e["keep"], e["drop"])
            ev = exposure.update(vision.main_face_luma(f, attention.target), f.t, getattr(f, "metadata", None))
            if ev is not None:
                cam.set_ev(ev)
            # only tracks confirmed as real faces (big-stream check): lamps / hands never pull the head
            if conv is not None:
                attention.partner = conv.partner()     # never bored of whom it is talking with
            if session is not None:
                session.frame(f)
            if show is not None:                   # something held up to it: the head goes there
                shown_events = show.process(f, pose_at, bodies.tracker.visible(f.t), faces_seen)
                events += shown_events
                if conv is not None:
                    for e in shown_events:
                        conv.shown(e)
            pan, tilt, mode = attention.update(faces_seen, f.t, bodies.seekable(f.t),
                                               shown=show.target(f.t) if show is not None else None)
            events += attention.events             # turned to a voice: face / last_seen / sweep
            attention.events = []
            drive_look = legs.head_target if legs is not None else None
            status["look"] = f"drive {legs.mode}/{legs.last_cmd.state}" if drive_look is not None else mode
            if drive_look is not None:              # driving: the drive points the head (floor / people)
                life.set_focus(*drive_look, "drive")
            else:
                life.set_focus(pan, tilt, "idle" if mode in ("idle", "hold", "search") else
                               "listen" if mode == "listen" else "follow")
            if greeter is not None:
                events += [dict(kind="said", text=t) for t in
                           greeter.update(vision.tracker.visible(f.t, faces_only=True), events, f.t)]
            if conv is not None:
                conv.update(vision.tracker.visible(f.t, faces_only=True), events, f.t)
                heard = conv.events()
                partner = conv.partner()
                for e in heard:                         # a voice / its name: the head turns to the speaker
                    if e["kind"] == "voice" and partner is not None and e.get("person") != partner:
                        continue                        # talking with Emre: Selim's word to Emre does not turn it
                    if e["kind"] in ("voice", "called"):
                        attention.hear(f.t, e.get("person"), name=e["kind"] == "called")
                events += heard
            if rec is not None:
                faces_now = vision.tracker.visible(f.t, faces_only=True)
                if vision.tracker.visible(f.t) or bodies.tracker.visible(f.t) or \
                        f.t - last_frames_t >= config.DATA_FRAMES_IDLE_S:
                    rec.write("frames", frame_record(f, vision, store, pose_at(f.t), vision.stats["head_dps"],
                                                     (pan, tilt, mode), ev if ev is not None else cam.ev,
                                                     attention.target, bodies))
                    last_frames_t = f.t
                rec.keyframe(f.lores_rgb(), f.t, busy=bool(faces_now),
                             who={tr.person or f"track{tr.id}" for tr in faces_now})
                for e in events:
                    rec.write("events", {**e, "t": round(f.t, 4)})
                rec.tick()
            for e in events:
                e["t"] = round(time.time(), 2)
                if e.get("person"):
                    e["name"] = store.name(e["person"])
                events_file.write(json.dumps(e) + "\n")
                log.info("event %s", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()})
            events_file.flush()
            if events and events_file.tell() > EVENTS_MAX_BYTES:
                events_file.close()
                os.replace(events_path, str(events_path) + ".1")
                events_file = r["events_file"] = open(events_path, "a")
            frames += 1
            now = time.monotonic()
            if now >= next_view:
                save_view(f, vision, store, out / "view.jpg")
                next_view = now + 2.0
            if now >= next_state:                  # the dashboard's live view
                try:
                    write_state(out / "state.json", live_state(
                        f.t, fps_now, cam, vision, attention, (pan, tilt, mode), pose_at(f.t), store, conv, voice,
                        ears, mind, stt, sensors, rec, head, legs, drive_off))
                except Exception:  # noqa: BLE001 - the dashboard's view must never stop the robot
                    log.exception("alive: live state failed")
                next_state = now + 0.5
            if now >= next_status:
                fps = frames / max(1e-6, now - (next_status - 5.0)) if next_status > t_start else 0
                fps_now = fps
                frames = 0
                st = vision.stats
                log.info("%.1f fps | detect %.0f ms, embed %.0f ms, body %.0f ms | head %.0f deg/s | EV %+.1f | %s | "
                         "bodies %s | look %s %+.0f %+.0f%s%s",
                         fps, st["detect_ms"], st["embed_ms"], bodies.stats["ms"], st["head_dps"], cam.ev,
                         track_summary(vision.tracker.visible(f.t), store),
                         body_summary(bodies.tracker.visible(f.t)), mode, pan, tilt,
                         f" | drive {legs.mode} {legs.last_cmd.state or ''}" if legs is not None and legs.active else "",
                         (f" | saved {collector.saved}" if collector else "")
                         + (f" | ears {ears.status()}, {(ears.chunks - ears_chunks) / 5.0:.0f} chunks/s"
                            f"{', MUTED' if voice.busy else ''}{'' if ears.thread.is_alive() else ', STOPPED'}"
                            if ears is not None and ears.thread is not None else ""))
                ears_chunks = ears.chunks if ears is not None else 0
                next_status = now + 5.0
                if learn is not None:
                    try:
                        learn.tick(now)
                    except Exception:  # noqa: BLE001 - learning must never stop the robot
                        log.exception("alive: learning supervisor failed")
        if args.seconds and time.monotonic() - t_start >= args.seconds:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
