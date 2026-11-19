#!/usr/bin/env python3
"""
Showcase without the head (camera, speaker, microphone only - all on the Pi's own power): the robot recognises
people it knows and greets them on eye contact, and MEETS a new face - asks the name (Turkish, faster-whisper),
confirms it, stores it and greets them by name from then on.

    python3 code/demo.py --minutes 15          # stop alive.py first (only one program may own the camera)

Log: ~/scratch/alive/demo.log. A first step of the "mind" (meeting new people), without Claude yet.
"""
import argparse
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import config                                  # noqa: E402
from robot.camera import Camera, FaceExposure             # noqa: E402
from robot.greeter import Greeter                         # noqa: E402
from robot.people import PeopleStore                      # noqa: E402
from robot.tts import Voice                               # noqa: E402
from robot.vision import EmbedWorker, Vision, track_summary   # noqa: E402
from hwtest.common import Listener, heard_any, plain_tr   # noqa: E402
from alive import load_models                             # noqa: E402

log = logging.getLogger("demo")
YES = ("evet", "dogru", "aynen", "tamam", "evt")
NO = ("hayir", "yanlis", "degil")
FILLER = {"benim", "ben", "adim", "adım", "ismim", "isim", "evet", "merhaba", "selam", "ad", "bu", "o", "şey", "sey",
          "efendim", "hayır", "hayir", "değil", "degil", "yanlış", "yanlis", "ımm", "imm", "hmm", "ee", "eee", "ıı",
          "aa", "yani", "tamam", "ne", "ki"}
TR_UPPER = str.maketrans("iıçğöşü", "İIÇĞÖŞÜ")


def extract_name(text):
    """'Benim adım ahmet.' / 'Adım Ahmet' / 'Ahmet' -> 'Ahmet' (None if no name-like word)."""
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşüâîû]+", text)
    low = [w.replace("I", "ı").replace("İ", "i").lower() for w in words]
    pick = None
    for key in ("adım", "adim", "ismim", "ben"):
        if key in low and low.index(key) + 1 < len(low):
            pick = low[low.index(key) + 1]
            break
    if pick is None:
        rest = [w for w in low if w not in FILLER and len(w) >= 2]
        pick = rest[0] if rest else None
    if not pick or pick in FILLER:
        return None
    return pick[0].translate(TR_UPPER).upper() + pick[1:]


class Meeting(threading.Thread):
    """One conversation with a new face: name, confirmation, stored."""

    def __init__(self, voice, ear, store, pid):
        super().__init__(name="meeting", daemon=True)
        self.voice, self.ear, self.store, self.pid = voice, ear, store, pid
        self.named = False

    def say(self, text):
        while self.voice.busy:                  # never over a greeting
            time.sleep(0.1)
        log.info("says: %s", text)
        self.voice.say_now(text)

    def hear(self, seconds):
        text, rms = self.ear.hear(seconds)
        log.info("heard (rms %.4f): %r", rms, text)
        return text

    def run(self):
        try:
            self.say("Merhaba! Ben Arcby. Seni daha önce görmedim. Adın ne?")
            for attempt in range(3):
                name = extract_name(self.hear(5))
                if name:
                    self.say(f"{name}, doğru mu?")
                    answer = self.hear(3)
                    if heard_any(answer, YES) and not heard_any(answer, NO):
                        self.store.set_name(self.pid, name)
                        self.named = True
                        log.info("met %s = %s", self.pid, name)
                        self.say(f"Memnun oldum {name}! Seni artık tanıyacağım.")
                        return
                if attempt < 2:
                    self.say("Pardon, anlayamadım. Adını bir daha söyler misin?")
            self.say("Tamam, sonra tekrar tanışırız.")
        except Exception:  # noqa: BLE001 - the demo goes on
            log.exception("meeting failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--greet-again", type=float, default=180, help="seconds out of view before greeting again")
    ap.add_argument("--look-s", type=float, default=config.GREET_LOOK_S,
                    help="eye contact needed before a greeting (0 = greet on sight: people far away)")
    ap.add_argument("--enrol-px", type=float, default=config.FACE_ENROL_MIN_PX,
                    help="smallest face (big-stream px) a new person is stored from")
    args = ap.parse_args()
    out = Path.home() / "scratch" / "alive"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "demo.log"), logging.StreamHandler()])

    store = PeopleStore()
    det, emb, refiner, gaze_refiner = load_models(None)
    vision = Vision(det, EmbedWorker(emb), store, refiner=refiner, gaze_refiner=gaze_refiner,
                    enrol_min_px=args.enrol_px)
    voice = Voice()
    greeter = Greeter(voice, store, away_s=args.greet_again, look_s=args.look_s)
    greeter.last_seen = {}                      # a show: greet everyone once at the start
    ear = Listener("tiny")
    exposure = FaceExposure()
    cam = Camera().start()
    log.info("demo: %d people known: %s", len(store.people()), [n for _, n, _ in store.people()])
    meeting, met, retry_at = None, set(), {}
    t_end = time.monotonic() + args.minutes * 60
    next_status = 0.0
    try:
        while time.monotonic() < t_end:
            if meeting is not None and meeting.is_alive():
                time.sleep(0.2)                 # listening: the CPU goes to speech recognition, not to the camera
                continue
            with cam.frame() as f:
                events = vision.process(f, lambda t: (0.0, 0.0))
                ev = exposure.update(vision.main_face_luma(f), f.t)
                if ev is not None:
                    cam.set_ev(ev)
                faces_now = vision.tracker.visible(f.t, faces_only=True)
                if meeting is not None and not meeting.is_alive() and not meeting.named and meeting.pid in met:
                    met.discard(meeting.pid)        # no name yet: ask again a little later
                    retry_at[meeting.pid] = time.monotonic() + 20
                busy = meeting is not None and meeting.is_alive()
                if not busy:
                    greeter.update(faces_now, events, f.t)
                    for e in events:            # a face it does not know (yet): meet them
                        pid = e.get("person")
                        new = e["kind"] == "enrolled" or (e["kind"] == "person_seen" and pid and not store.name(pid))
                        if new and pid not in met and time.monotonic() >= retry_at.get(pid, 0):
                            met.add(pid)
                            meeting = Meeting(voice, ear, store, pid)
                            meeting.start()
                            break
                for e in events:
                    if e["kind"] in ("enrolled", "person_seen", "merged", "said"):
                        log.info("event %s", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()})
                if time.monotonic() >= next_status:
                    log.info("%s", track_summary(vision.tracker.visible(f.t), store))
                    next_status = time.monotonic() + 3.0
    except KeyboardInterrupt:
        pass
    finally:
        if meeting is not None:
            meeting.join(timeout=15)
        voice.close(timeout=10)
        vision.close()
        cam.close()
        ear.close()
        log.info("demo: stopped; people known: %s", [(p, n) for p, n, _ in store.people()])


if __name__ == "__main__":
    main()
