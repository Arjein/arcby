"""
Greeter: say hello to people the robot knows when they come (back) - not every time it sees them.

    greeter = Greeter(voice, store)          # voice = robot/tts.Voice, store = robot/people.PeopleStore
    greeter.prepare_async()                  # make everyone's phrases ahead (cached: greetings play at once)
    greeter.update(tracks, events, t)        # every frame: visible tracks, vision events, frame time; returns what it said

A known, named person who was not seen for GREET_AWAY_S (or never yet) becomes "to greet" when vision decides
who they are (a "person_seen" event) - and is greeted once they LOOK at the robot for GREET_LOOK_S (humans greet
on eye contact; someone walking past with their phone is not greeted). People ready within GREET_GROUP_S are
greeted in one sentence ("Merhaba Selim ve Meltem!"). Someone sitting in front of the robot for an hour is greeted
once. "Last seen" starts from the people store, so a restart does not greet everyone again. Phrases depend on the
time of day, and after GREET_LONG_AWAY_S it is a welcome back. Unnamed people are not greeted (yet).

Waving (a "waved" event from robot/body.py) is answered at once by name when the waving body's face is known; when
its face is there but not recognised yet, after up to GREET_WAVE_NAME_S (by name if it is recognised by then), else a
plain hello - at most once per GREET_WAVE_AGAIN_S per person, never over the robot's own voice.
"""
import logging
import random
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger(__name__)

# time of day (from hour, to hour) -> phrases; {name} = how the person is addressed (PeopleStore.call)
PHRASES = {
    "morning": ((5, 11), ("Günaydın {name}!", "Günaydın {name}, uyandın mı?")),
    "day": ((11, 18), ("Merhaba {name}!", "Selam {name}, hoş geldin.")),
    "evening": ((18, 22), ("İyi akşamlar {name}!", "Akşam oldu {name}, hoş geldin.")),
    "night": ((22, 5), ("Merhaba {name}.", "Bu saatte mi {name}?")),
}
WELCOME_BACK = ("Tekrar hoş geldin {name}!",)
# for the people it addresses formally ("Meltem Hanım", "Halil Bey" - Selim, 2026-11-01: his family, with respect)
FORMAL = {
    "morning": ((5, 11), ("Günaydın {name}!", "Günaydın {name}, hoş geldiniz.")),
    "day": ((11, 18), ("Merhaba {name}!", "Hoş geldiniz {name}.")),
    "evening": ((18, 22), ("İyi akşamlar {name}!", "İyi akşamlar {name}, hoş geldiniz.")),
    "night": ((22, 5), ("Merhaba {name}.", "İyi geceler {name}.")),
}
FORMAL_WELCOME_BACK = ("Tekrar hoş geldiniz {name}!",)
WAVE = ("Selam {name}!", "Merhaba {name}!")             # someone waved at it
FORMAL_WAVE = ("Merhaba {name}!", "Merhabalar {name}!")
WAVE_STRANGER = ("Merhaba!", "Selam!")


def formal(call):
    return call.endswith((" Bey", " Hanım"))
GROUP = {"morning": ("Günaydın {name}!",), "day": ("Merhaba {name}!",), "evening": ("İyi akşamlar {name}!",),
         "night": ("Merhaba {name}.",)}
GROUP_WELCOME_BACK = ("Tekrar hoş geldiniz {name}!",)


def join_names(names):
    """["Selim", "Meltem", "Emre"] -> "Selim, Meltem ve Emre"."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " ve " + names[-1]


def part_of_day(hour):
    for part, ((start, end), _) in PHRASES.items():
        if (start <= hour < end) if start < end else (hour >= start or hour < end):
            return part
    return "day"


class Greeter:
    def __init__(self, voice, store, clock=time.time, rng=None, away_s=config.GREET_AWAY_S,
                 long_away_s=config.GREET_LONG_AWAY_S, look_s=config.GREET_LOOK_S, group_s=config.GREET_GROUP_S,
                 pending_s=config.GREET_PENDING_S, tz=config.TTS_TIMEZONE):
        self.voice, self.store, self._clock = voice, store, clock
        self.rng = rng or random.Random()
        self.away_s, self.long_away_s = away_s, long_away_s
        self.look_s, self.group_s, self.pending_s = look_s, group_s, pending_s
        self.pending = {}                   # person id -> {"away": s, "gone_since": wall time or None}
        self.group_until = None             # first person ready: wait this long for the others
        self.tz = ZoneInfo(tz)
        # last time each person was seen (wall clock): from the store, then from every frame
        self.last_seen = {pid: p.get("last_seen", 0.0) for pid, p in store.index.items()}
        self.waved_at = {}                  # person id (None = someone unknown) -> wall time of the last wave answer
        self.wave_wait = None               # (face track id, until wall time): a wave waiting for its face's name

    def phrases_for(self, pid):
        name = self.store.call(pid)
        if not name:
            return []
        phrases, back = (FORMAL, FORMAL_WELCOME_BACK) if formal(name) else (PHRASES, WELCOME_BACK)
        wave = FORMAL_WAVE if formal(name) else WAVE
        return [p.format(name=name) for _, (_, ps) in phrases.items() for p in ps] + \
            [p.format(name=name) for p in back + wave]

    def prepare_async(self):
        """Make every named person's phrases in the background (each is paid for once, then cached)."""
        texts = [t for pid in list(self.store.index) for t in self.phrases_for(pid)] + list(WAVE_STRANGER)

        def run():
            try:
                done = self.voice.prepare(texts)
                log.info("greeter: %d phrases ready (%s)", len(texts),
                         ", ".join(f"{s} {done.count(s)}" for s in sorted(set(done))))
            except Exception:  # noqa: BLE001 - greetings then just take a moment longer
                log.exception("greeter: preparing phrases failed")
        threading.Thread(target=run, name="greeter-prepare", daemon=True).start()
        return len(texts)

    def update(self, tracks, events, t):
        """Note who came back, greet the ones looking at the robot, note who is in view. Returns the texts.
        tracks = visible face tracks (robot/tracker.Track), t = their frame time."""
        now = self._clock()
        waved = self._waved(events, now, tracks)
        if waved:
            return waved
        for e in events:                    # came (back) after a while: to greet - once they look at the robot
            pid = e.get("person") if e.get("kind") == "person_seen" else None
            if pid is not None and pid not in self.pending and self.store.call(pid):
                away = now - self.last_seen.get(pid, 0.0)
                if away >= self.away_s:
                    self.pending[pid] = {"away": away, "known": self.last_seen.get(pid, 0.0) > 0, "gone_since": None}
        visible = {tr.person: tr for tr in tracks if tr.person is not None}
        for pid in visible:
            self.last_seen[pid] = now
        for pid, p in list(self.pending.items()):
            if pid in visible:
                p["gone_since"] = None
            elif p["gone_since"] is None:
                p["gone_since"] = now
            elif now - p["gone_since"] >= self.pending_s:
                del self.pending[pid]       # left without looking at the robot: no greeting
        ready = [pid for pid in self.pending if pid in visible and visible[pid].looked_for(t) >= self.look_s]
        if not ready:
            return []
        waiting = [pid for pid in self.pending if pid in visible and pid not in ready]
        if waiting:                         # someone else in view may look in a moment: one sentence for all
            if self.group_until is None:
                self.group_until = now + self.group_s
            if now < self.group_until:
                return []
        self.group_until = None
        text = self._phrase([(pid, self.pending.pop(pid)) for pid in ready])
        self.voice.say(text)
        log.info("greeter: %s", text)
        return [text]

    def _waved(self, events, now, tracks=()):
        """Answer a wave (first one in events that is due, or the one waiting for its face's name). Returns [text]
        or []."""
        if self.wave_wait is not None:
            track, until = self.wave_wait
            face = next((tr for tr in tracks if getattr(tr, "id", None) == track), None)
            pid = face.person if face is not None else None
            if pid is not None or now >= until or face is None:
                self.wave_wait = None
                events = [dict(kind="waved", person=pid, track=None)] + list(events)
        for e in events:
            if e.get("kind") != "waved":
                continue
            pid = e.get("person")
            call = self.store.call(pid) if pid else None
            key = pid if call else None
            if now - self.waved_at.get(key, -1e18) < config.GREET_WAVE_AGAIN_S or getattr(self.voice, "busy", False):
                continue
            if pid is None and e.get("track") is not None and self.wave_wait is None:
                self.wave_wait = (e["track"], now + config.GREET_WAVE_NAME_S)    # a face is on it: its name first
                continue
            self.waved_at[key] = now
            if call:
                text = self.rng.choice(FORMAL_WAVE if formal(call) else WAVE).format(name=call)
                self.pending.pop(pid, None)     # that was the hello: no second one on eye contact
                self.last_seen[pid] = now
            else:
                text = self.rng.choice(WAVE_STRANGER)
            self.voice.say(text)
            log.info("greeter: waved at - %s", text)
            return [text]
        return []

    def _phrase(self, people):
        """people = [(id, pending info)] -> one greeting for all of them."""
        names = [self.store.call(pid) for pid, _ in people]
        welcome_back = all(p["known"] and p["away"] >= self.long_away_s for _, p in people)
        hour = datetime.fromtimestamp(self._clock(), self.tz).hour
        if len(names) == 1 and formal(names[0]):
            pool = FORMAL_WELCOME_BACK if welcome_back else FORMAL[part_of_day(hour)][1]
        elif len(names) == 1:
            pool = WELCOME_BACK if welcome_back else PHRASES[part_of_day(hour)][1]
        else:
            pool = GROUP_WELCOME_BACK if welcome_back else GROUP[part_of_day(hour)]
        return self.rng.choice(pool).format(name=join_names(names))
