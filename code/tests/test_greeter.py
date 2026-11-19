"""robot/greeter.py: who is greeted when (eye contact, after an absence, groups), with what (fake voice)."""
import random
import tempfile
import threading
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from robot import config
from robot.greeter import GROUP, PHRASES, WELCOME_BACK, Greeter, join_names, part_of_day
from robot.people import PeopleStore


class FakeVoice:
    def __init__(self):
        self.said, self.prepared = [], []

    def say(self, text, level=None):
        self.said.append(text)

    def prepare(self, texts):
        self.prepared += texts
        return ["gemini"] * len(texts)


class Seen:
    """A visible track: who, and since when it looks at the robot (None = not looking)."""

    def __init__(self, person, looking_since=None):
        self.person, self.looking_since = person, looking_since

    def looked_for(self, t):
        return 0.0 if self.looking_since is None else t - self.looking_since


def at(hour, minute=0, second=0):
    return datetime(2026, 11, 1, hour, minute, second, tzinfo=ZoneInfo(config.TTS_TIMEZONE)).timestamp()


def seen(*pids):
    return [dict(kind="person_seen", person=p) for p in pids]


class GreeterTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.now = [at(9)]
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        rng = np.random.default_rng(0)
        self.selim, self.meltem, self.anon = (self.store.enrol([rng.normal(size=8) / 3]) for _ in range(3))
        self.store.set_name(self.selim, "Selim")
        self.store.set_name(self.meltem, "Meltem")
        self.voice = FakeVoice()

    def tearDown(self):
        self.dir.cleanup()

    def greeter(self):
        return Greeter(self.voice, self.store, clock=lambda: self.now[0], rng=random.Random(0))

    def step(self, g, tracks, events=(), dt=0.1):
        """One frame dt seconds later (frame time = wall time here)."""
        self.now[0] += dt
        return g.update(tracks, list(events), self.now[0])

    def test_greeted_on_eye_contact_once(self):
        g = self.greeter()
        self.assertEqual(self.step(g, [Seen(self.selim)], seen(self.selim)), [])        # recognised, not looking
        t0 = self.now[0]
        self.assertEqual(self.step(g, [Seen(self.selim, t0)]), [])                   # just started looking
        said = []
        for _ in range(15):
            said += self.step(g, [Seen(self.selim, t0)])                             # looks for 1.5 s
        self.assertEqual(len(said), 1)
        self.assertIn(said[0], [p.format(name="Selim") for p in PHRASES["morning"][1]])
        for _ in range(600):                                                         # an hour, re-recognised
            said += self.step(g, [Seen(self.selim, t0)], seen(self.selim), dt=6.0)
        self.assertEqual(len(said), 1)

    def test_the_family_is_greeted_formally(self):
        self.store.set_call(self.meltem, "Meltem Hanım")
        phrases = self.greeter().phrases_for(self.meltem)
        self.assertTrue(all("Meltem Hanım" in p for p in phrases))
        self.assertFalse(any(w in p for p in phrases for w in ("uyandın", "hoş geldin.", "hoş geldin!", "Bu saatte")))
        self.assertTrue(any("hoş geldiniz" in p for p in phrases))
        self.assertIn("uyandın mı?", " ".join(self.greeter().phrases_for(self.selim)))     # Selim: informal

    def test_no_greeting_without_eye_contact(self):
        g = self.greeter()
        said = self.step(g, [Seen(self.selim)], seen(self.selim))
        for _ in range(50):
            said += self.step(g, [Seen(self.selim)])                                 # never looks at the robot
        for _ in range(12):
            said += self.step(g, [], dt=1.0)                                        # walks off (12 s)
        self.assertEqual(said, [])
        self.assertEqual(g.pending, {})

    def test_a_group_gets_one_sentence(self):
        g = self.greeter()
        self.now[0] = at(12)
        self.step(g, [Seen(self.selim), Seen(self.meltem)], seen(self.selim, self.meltem))
        t0 = self.now[0]
        said = []
        for i in range(30):                                  # Selim looks at once, Meltem 1.5 s later
            said += self.step(g, [Seen(self.selim, t0), Seen(self.meltem, t0 + 1.5 if i >= 15 else None)])
        self.assertEqual(said, ["Merhaba Selim ve Meltem!"])
        self.assertEqual(join_names(["Selim", "Meltem", "Emre"]), "Selim, Meltem ve Emre")

    def test_greets_again_after_being_away_and_welcome_back(self):
        g = self.greeter()
        self.step(g, [Seen(self.selim)], seen(self.selim))
        said = [x for _ in range(15) for x in self.step(g, [Seen(self.selim, self.now[0] - 5)])]
        self.assertEqual(len(said), 1)
        self.now[0] += 600                                   # 10 min away: still "here"
        self.step(g, [Seen(self.selim)], seen(self.selim))
        self.assertEqual([x for _ in range(15) for x in self.step(g, [Seen(self.selim, self.now[0] - 5)])], [])
        self.now[0] += 4 * 3600                              # 4 h away: welcome back
        self.step(g, [Seen(self.selim)], seen(self.selim))
        said = [x for _ in range(15) for x in self.step(g, [Seen(self.selim, self.now[0] - 5)])]
        self.assertEqual(said, [WELCOME_BACK[0].format(name="Selim")])

    def test_unnamed_people_are_not_greeted(self):
        g = self.greeter()
        self.step(g, [Seen(self.anon)], seen(self.anon))
        self.assertEqual([x for _ in range(20) for x in self.step(g, [Seen(self.anon, self.now[0] - 5)])], [])

    def test_a_restart_does_not_greet_who_was_just_here(self):
        self.store = PeopleStore(self.dir.name, clock=lambda: self.now[0])
        self.store.seen(self.selim)                           # seen by the previous run a moment ago
        self.now[0] += 60
        g = self.greeter()
        self.step(g, [Seen(self.selim)], seen(self.selim))
        self.assertEqual([x for _ in range(20) for x in self.step(g, [Seen(self.selim, self.now[0] - 5)])], [])

    def test_phrases_by_time_of_day(self):
        self.assertEqual([part_of_day(h) for h in (6, 12, 19, 23, 2)], ["morning", "day", "evening", "night", "night"])
        self.assertEqual(set(GROUP), set(PHRASES))

    def test_prepare_makes_every_named_persons_phrases(self):
        g = self.greeter()
        self.assertEqual((len(g.phrases_for(self.selim)), g.phrases_for(self.anon)), (11, []))
        n = g.prepare_async()
        for t in threading.enumerate():
            if t.name == "greeter-prepare":
                t.join(2)
        self.assertEqual((n, len(self.voice.prepared)), (24, 24))     # 2 people x 11 + 2 wave hellos for strangers


if __name__ == "__main__":
    unittest.main()
