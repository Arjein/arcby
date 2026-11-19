"""alive.py's drive glue: the drive opens only with the battery on; hand signs start / stop it."""
import threading
import unittest
from unittest import mock

import alive
from robot import power


class FakeLegs:
    def __init__(self, active=False, ok=True):
        self.active, self.ok, self.calls = active, ok, []
        self.mode = "follow" if active else "stopped"

    def stop(self, why):
        self.calls.append(("stop", why))
        self.active = False
        return True

    def start(self, mode, why="", max_s=None):
        self.calls.append(("start", mode, why))
        if self.ok:
            self.active, self.mode = True, mode
        return (True, "") if self.ok else (False, "safety: tipped")


class FakeVoice:
    def __init__(self):
        self.said = []

    def say(self, text, **kw):
        self.said.append(text)


class OpenLegsTest(unittest.TestCase):
    def test_battery_off_no_drive(self):
        with mock.patch.object(power, "battery_on", return_value=False):
            r = {}
            with self.assertRaises(power.BatteryOff):
                alive.open_legs(None, None, threading.Event(), r)
        self.assertIsNone(r.get("drive_robot"))                 # nothing opened, nothing to close


class SignsTest(unittest.TestCase):
    def test_gel_gel_starts_following_the_one_who_beckoned(self):
        legs, voice = FakeLegs(), FakeVoice()
        follow = alive.drive_signs(legs, None, [dict(kind="beckoned", body=4, person=None, pan=12.0)], voice)
        self.assertEqual(legs.calls, [("start", "follow", "gel gel (someone)")])
        self.assertEqual((follow.body_id, follow.last[0]), (4, 12.0))
        self.assertEqual(voice.said, ["Geliyorum."])

    def test_a_refused_start_follows_nobody(self):
        legs = FakeLegs(ok=False)
        self.assertIsNone(alive.drive_signs(legs, None, [dict(kind="beckoned", body=4)]))

    def test_a_stop_sign_stops_a_drive(self):
        legs = FakeLegs(active=True)
        self.assertIsNone(alive.drive_signs(legs, object(), [dict(kind="stop_sign", body=2)]))
        self.assertEqual(legs.calls, [("stop", "stop sign")])

    def test_a_stop_sign_while_standing_does_nothing(self):
        legs = FakeLegs()
        alive.drive_signs(legs, None, [dict(kind="stop_sign", body=2), dict(kind="waved", body=2)])
        self.assertEqual(legs.calls, [])


class LiveStateTest(unittest.TestCase):
    def test_the_drive_card_says_why_there_is_no_drive(self):
        from types import SimpleNamespace
        vision = SimpleNamespace(tracker=SimpleNamespace(visible=lambda t, faces_only=False: []), stats={})
        attention = SimpleNamespace(target=None)
        store = SimpleNamespace(people=lambda: [], name=lambda p: None, call=lambda p: None)
        st = alive.live_state(0.0, 10.0, SimpleNamespace(ev=0.0), vision, attention, (0, 0, "idle"), (0, 0), store,
                              drive_off="battery off (or no head)")
        self.assertEqual(st["drive"], {"mode": "off", "why": "battery off (or no head)"})


if __name__ == "__main__":
    unittest.main()
