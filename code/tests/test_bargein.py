"""Barge-in: robot/bargein.py (echo reference + per-band detector), Voice.interrupt, the ears listening through the
mute, and the conversation's rule (someone looking / a conversation open)."""
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from robot import tts
from robot.bargein import BargeIn, EchoRef, RATE, FRAME, band_power

from tests.test_conversation import FakeStt, FakeVoice, track


def frames_of(x):
    """16 kHz mono -> what the speaker plays (48 kHz stereo 16-bit bytes)."""
    y = np.repeat(np.clip(x, -1, 1), 3)
    return (np.repeat(y, 2) * 32767).astype("<i2").tobytes()


def speechy(n, rng):
    """A voice-like sound: noise shaped by a syllable envelope (~4 Hz)."""
    t = np.arange(n) / RATE
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    return (rng.normal(0, 0.1, n) * env).astype(np.float32)


class Room:
    """The robot plays `played` from t=0; the mic hears it 20 dB down after `lag` (+ someone else from `other_at`)."""

    def __init__(self, seconds=4.0, other_at=None, lag=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.now = [0.0]
        self.echo = EchoRef(clock=lambda: self.now[0], start_delay_s=0.0)
        self.played = speechy(int(seconds * RATE), rng)
        self.echo.add(frames_of(self.played), pid=1)
        n = int((seconds + 1) * RATE)
        self.mic = rng.normal(0, 1e-4, n).astype(np.float32)
        k = int(lag * RATE)
        self.mic[k:k + len(self.played)] += 0.1 * self.played
        if other_at is not None:
            i = int(other_at * RATE)
            self.mic[i:i + RATE] += speechy(RATE, np.random.default_rng(seed + 1)) * 0.5

    def run(self, det):
        """Times (s) at which the detector said barge-in."""
        out = []
        for i in range(0, len(self.mic) - FRAME, FRAME):
            if det.feed(self.mic[i:i + FRAME], i / RATE):
                out.append(i / RATE)
        return out


def flat_det(echo, **kw):
    return BargeIn(echo, lag_s=0.1, coupling=[-20.0] * 9, noise=[-48.0] * 9, **kw)   # the room: 1e-4 noise


class DetectorTest(unittest.TestCase):
    def test_its_own_echo_is_not_a_barge_in(self):
        room = Room()
        self.assertEqual(room.run(flat_det(room.echo)), [])

    def test_someone_talking_over_it_is(self):
        room = Room(other_at=2.0)
        hits = room.run(flat_det(room.echo))
        self.assertTrue(hits)
        self.assertGreaterEqual(hits[0], 2.0)
        self.assertLess(hits[0], 2.5)                       # within half a second

    def test_the_echo_level_is_learnt(self):
        room = Room(seconds=6.0)
        det = BargeIn(room.echo, lag_s=0.1, coupling=[-26.0] * 9, noise=[-48.0] * 9)   # 6 dB off at the start
        room.run(det)
        self.assertLess(np.abs(det.coupling[2:] + 20).max(), 3.5)

    def test_nothing_playing_nothing_detected(self):
        echo = EchoRef(clock=lambda: 0.0)
        det = flat_det(echo)
        self.assertFalse(any(det.feed(np.random.default_rng(0).normal(0, 0.1, FRAME), t / 10) for t in range(20)))

    def test_reference_timing_follows_speakerd_and_stops_at_a_cut(self):
        now = [10.0]
        echo = EchoRef(clock=lambda: now[0], start_delay_s=0.03)
        echo.add(frames_of(np.full(RATE, 0.5, np.float32)), pid=7)      # 1 s, predicted 10.03 .. 11.03
        echo.ended(7, 11.13)                                             # speakerd: it ends at 11.13
        self.assertEqual(echo.segment(10.05, 16).max(), 0.0)
        self.assertGreater(echo.segment(10.2, 16).min(), 0.4)
        echo.cut(10.6)
        self.assertEqual(echo.segment(10.7, 16).max(), 0.0)
        self.assertTrue(echo.playing(10.7))
        self.assertFalse(echo.playing(13.0))

    def test_band_power_shape(self):
        self.assertEqual(band_power(np.zeros(FRAME)).shape, (9,))


class SelfTuningTest(unittest.TestCase):
    """2026-11-02: a damaged speaker's echo was far over the old prediction - every reply cut at once."""

    def loud_room(self, seed=0):
        return Room(seconds=4.0, seed=seed)                    # echo at -20 dB...

    def test_a_false_alarm_raises_the_bands_that_fired_until_the_echo_passes(self):
        d = tempfile.mkdtemp()
        tune = os.path.join(d, "barge.json")
        det = BargeIn(self.loud_room().echo, lag_s=0.1, coupling=[-40.0] * 9, noise=[-48.0] * 9, learn=0.0,
                      tune_file=tune)                           # ...predicted at -40: its own voice "barges in"
        for i in range(8):
            room = self.loud_room(seed=i)
            det.echo = room.echo
            det.reset()
            if not room.run(det):
                break
            add = det.false_alarm()
            self.assertTrue((add >= 0).all() and add.max() <= 6.0)
        self.assertLess(i, 7)                                   # a few false alarms, then its echo passes
        room = self.loud_room(seed=99)
        det.echo = room.echo
        det.reset()
        self.assertEqual(room.run(det), [])
        again = BargeIn(EchoRef(clock=lambda: 0.0), tune_file=tune)    # saved: the next start knows it
        np.testing.assert_allclose(again.coupling, det.coupling, atol=0.01)
        self.assertEqual(again.tuned, "learnt")
        self.assertGreaterEqual(again.false_alarms, 1)
        room = Room(other_at=2.0, seed=5)                       # someone talking over it is still heard
        det.echo = room.echo
        det.reset()
        self.assertTrue(room.run(det))

    def test_a_false_alarm_without_a_barge_in_changes_nothing(self):
        det = flat_det(EchoRef(clock=lambda: 0.0))
        before = det.coupling.copy()
        self.assertEqual(det.false_alarm().max(), 0.0)
        np.testing.assert_array_equal(det.coupling, before)

    def test_coupling_never_over_the_cap(self):
        from robot import config
        det = BargeIn(EchoRef(clock=lambda: 0.0), coupling=[-2.0] * 9, noise=[-48.0] * 9)
        det.fired = [np.full(9, 30.0)]
        det.false_alarm()
        self.assertLessEqual(det.coupling.max(), config.BARGE_COUPLING_MAX_DB)

    def test_calibration_fit_stops_false_alarms(self):
        from robot.bargein import fit_coupling, db
        room = self.loud_room()
        det = BargeIn(room.echo, lag_s=0.1, coupling=[-40.0] * 9, noise=[-48.0] * 9, learn=0.0)
        self.assertTrue(room.run(det))                          # the old values: false barge-ins
        tail, mic_db, tail_db = np.zeros(9), [], []
        for i in range(0, len(room.mic) - FRAME, FRAME):
            tail = np.maximum(band_power(room.echo.segment(i / RATE - 0.1)), tail * 10 ** (-0.25))
            mic_db.append(db(band_power(room.mic[i:i + FRAME])))
            tail_db.append(db(tail))
        coupling, used = fit_coupling(np.array(mic_db), np.array(tail_db), [-48.0] * 9, default=[-40.0] * 9)
        self.assertTrue((used >= 20).all())
        self.assertTrue((np.abs(coupling[1:] + 20) < 6).all())  # the echo is at -20 dB
        self.assertEqual(self.loud_room(seed=3).run(
            BargeIn(self.loud_room(seed=3).echo, lag_s=0.1, coupling=coupling, noise=[-48.0] * 9)), [])
        room = Room(other_at=2.0, seed=4)
        self.assertTrue(room.run(BargeIn(room.echo, lag_s=0.1, coupling=coupling, noise=[-48.0] * 9)))

    def test_bad_tuning_files_are_ignored(self):
        from robot.bargein import load_tuning
        d = tempfile.mkdtemp()
        for i, text in enumerate(("{", "[]", '{"coupling": [1, 2]}', '{"coupling": ' + str([float("nan")] * 9)
                                  .replace("nan", "NaN") + "}")):
            p = os.path.join(d, f"{i}.json")
            with open(p, "w") as f:
                f.write(text)
            self.assertNotIn("coupling", load_tuning(p))
        self.assertEqual(load_tuning(os.path.join(d, "missing.json")), {})


class SnippetTest(unittest.TestCase):
    def test_the_newest_are_kept_by_time_written_not_by_name(self):   # 15:37 today before 20:19 yesterday
        from robot.bargein import prune
        d = tempfile.mkdtemp()
        for i, name in enumerate(("201919.wav", "201921.wav", "153705.wav", "153707.wav")):
            p = os.path.join(d, name)
            open(p, "wb").close()
            os.utime(p, (1000 + i, 1000 + i))
        prune(d, 2)
        self.assertEqual(sorted(os.listdir(d)), ["153705.wav", "153707.wav"])

    def test_snippet_names_carry_the_date(self):
        from robot.bargein import save_snippet
        d = tempfile.mkdtemp()
        echo = EchoRef(clock=lambda: 0.0)
        save_snippet([(np.zeros(FRAME, np.float32), 0.0)], echo, out_dir=d, when=1793623046.0)
        (name,) = os.listdir(d)
        self.assertRegex(name, r"^20\d{6}-\d{6}\.wav$")


class NoiseTest(unittest.TestCase):
    def test_the_room_floor_is_learnt_while_the_robot_is_silent(self):
        det = BargeIn(EchoRef(clock=lambda: 0.0), noise=[-80.0] * 9)
        rng = np.random.default_rng(0)
        for _ in range(300):
            det.learn_noise(rng.normal(0, 1e-3, FRAME))
        self.assertGreater(det.noise.min(), -45)


class VoiceInterruptTest(unittest.TestCase):
    def test_interrupt_drops_the_queue_and_stops_the_speaker(self):
        d = tempfile.mkdtemp()
        stopped, played = [], []
        release = threading.Event()

        def slow_player(wav):
            played.append(len(wav))
            release.wait(2)
        v = tts.Voice(key="k", cache_dir=d, synth=lambda *a: (np.zeros(2400, np.float32), 24000),
                      player=slow_player, stopper=lambda: (stopped.append(1), release.set()),
                      clock=lambda: 1793490000.0, fallback_models=(), espeak=False,
                      level_file=os.path.join(d, "none.json"))
        with v.hold():
            v.say("Bir.")
            v.say("İki.")
            v.say("Üç.")
            deadline = time.time() + 2
            while not played and time.time() < deadline:
                time.sleep(0.01)
            v.interrupt()
            self.assertFalse(v._holds)                              # a hold no longer keeps it busy
        v.close(timeout=2)
        self.assertEqual(stopped, [1])
        self.assertEqual(len(played), 1)                            # "İki", "Üç" never played
        self.assertFalse(v.busy)


class EarsBargeTest(unittest.TestCase):
    def test_listens_through_the_mute_after_a_barge_in(self):
        from robot.mic import Ears, CHUNK
        from tests.test_ears import FakeArecord, FakeVad
        now = [100.0]

        def clock():
            now[0] += 0.016
            return now[0]
        vad = FakeVad(0.9)
        speaking = [True]
        e = Ears(vad, muted=lambda: speaking[0], on_utterance=lambda *a, **k: got.append(1), clock=clock,
                 popen=lambda *a, **k: FakeArecord(10000))
        got, asked = [], []

        class Det:
            def __init__(self):
                self.n = 0

            def feed(self, chunk, t):
                self.n += 1
                return self.n == 10                                  # the 10th chunk: someone talks over it

            def reset(self):
                pass

            def hot_s(self):
                return 0.3
        e.barge = Det()
        e.on_barge = lambda: (asked.append(1), speaking.__setitem__(0, False)) and True or True
        e.proc = e._open()
        for _ in range(9):
            e._chunk(bytes(CHUNK * 24))
        self.assertEqual(e.chunks, 0)                               # muted: nothing heard
        e._chunk(bytes(CHUNK * 24))                                 # barge-in
        self.assertEqual(asked, [1])
        self.assertEqual(e.barged, 1)
        speaking[0] = True                                          # its voice still fading out: busy a moment
        for _ in range(20):
            e._chunk(bytes(CHUNK * 24))
        self.assertGreater(e.chunks, 0)                             # ...but the ears listen now


class ConversationBargeTest(unittest.TestCase):
    def setUp(self):
        from robot.conversation import Conversation
        from robot.people import PeopleStore
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        self.now = [100.0]
        self.voice = FakeVoice()
        self.voice.busy = True                                      # it is speaking
        self.voice.interrupts = 0
        self.voice.interrupt = lambda: setattr(self.voice, "interrupts", self.voice.interrupts + 1)
        self.c = Conversation(self.voice, FakeStt(), self.store, clock=lambda: self.now[0],
                              wall=lambda: 1793490000.0, start=False, sleep=lambda s: None)

    def tearDown(self):
        self.dir.cleanup()

    def test_only_when_someone_looks_or_a_conversation_is_open(self):
        self.assertFalse(self.c.barge_in())                         # a door, a TV: nobody looking
        self.assertEqual(self.voice.interrupts, 0)
        self.c.update([track(1, None)], [], self.now[0])
        self.assertTrue(self.c.barge_in())
        self.assertEqual(self.voice.interrupts, 1)
        self.assertIn("interrupted", [e["kind"] for e in self.c.events()])

    def test_an_answer_right_after_it_spoke_is_heard_but_cuts_nothing(self):   # scenario simulator 2026-11-01
        self.voice.busy = False                                     # finished: the ears were in their hold
        self.c.update([track(1, None)], [], self.now[0])
        self.assertTrue(self.c.barge_in())                          # listen now...
        self.assertEqual(self.voice.interrupts, 0)                  # ...nothing to stop
        self.assertEqual(self.c._cut, 0)                            # its own words are not taken out of the answer
        self.assertNotIn("interrupted", [e["kind"] for e in self.c.events()])


if __name__ == "__main__":
    unittest.main()
