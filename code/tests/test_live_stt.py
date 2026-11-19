"""robot/live_stt.py (fake websocket), robot/wsock.py framing, and the live stream in robot/conversation.py: only a
sentence said to the robot is ever streamed; the clip STT takes over when the live one fails."""
import json
import queue
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot import wsock
from robot.conversation import Conversation
from robot.live_stt import LiveSTT, LiveSession, setup_message
from robot.people import PeopleStore
from robot.stt import DailyLimit

from tests.test_conversation import FakeStt, FakeVoice, track


class FakeWs:
    """Answers setupComplete, then the words once activityEnd arrives (or never: words=None)."""

    def __init__(self, words="Merhaba robot", parts=1):
        self.sent, self.inbox, self.words, self.parts = [], queue.Queue(), words, parts
        self.closed = False

    def send_text(self, text):
        msg = json.loads(text)
        self.sent.append(msg)
        if "setup" in msg:
            self.inbox.put(json.dumps({"setupComplete": {}}))
        if msg.get("realtimeInput", {}).get("activityEnd") is not None and self.words is not None:
            for w in self.words.split(" ", self.parts - 1) if self.parts > 1 else [self.words]:
                self.inbox.put(json.dumps({"serverContent": {"inputTranscription": {"text": w}}}))
            self.inbox.put(json.dumps({"serverContent": {"generationComplete": True}}))

    def recv(self, timeout=None):
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.closed = True

    def audio_sent(self):
        return sum(1 for m in self.sent if "audio" in m.get("realtimeInput", {}))


class LiveSessionTest(unittest.TestCase):
    def test_words_after_the_end(self):
        ws = FakeWs("Hâlâ beni dinliyor musun?", parts=2)
        s = LiveSession(lambda: ws, setup_message())
        for _ in range(5):
            s.feed(np.zeros(512, np.float32))
        self.assertEqual(s.finish(timeout=2), "Hâlâ beni dinliyor musun?")
        self.assertEqual(ws.audio_sent(), 5)
        kinds = [next(iter(m.get("realtimeInput", {"setup": 1}))) for m in ws.sent]
        self.assertEqual(kinds[:2], ["setup", "activityStart"])
        self.assertEqual(kinds.count("activityEnd"), 1)
        s._thread.join(timeout=2)
        self.assertTrue(ws.closed)

    def test_no_answer_in_time_is_none(self):
        s = LiveSession(lambda: FakeWs(words=None), setup_message())
        self.assertIsNone(s.finish(timeout=0.3))
        self.assertIn("no answer", s.error)

    def test_cannot_connect_is_none_not_a_crash(self):
        def refuse():
            raise wsock.Closed(None, "upgrade refused: HTTP/1.1 403")
        s = LiveSession(refuse, setup_message())
        self.assertIsNone(s.finish(timeout=1))
        self.assertIn("403", s.error)

    def test_setup_asks_turkish_and_our_own_activity_detection(self):
        m = setup_message()["setup"]
        self.assertIn("tr-TR", m["inputAudioTranscription"]["languageCodes"])
        self.assertTrue(m["realtimeInputConfig"]["automaticActivityDetection"]["disabled"])

    def test_daily_limit_and_no_key(self):
        with tempfile.TemporaryDirectory() as d:
            live = LiveSTT(key="k", connect=lambda: FakeWs(), limit=DailyLimit(Path(d) / "u.json", 1))
            s = live.open()
            self.assertIsNotNone(s)
            s.cancel()
            self.assertIsNone(live.open())
            self.assertIsNone(LiveSTT(key="", limit=DailyLimit(Path(d) / "v.json", 5)).open())


class FrameTest(unittest.TestCase):
    def test_client_frames_are_masked_with_the_right_length(self):
        for n in (5, 300, 70000):
            f = wsock.frame(wsock.OP_TEXT, b"x" * n, mask_key=b"\x01\x02\x03\x04")
            self.assertEqual(f[0], 0x81)
            self.assertTrue(f[1] & 0x80)
            head = 2 + (2 if 126 <= n < 1 << 16 else 8 if n >= 1 << 16 else 0)
            self.assertEqual(wsock.mask(f[head + 4:], b"\x01\x02\x03\x04"), b"x" * n)


class FakeLive:
    def __init__(self, words="Saat kaç?"):
        self.words, self.ws = words, []

    def open(self):
        ws = FakeWs(self.words)
        self.ws.append(ws)
        return LiveSession(lambda: ws, setup_message())


class ConversationStreamTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        self.selim = self.store.enrol([np.random.default_rng(0).normal(size=8) / 3])
        self.store.set_name(self.selim, "Selim")
        self.now = [100.0]
        self.voice, self.stt, self.live = FakeVoice(), FakeStt(), FakeLive()
        self.c = Conversation(self.voice, self.stt, self.store, clock=lambda: self.now[0],
                              wall=lambda: 1793490000.0, start=False, sleep=lambda s: None, live=self.live)

    def tearDown(self):
        self.dir.cleanup()

    def say(self, looking):
        """A sentence as the ears hand it over: started, 10 pieces, ended, then the whole clip."""
        if looking:
            self.c.update([track(1, self.selim)], [], self.now[0])
        onset = self.now[0]
        self.c.speech_started(onset, np.zeros(1600, np.float32))
        for _ in range(10):
            self.now[0] += 0.032
            self.c.speech_chunk(np.zeros(512, np.float32))
        self.now[0] += 1.2                              # past the look window: decided by now
        self.c.speech_chunk(np.zeros(512, np.float32))
        self.c.speech_ended(True)
        self.c._utterance(np.zeros(8000, np.float32), onset - 0.2, self.now[0],
                          scene=self.c._scene(onset - 0.2, self.now[0], onset))
        self.now[0] += 2

    def test_said_to_the_robot_streams_and_the_clip_stt_is_not_needed(self):
        self.say(looking=True)
        self.assertEqual(len(self.live.ws), 1)
        self.assertEqual(self.live.ws[0].audio_sent(), 12)          # pre-roll + every piece
        self.assertEqual(self.stt.calls, [])
        self.assertTrue(self.voice.said[-1].startswith("Saat "))

    def test_not_looking_nothing_leaves_the_robot(self):
        self.say(looking=False)
        self.assertEqual(self.live.ws, [])
        self.assertEqual(self.stt.calls, [])
        self.assertEqual(self.c.dropped, 1)

    def test_live_heard_nothing_the_clip_stt_takes_over(self):
        self.live.words = ""
        self.stt.answers.append("Saat kaç?")
        self.say(looking=True)
        self.assertEqual(len(self.stt.calls), 1)
        self.assertTrue(self.voice.said[-1].startswith("Saat "))

    def test_its_own_voice_or_a_click_cancels(self):
        self.c.update([track(1, self.selim)], [], self.now[0])
        self.c.speech_started(self.now[0], np.zeros(1600, np.float32))
        self.c.speech_ended(False)
        self.assertEqual(self.c._streams, {})
        self.live.ws[0] and self.assertIsNotNone(self.live.ws[0])


if __name__ == "__main__":
    unittest.main()
