"""robot/mic.py + robot/stt.py: utterances out of the speech probabilities, audio conversion, the Gemini request."""
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
import wave

import numpy as np

from robot.mic import CHUNK_S, Segmenter, to_16k_mono
from robot.stt import DailyLimit, GeminiSTT, wav_bytes
from robot.tts import Refused
from robot.vad import CHUNK


def run(seg, probs):
    """Feed chunks whose samples = their index; returns the utterances as (first chunk, last chunk, t0)."""
    out = []
    for i, p in enumerate(probs):
        got = seg.feed(np.full(CHUNK, i, np.float32), p, i * CHUNK_S)
        if got is not None:
            pcm, t0, t1 = got
            out.append((int(pcm[0]), int(pcm[-1]), round(t0, 3)))
    return out


class SegmenterTest(unittest.TestCase):
    def seg(self):
        return Segmenter(on=0.5, off=0.35, start_s=0.1, end_s=0.6, preroll_s=0.3, min_s=0.4, max_s=12.0)

    def test_one_utterance_with_preroll_and_a_short_tail(self):
        probs = [0.0] * 20 + [0.9] * 40 + [0.1] * 30           # 1.3 s speech after 0.64 s quiet
        (first, last, t0), = run(self.seg(), probs)
        self.assertEqual(first, 20 - 9)                         # 0.3 s (9 chunks) of what came before
        self.assertLessEqual(last, 60 + 5)                      # ends shortly after the speech
        self.assertAlmostEqual(t0, (20 - 9) * CHUNK_S, places=3)

    def test_a_click_is_not_speech_and_too_short_is_dropped(self):
        self.assertEqual(run(self.seg(), [0.0] * 10 + [0.9, 0.9] + [0.0] * 40), [])    # 64 ms click
        self.assertEqual(run(self.seg(), [0.0] * 10 + [0.9] * 4 + [0.0] * 40), [])     # 0.13 s of speech: too short
        self.assertEqual(len(run(self.seg(), [0.0] * 10 + [0.9] * 15 + [0.0] * 40)), 1)

    def test_pauses_between_words_do_not_split(self):
        probs = [0.9] * 15 + [0.2] * 10 + [0.9] * 15 + [0.0] * 30   # a 0.32 s pause inside
        self.assertEqual(len(run(self.seg(), probs)), 1)

    def test_long_speech_is_cut(self):
        self.assertGreaterEqual(len(run(self.seg(), [0.9] * 1000)), 2)              # 32 s -> pieces of <= 12 s


class AudioTest(unittest.TestCase):
    def test_48k_stereo_int32_to_16k_mono(self):
        t = np.arange(4800) / 48000
        left = (np.sin(2 * np.pi * 440 * t) * 2 ** 30).astype("<i4")
        raw = np.stack([left, np.zeros_like(left)], axis=1).tobytes()
        x = to_16k_mono(raw)
        self.assertEqual(len(x), 1600)
        self.assertAlmostEqual(float(np.abs(x).max()), 0.5, delta=0.05)

    def test_wav_is_16k_mono_and_normalised(self):
        w = wave.open(io.BytesIO(wav_bytes(np.full(1600, 0.01, np.float32))))
        self.assertEqual((w.getframerate(), w.getnchannels()), (16000, 1))
        x = np.frombuffer(w.readframes(1600), "<i2")
        self.assertGreater(int(x.max()), 29000)


class HighPassTest(unittest.TestCase):
    def test_rumble_out_speech_band_in_across_chunks(self):
        from robot.mic import HighPass
        hp = HighPass(cutoff_hz=100, rate=16000)
        t = np.arange(16000 * 2) / 16000
        drift = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.2                 # slow wander + DC, like the I2S mic
        voice = 0.01 * np.sin(2 * np.pi * 400 * t)
        out = np.concatenate([hp((drift + voice)[i:i + 512]) for i in range(0, len(t), 512)])
        self.assertLess(float(np.abs(out[:512]).max()), 0.05)        # no start-up jump from the offset
        settled = out[8000:]
        self.assertLess(float(np.abs(settled).max()), 0.015)          # the 0.7 drift is gone...
        self.assertGreater(float(np.sqrt(np.mean(settled ** 2))), 0.006)   # ...the voice (rms 0.007) stays


class AutoGainTest(unittest.TestCase):
    @staticmethod
    def feed(agc, x, speech):
        y = agc(x)
        agc.learn(x, speech)
        return y

    @staticmethod
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    def test_quiet_speech_up_silence_not_blown_up(self):
        from robot.mic import AutoGain
        agc = AutoGain(target=0.1, max_gain=50)
        rng = np.random.default_rng(0)
        speech = [self.feed(agc, rng.normal(0, 0.004, 512).astype(np.float32), True) for _ in range(60)]  # -48 dBFS
        self.assertGreater(self.rms(speech[-1]), 0.05)
        silence = [self.feed(agc, rng.normal(0, 0.0002, 512).astype(np.float32), False) for _ in range(30)]
        self.assertLess(self.rms(silence[-1]), 0.02)                                       # stays quiet
        self.assertLessEqual(agc.gain, 50)

    def test_a_loud_noise_that_is_not_speech_leaves_the_gain_alone(self):   # a click, the head's servos
        from robot.mic import AutoGain
        agc = AutoGain(target=0.1, max_gain=120)
        rng = np.random.default_rng(1)
        before = agc.gain
        for _ in range(10):
            self.feed(agc, rng.normal(0, 0.02, 512).astype(np.float32), False)          # -34 dBFS clatter
        self.assertGreaterEqual(agc.gain, before)                                       # not pulled down
        quiet = self.feed(agc, rng.normal(0, 0.0007, 512).astype(np.float32), True)     # quiet speech right after
        self.assertGreater(self.rms(quiet), 0.02)                                       # still heard


class NormaliseTest(unittest.TestCase):
    def test_a_quiet_utterance_is_scaled_up_once_without_clipping(self):
        from robot.mic import normalise
        rng = np.random.default_rng(2)
        x = (rng.normal(0, 0.001, 16000) * np.linspace(0.2, 1, 16000)).astype(np.float32)   # quiet, getting louder
        y = normalise(x)
        self.assertAlmostEqual(float(np.abs(y).max()), 0.5, places=5)
        np.testing.assert_allclose(y / np.abs(y).max(), x / np.abs(x).max(), rtol=1e-5)      # the shape kept
        self.assertEqual(len(normalise(np.zeros(10, np.float32))), 10)


class GeminiSttTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.now = [1793462160.0]

    def tearDown(self):
        self.dir.cleanup()

    def stt(self, post, per_day=200):
        return GeminiSTT(model="m-flash", key="k", post=post, clock=lambda: self.now[0],
                         limit=DailyLimit(Path(self.dir.name) / "u.json", per_day, clock=lambda: self.now[0]))

    def test_daily_limit_then_deaf_until_tomorrow(self):
        calls = []

        def post(req, timeout):
            calls.append(1)
            return {"candidates": [{"content": {"parts": [{"text": "Evet"}]}}]}
        stt = self.stt(post, per_day=2)
        self.assertEqual([stt.transcribe(np.zeros(1600, np.float32)) for _ in range(3)], ["Evet", "Evet", ""])
        self.assertIn("daily limit", stt.last_error)
        self.now[0] += 86400
        self.assertEqual(stt.transcribe(np.zeros(1600, np.float32)), "Evet")
        self.assertEqual(len(calls), 3)

    def test_no_credit_pauses(self):
        calls = []

        def post(req, timeout):
            calls.append(1)
            raise Refused(402, "credits depleted")
        stt = self.stt(post)
        self.assertEqual(stt.transcribe(np.zeros(1600, np.float32)), "")
        self.assertEqual(stt.transcribe(np.zeros(1600, np.float32)), "")
        self.assertEqual(len(calls), 1)                             # asked once, then paused

    def test_request_and_answer(self):
        seen = {}

        def post(req, timeout):
            seen["body"], seen["key"] = json.loads(req.data), req.get_header("X-goog-api-key")
            seen["url"] = req.full_url
            return {"candidates": [{"content": {"parts": [{"text": "Benim adım Doğu."}]}}]}
        stt = self.stt(post)
        self.assertEqual(stt.transcribe(np.zeros(16000, np.float32), prompt="Doğu, Emre"), "Benim adım Doğu.")
        parts = seen["body"]["contents"][0]["parts"]
        self.assertEqual(parts[0]["inline_data"]["mime_type"], "audio/wav")
        self.assertTrue(base64.b64decode(parts[0]["inline_data"]["data"]).startswith(b"RIFF"))
        self.assertIn("Turkish", parts[1]["text"])
        self.assertIn("Doğu, Emre", parts[1]["text"])
        self.assertEqual(seen["key"], "k")
        self.assertIn("/models/m-flash:generateContent", seen["url"])


def db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-12)


def raw48(left):
    """float -1..1 at 48 kHz on the left channel -> arecord S32_LE stereo bytes."""
    li = (np.asarray(left, np.float64) * 2 ** 31).clip(-2 ** 31, 2 ** 31 - 1).astype("<i4")
    return np.stack([li, np.zeros_like(li)], axis=1).tobytes()


class DriftChainTest(unittest.TestCase):
    """The mic's slow drift through the live chain (to_16k_mono + HighPass, 32 ms at a time) must not come out as
    clicks: the per-chunk mean subtraction made -56 dBFS out of silence (2026-11-01)."""

    def test_drift_stays_out_of_the_speech_band(self):
        from robot.mic import Decimator, HighPass
        t = np.arange(48000 * 6) / 48000
        drift = 0.012 * np.sin(2 * np.pi * 3 * t) + 0.01 * np.sin(2 * np.pi * 11 * t) + 0.003   # ~ -36 dBFS
        down, hp = Decimator(), HighPass()
        out = np.concatenate([hp(to_16k_mono(raw48(drift[i:i + 1536]), down=down))
                              for i in range(0, len(t), 1536)])
        self.assertLess(db(out[16000:]), -80)

    def test_folding_above_8k_is_gone(self):
        from robot.mic import Decimator
        t = np.arange(48000) / 48000
        down = Decimator()
        out = np.concatenate([to_16k_mono(raw48(0.5 * np.sin(2 * np.pi * 11000 * t[i:i + 1536])), down=down)
                              for i in range(0, len(t), 1536)])
        self.assertLess(db(out[1000:]), -40)                     # was 0.42 of 0.5 at 5 kHz with the 3-sample mean

    def test_speech_band_passes_across_chunks(self):
        from robot.mic import Decimator
        t = np.arange(48000) / 48000
        down = Decimator()
        out = np.concatenate([to_16k_mono(raw48(0.5 * np.sin(2 * np.pi * 1000 * t[i:i + 1536])), down=down)
                              for i in range(0, len(t), 1536)])
        self.assertAlmostEqual(float(np.abs(out[1000:]).max()), 0.5, delta=0.02)
        self.assertEqual(len(out), 16000)


class LimiterTest(unittest.TestCase):
    def test_loud_speech_is_limited_not_clipped_and_the_gain_learns(self):
        from robot.mic import AutoGain
        agc = AutoGain(target=0.1, max_gain=120, too_loud_s=1.0)
        agc.loud = 0.1 / 120                                       # after a quiet spell: waiting at x120
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.01, 512).astype(np.float32)           # -40 dBFS speech at x120
        y = agc(x)
        self.assertLessEqual(float(np.abs(y).max()), 0.9 + 1e-6)
        np.testing.assert_allclose(y / np.abs(y).max(), x / np.abs(x).max(), rtol=1e-5)   # shape kept, no clip
        for _ in range(40):                                        # the VAD never says speech: still learns
            agc(x)
            agc.learn(x, False)
        self.assertLess(agc.gain, 20)

    def test_a_short_loud_click_still_leaves_the_gain_alone(self):
        from robot.mic import AutoGain
        agc = AutoGain(target=0.1, max_gain=120, too_loud_s=1.0)
        agc.loud = 0.1 / 120
        rng = np.random.default_rng(4)
        before = agc.gain
        for _ in range(10):                                        # 0.3 s of servo noise
            x = rng.normal(0, 0.02, 512).astype(np.float32)
            agc(x)
            agc.learn(x, False)
        self.assertGreaterEqual(agc.gain, before)


class SegmenterOnsetTest(unittest.TestCase):
    def seg(self):
        return Segmenter(on=0.5, off=0.35, start_s=0.1, end_s=0.6, preroll_s=0.3, min_s=0.16, max_s=12.0)

    def test_onset_is_the_first_speech_chunk_and_t0_the_preroll(self):
        seg = self.seg()
        probs = [0.0] * 20 + [0.9] * 10 + [0.0] * 30
        got = [seg.feed(np.zeros(CHUNK, np.float32), p, i * CHUNK_S) for i, p in enumerate(probs)]
        u, = [g for g in got if g is not None]
        pcm, t0, t1 = u                                           # still unpacks as 3
        self.assertAlmostEqual(u.onset, 20 * CHUNK_S)
        self.assertAlmostEqual(t0, 11 * CHUNK_S)

    def test_dur_is_kept(self):
        self.assertEqual(len(run(self.seg(), [0.0] * 10 + [0.9] * 6 + [0.0] * 40)), 1)     # 0.19 s of speech
        self.assertEqual(run(self.seg(), [0.0] * 10 + [0.9] * 4 + [0.0] * 40), [])         # 0.13 s: no

    def test_the_tail_is_the_next_preroll(self):
        probs = [0.9] * 20 + [0.0] * 19 + [0.9] * 20 + [0.0] * 30     # second phrase 0.6 s after the first ended
        (_, _, _), (first2, _, _) = run(self.seg(), probs)
        self.assertEqual(first2, 39 - 9)                          # a full 0.3 s pre-roll in front of it


class FakeVad:
    def __init__(self, probs):
        self.probs, self.resets, self.i = probs, 0, 0

    def prob(self, chunk):
        p = self.probs(self.i) if callable(self.probs) else self.probs
        self.i += 1
        return p

    def reset(self):
        self.resets += 1


class FakeArecord:
    def __init__(self, chunks):
        self.left = chunks
        self.stdout = self

    def read(self, n):
        if self.left <= 0:
            return b""
        self.left -= 1
        return bytes(n)

    def poll(self):
        return 0 if self.left <= 0 else None

    def terminate(self):
        self.left = 0

    def wait(self, timeout=None):
        return 0


class EarsLoopTest(unittest.TestCase):
    def ears(self, vad, muted=lambda: False, on=None, procs=(1000,)):
        from robot.mic import Ears
        now = [100.0]

        def clock():
            now[0] += CHUNK_S                                       # one chunk read per clock call pair
            return now[0]
        started = []

        def popen(cmd, **kw):
            p = FakeArecord(procs[len(started)] if len(started) < len(procs) else 0)
            started.append(p)
            return p
        e = Ears(vad, muted=muted, on_utterance=on, clock=clock, popen=popen)
        return e, started

    def test_vad_reset_while_idle_even_when_noise_scores_0_4(self):
        import threading
        vad = FakeVad(0.4)                                          # a TV: never speech, never "quiet" either
        e, _ = self.ears(vad, procs=(int(10 / CHUNK_S),))
        stop = threading.Event()
        e.proc = e._open()
        for _ in range(int(10 / CHUNK_S)):
            e._chunk(bytes(CHUNK * 24))
        self.assertGreaterEqual(vad.resets, 2)

    def test_the_live_listener_gets_start_pieces_and_end(self):
        calls = []

        class Listener:
            def speech_started(self, onset, pcm):
                calls.append(("start", len(pcm)))

            def speech_chunk(self, pcm):
                calls.append(("chunk", len(pcm)))

            def speech_ended(self, kept):
                calls.append(("end", kept))
        vad = FakeVad(0.9)
        got = []
        e, _ = self.ears(vad, on=lambda pcm, t0, t1: got.append(len(pcm)))
        e.listener = Listener()
        e.proc = e._open()
        for _ in range(20):
            e._chunk(bytes(CHUNK * 24))
        vad.probs = 0.0
        for _ in range(40):
            e._chunk(bytes(CHUNK * 24))
        kinds = [c[0] for c in calls]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "end")
        self.assertEqual(calls[-1], ("end", True))
        self.assertEqual(kinds.count("start"), 1)
        self.assertEqual(len(got), 1)                               # the clip still comes, after the end

    def test_pause_s_for_the_listening_nods(self):
        vad = FakeVad(0.9)
        e, _ = self.ears(vad)
        e.proc = e._open()
        self.assertIsNone(e.pause_s)                                # nobody speaking
        for _ in range(10):
            e._chunk(bytes(CHUNK * 24))
        self.assertEqual(e.pause_s, 0.0)                            # a sentence, sound coming in
        vad.probs = 0.0
        e._chunk(bytes(CHUNK * 24))
        e._chunk(bytes(CHUNK * 24))
        self.assertAlmostEqual(e.pause_s, 2 * CHUNK_S)              # a short pause inside it

    def test_deaf_for_a_while_after_the_robot_spoke(self):
        from robot import config as cfg
        speaking = [True]
        vad = FakeVad(0.9)
        e, _ = self.ears(vad, muted=lambda: speaking[0])
        e.proc = e._open()
        for _ in range(10):
            e._chunk(bytes(CHUNK * 24))
        speaking[0] = False
        n_hold = int(cfg.MIC_UNMUTE_HOLD_S / (2 * CHUNK_S))         # the fake clock moves 2 chunks per _chunk
        for _ in range(n_hold - 1):
            e._chunk(bytes(CHUNK * 24))
        self.assertEqual(e.chunks, 0)                               # nothing went to the VAD yet
        for _ in range(3):
            e._chunk(bytes(CHUNK * 24))
        self.assertGreater(e.chunks, 0)

    def test_onset_passed_only_to_callbacks_that_take_it(self):
        got_new, got_old = [], []
        probs = lambda i: 0.9 if 20 <= i < 40 else 0.0             # noqa: E731
        e, _ = self.ears(FakeVad(probs), on=lambda pcm, t0, t1, onset=None: got_new.append((t0, onset)))
        e.proc = e._open()
        for _ in range(80):
            e._chunk(bytes(CHUNK * 24))
        (t0, onset), = got_new
        self.assertGreater(onset, t0)
        e, _ = self.ears(FakeVad(probs), on=lambda pcm, t0, t1: got_old.append(t0))
        e.proc = e._open()
        for _ in range(80):
            e._chunk(bytes(CHUNK * 24))
        self.assertEqual(len(got_old), 1)

    def test_chunk_times_follow_the_sample_count(self):
        e, _ = self.ears(FakeVad(0.0))
        times = [e._chunk_time() for _ in range(5)]
        np.testing.assert_allclose(np.diff(times), CHUNK_S)        # not the (jumping) read times

    def test_arecord_that_ends_is_started_again(self):
        import threading
        e, started = self.ears(FakeVad(0.0), procs=(3, 3, 1000))
        stop = threading.Event()
        waits = []
        stop.wait = lambda s: waits.append(s) or False              # no real sleeping
        e.proc = e._open()
        orig = e._chunk
        n = [0]

        def chunk(raw):
            n[0] += 1
            if n[0] > 20:
                stop.set()
            orig(raw)
        e._chunk = chunk
        e._run(stop)
        self.assertEqual(e.restarts, 2)
        self.assertEqual(waits, [1.0, 2.0])
        self.assertIn("2 mic restarts", e.status())


class CleanTest(unittest.TestCase):
    def test_not_words_are_nothing(self):
        from robot.stt import clean
        for junk in ("...", ".", "(anlaşılmıyor)", "[müzik]", "*inaudible*", "  ", ""):
            self.assertEqual(clean(junk), "", junk)
        self.assertEqual(clean("Evet. (gülüyor)"), "Evet.")
        self.assertEqual(clean("Merhaba Arkbi"), "Merhaba Arkbi")

    def test_thinking_config_only_when_set(self):
        seen = []

        def post(req, timeout):
            seen.append(json.loads(req.data)["generationConfig"])
            return {"candidates": [{"content": {"parts": [{"text": "(sessizlik)"}]}}]}
        with tempfile.TemporaryDirectory() as d:
            lim = DailyLimit(Path(d) / "u.json", 10)
            self.assertEqual(GeminiSTT(model="m", key="k", post=post, limit=lim, thinking=None).transcribe(
                np.zeros(1600)), "")
            GeminiSTT(model="m", key="k", post=post, limit=lim, thinking={"thinkingLevel": "low"}).transcribe(
                np.zeros(1600))
        self.assertNotIn("thinkingConfig", seen[0])
        self.assertEqual(seen[1]["thinkingConfig"], {"thinkingLevel": "low"})

    def test_thinking_is_off_by_default_per_model_family(self):
        from robot import config
        from robot.stt import thinking_config
        self.assertEqual(config.STT_GEMINI_THINKING, "off")         # Selim 2026-11-01: no time lost there
        self.assertEqual(thinking_config("gemini-3.8-flash", "off"), {"thinkingLevel": "minimal"})
        self.assertEqual(thinking_config("gemini-2.5-flash", "off"), {"thinkingBudget": 0})
        self.assertIsNone(thinking_config("gemini-3.8-flash", None))

    def test_a_model_that_refuses_the_thinking_field_is_asked_again_without_it_for_good(self):
        from robot.tts import Refused
        seen = []

        def post(req, timeout):
            gen = json.loads(req.data)["generationConfig"]
            seen.append("thinkingConfig" in gen)
            if "thinkingConfig" in gen:
                raise Refused(400, 'Invalid JSON payload received. Unknown name "thinkingLevel"')
            return {"candidates": [{"content": {"parts": [{"text": "Merhaba"}]}}]}
        with tempfile.TemporaryDirectory() as d:
            stt = GeminiSTT(model="m", key="k", post=post, limit=DailyLimit(Path(d) / "u.json", 10), thinking="off")
            with self.assertLogs("robot.stt", "WARNING") as logs:
                self.assertEqual(stt.transcribe(np.zeros(1600)), "Merhaba")
            self.assertEqual(stt.transcribe(np.zeros(1600)), "Merhaba")
        self.assertEqual(seen, [True, False, False])                 # one retry, then never sent again
        self.assertEqual(len(logs.output), 1)
        self.assertIsNotNone(stt.last_seconds)

    def test_a_model_without_minimal_thinking_gets_low(self):        # 3.8 Flash, 2026-11-01
        from robot.tts import Refused
        seen = []

        def post(req, timeout):
            think = json.loads(req.data)["generationConfig"].get("thinkingConfig")
            seen.append(think)
            if think == {"thinkingLevel": "minimal"}:
                raise Refused(400, "Thinking level MINIMAL is not supported for this model.")
            return {"candidates": [{"content": {"parts": [{"text": "Merhaba"}]}}]}
        with tempfile.TemporaryDirectory() as d:
            stt = GeminiSTT(model="gemini-3.8-flash", key="k", post=post, limit=DailyLimit(Path(d) / "u.json", 10),
                            thinking="off")
            with self.assertLogs("robot.stt", "WARNING"):
                self.assertEqual(stt.transcribe(np.zeros(1600)), "Merhaba")
            stt.transcribe(np.zeros(1600))
        self.assertEqual(seen, [{"thinkingLevel": "minimal"}, {"thinkingLevel": "low"}, {"thinkingLevel": "low"}])

    def test_other_400s_are_not_retried(self):
        from robot.tts import Refused
        calls = []

        def post(req, timeout):
            calls.append(1)
            raise Refused(400, "Request payload size exceeds the limit")
        with tempfile.TemporaryDirectory() as d:
            stt = GeminiSTT(model="m", key="k", post=post, limit=DailyLimit(Path(d) / "u.json", 10), thinking="off")
            self.assertEqual(stt.transcribe(np.zeros(1600)), "")
        self.assertEqual(len(calls), 1)

    def test_the_end_silence_still_keeps_dur(self):
        from robot import config
        seg = Segmenter(on=0.5, off=0.35, start_s=0.1, end_s=config.MIC_END_S, preroll_s=0.3,
                        min_s=config.MIC_MIN_S, max_s=12.0)
        self.assertLessEqual(config.MIC_END_S, 0.45)
        self.assertEqual(len(run(seg, [0.0] * 10 + [0.9] * 6 + [0.0] * 40)), 1)     # "Dur": 0.19 s of speech
        # a 0.3 s breath inside a sentence ("Arkbi, ... bana bak") does not split it
        self.assertEqual(len(run(seg, [0.0] * 10 + [0.9] * 15 + [0.0] * 9 + [0.9] * 15 + [0.0] * 40)), 1)

    def test_prompt_never_translates(self):
        from robot.stt import GeminiSTT as G
        self.assertIn("never translate", G.PROMPT)


class WhisperRestartTest(unittest.TestCase):
    def test_a_dead_server_is_started_again(self):
        from robot.stt import WhisperSTT

        class Proc:
            n = 0

            def __init__(self, *a, **kw):
                Proc.n += 1
                self.dead = False
                self.lines = ["ready\n"]
                self.stdin, self.stdout = self, self

            def readline(self):
                return self.lines.pop(0) if self.lines else ""

            def write(self, s):
                if not self.dead:
                    self.lines.append("Merhaba\n")

            def flush(self):
                pass

            def poll(self):
                return 1 if self.dead else None

            def kill(self):
                self.dead = True

            def close(self):
                pass

            def wait(self, timeout=None):
                return 0
        w = WhisperSTT(popen=Proc)
        self.assertEqual(w.transcribe(np.zeros(1600, np.float32)), "Merhaba")
        w.proc.dead = True
        self.assertEqual(w.transcribe(np.zeros(1600, np.float32)), "Merhaba")
        self.assertEqual(Proc.n, 2)
        w.close()


if __name__ == "__main__":
    unittest.main()
