"""robot/tts.py: cache, daily limit, fallback, volume by time of day, the voice thread (no network, no sound)."""
import base64
import io
import json
import os
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np

from robot import config, tts


def tone(n=2400):
    return (0.5 * np.sin(np.arange(n) / 5)).astype(np.float32), 24000


def at(hour):
    """A clock showing this hour in Turkey."""
    return lambda: datetime(2026, 10, 31, hour, 15, tzinfo=ZoneInfo(config.TTS_TIMEZONE)).timestamp()


class VoiceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.calls, self.played, self.espoken = [], [], []

    def tearDown(self):
        self.dir.cleanup()

    def voice(self, key="test-key", daily=200, synth=None, clock=at(14), fallback_models=(), espeak=True):
        def fake_synth(text, key, model, voice, style):
            self.calls.append(text)
            return tone()

        def fake_espeak(text, lang):
            self.espoken.append((text, lang))
            return tone(1200)
        return tts.Voice(key=key, cache_dir=self.dir.name, daily_requests=daily, synth=synth or fake_synth,
                         fallback=fake_espeak, player=self.played.append, clock=clock,
                         fallback_models=fallback_models, espeak=espeak,
                         level_file=os.path.join(self.dir.name, "no-level.json"))   # never the robot's own setting

    def test_a_phrase_is_paid_for_once(self):
        v = self.voice()
        self.assertEqual(v.say_now("Merhaba Selim"), "gemini")
        self.assertEqual(v.say_now("Merhaba Selim"), "cache")
        self.assertEqual(self.voice().say_now("Merhaba Selim"), "cache")      # also after a restart
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.played), 3)

    def test_the_name_is_said_arkbi(self):
        self.voice().say_now("Ben Arcby.")
        self.assertEqual(self.calls, ["Ben Arkbi."])

    def test_daily_limit_then_espeak_and_a_new_day(self):
        v = self.voice(daily=2)
        self.assertEqual([v.say_now(t) for t in ("bir", "iki", "üç")], ["gemini", "gemini", "espeak"])
        self.assertEqual(v.usage()[1], 2)
        tomorrow = self.voice(daily=2, clock=lambda: at(14)() + 86400)
        self.assertEqual(tomorrow.say_now("dört"), "gemini")

    def test_errors_and_no_key_fall_back_to_espeak(self):
        def broken(*a):
            raise RuntimeError("HTTP 503")
        self.assertEqual(self.voice(synth=broken).say_now("Günaydın"), "espeak")
        self.assertEqual(self.espoken[-1], ("Günaydın", "tr"))
        self.assertEqual(self.voice(key="").say_now("Good morning"), "espeak")
        self.assertEqual(self.espoken[-1], ("Good morning", "en"))
        self.assertEqual(self.calls, [])

    def test_no_credit_pauses_gemini_for_a_while(self):
        now = [at(14)()]

        def no_credit(*a):
            self.calls.append(a[0])
            raise tts.Refused(402, "prepayment credits are depleted")
        v = self.voice(synth=no_credit, clock=lambda: now[0])
        self.assertEqual([v.say_now(t) for t in ("bir", "iki", "üç")], ["espeak"] * 3)
        self.assertEqual(len(self.calls), 1)                         # asked once, not for every phrase
        self.assertEqual(v.usage()[1], 0)                            # refused = nothing made, not counted
        now[0] += config.TTS_PAUSE_S + 1
        v.say_now("dört")
        self.assertEqual(len(self.calls), 2)                         # tries again after the pause

    def test_too_many_per_minute_waits_as_long_as_google_asks(self):
        now = [at(14)()]

        def busy(*a):
            self.calls.append(a[0])
            raise tts.Refused(429, "quota", retry_s=30.0)
        v = self.voice(synth=busy, clock=lambda: now[0])
        v.say_now("bir")
        self.assertEqual(v.paused[config.TTS_MODEL], now[0] + 30.0)
        now[0] += 31
        v.say_now("iki")
        self.assertEqual(len(self.calls), 2)

    def test_same_voice_from_the_next_model_when_one_is_out(self):
        used = []

        def synth(text, key, model, voice, style):
            used.append((model, style))
            if model == "primary":
                raise tts.Refused(429, "daily quota", retry_s=25000)
            return tone()
        v = tts.Voice(key="k", cache_dir=self.dir.name, model="primary", style="Say calmly", synth=synth,
                      player=self.played.append, clock=at(14), fallback_models=("second", "third"), espeak=False)
        self.assertEqual(v.say_now("Saat kaç"), "gemini")
        self.assertEqual(used, [("primary", "Say calmly"), ("second", "Say calmly")])   # 3.x: the style too
        self.assertEqual(v.say_now("Merhaba"), "gemini")
        self.assertEqual(used[-1], ("second", "Say calmly"))                 # primary skipped while it is out
        v.paused.clear()
        self.assertEqual(v.say_now("Saat kaç"), "cache")                    # made by the fallback: free

    def test_silent_not_espeak_when_no_model_can_speak(self):
        def broken(*a):
            raise RuntimeError("no internet")
        v = self.voice(synth=broken, fallback_models=("second",), espeak=False)
        self.assertEqual(v.say_now("Merhaba"), "silent")
        self.assertEqual(self.played, [])
        self.assertEqual(self.espoken, [])

    def test_an_empty_answer_is_tried_once_more(self):
        answers = [RuntimeError("no audio (finishReason OTHER)"), tone()]

        def flaky(*a):
            self.calls.append(a[0])
            x = answers.pop(0)
            if isinstance(x, Exception):
                raise x
            return x
        self.assertEqual(self.voice(synth=flaky).say_now("Günaydın Selim!"), "gemini")
        self.assertEqual(len(self.calls), 2)

    def test_quiet_at_night(self):
        self.assertEqual(self.voice(clock=at(14)).level(), config.TTS_LEVEL_DAY)
        self.assertEqual(self.voice(clock=at(23)).level(), config.TTS_LEVEL_NIGHT)
        self.assertEqual(self.voice(clock=at(7)).level(), config.TTS_LEVEL_NIGHT)

    def test_loudness_is_the_peak_at_48k_stereo(self):
        w = wave.open(io.BytesIO(tts.to_card(*tone(), level=0.3)))
        self.assertEqual((w.getframerate(), w.getnchannels()), (48000, 2))
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32767
        self.assertAlmostEqual(float(np.abs(x).max()), 0.3, places=2)

    def test_say_does_not_wait_and_keeps_the_order(self):
        gate = threading.Event()
        order = []

        def slow_player(wav):
            gate.wait(2)
            order.append(len(order))
        v = self.voice()
        v.player = slow_player
        v.say("bir")
        v.say("iki")
        self.assertTrue(v.busy)                                    # returned at once, still to speak
        gate.set()
        v.close()
        self.assertEqual(order, [0, 1])
        self.assertFalse(v.busy)

    def test_prepare_makes_phrases_without_playing(self):
        waits = []
        v = self.voice()
        self.assertEqual(v.prepare(["Merhaba Meltem", "Merhaba Halil", "Merhaba Meltem"], sleep=waits.append),
                         ["gemini", "gemini", "cache"])
        self.assertEqual(self.played, [])
        self.assertEqual(waits, [config.TTS_PREPARE_GAP_S] * 2)    # paced after each Gemini call only


class GeminiCallTest(unittest.TestCase):
    def test_request_and_answer(self):
        pcm = (np.arange(100) * 100).astype("<i2")
        seen = {}

        def post(req, timeout):
            seen["url"], seen["body"] = req.full_url, json.loads(req.data)
            seen["key"] = req.get_header("X-goog-api-key")
            return {"candidates": [{"content": {"parts": [{"inlineData": {
                "mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(pcm.tobytes()).decode()}}]}}]}
        x, rate = tts.gemini_pcm("Merhaba", "k", model="m-tts", voice="Charon", style="Say calmly", post=post)
        self.assertEqual(rate, 24000)
        self.assertEqual(len(x), 100)
        self.assertIn("/models/m-tts:generateContent", seen["url"])
        self.assertNotIn("k", seen["url"].split("models/")[1].replace("m-tts", ""))   # key not in the URL
        self.assertEqual(seen["key"], "k")
        self.assertEqual(seen["body"]["contents"][0]["parts"][0]["text"], 'Say calmly: "Merhaba"')
        voice = seen["body"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
        self.assertEqual(voice["voiceName"], "Charon")


class SecretTest(unittest.TestCase):
    def test_reads_the_key_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "secrets.env")
            with open(p, "w") as f:
                f.write("# keys\nOTHER=1\nGEMINI_API_KEY = \"AIzaTest\"\n")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GEMINI_API_KEY", None)
                self.assertEqual(tts.read_secret("GEMINI_API_KEY", p), "AIzaTest")
                self.assertIsNone(tts.read_secret("NOPE", p))
                self.assertIsNone(tts.read_secret("GEMINI_API_KEY", os.path.join(d, "missing")))
                os.environ["GEMINI_API_KEY"] = "fromenv"
                self.assertEqual(tts.read_secret("GEMINI_API_KEY", p), "fromenv")


class LevelTest(unittest.TestCase):
    def test_the_dashboard_setting_wins_and_is_kept_in_range(self):
        d = tempfile.mkdtemp()
        f = os.path.join(d, "level.json")
        v = tts.Voice(key="", cache_dir=d, clock=at(14), level_file=f)
        self.assertEqual(v.level(), tts.config.TTS_LEVEL_DAY)          # no file: the config
        for set_to, expect in ((0.3, 0.3), (7, 1.0), (-1, 0.0)):
            with open(f, "w") as h:
                json.dump({"level": set_to}, h)
            self.assertEqual(v.level(), expect)
        with open(f, "w") as h:
            h.write("not json")
        self.assertEqual(v.level(), tts.config.TTS_LEVEL_DAY)          # broken file: the config


class SpeakerTest(unittest.TestCase):
    """robot/tts.Speaker: the client of robot/speakerd.py (run for real here, with its 'null' card)."""

    def phrase(self, seconds=0.1):
        n = int(48000 * seconds)
        return tts.to_card(np.sin(np.arange(n) / 5.0).astype(np.float32), 48000, 0.5)

    def test_plays_and_waits_until_the_sound_has_left_the_speaker(self):
        waits = []
        sp = tts.Speaker(device="null", sleep=waits.append)
        self.assertTrue(sp.alive)
        self.assertIn("null", sp.info)
        sp.play(self.phrase(0.2))
        self.assertEqual(len(waits), 1)
        self.assertGreater(waits[0], 0.1)                  # the phrase was still playing when "done" came...
        self.assertLess(waits[0], 0.2 + sp.ECHO_S + 0.2)   # ...+ the room echo
        sp.close()
        self.assertEqual(sp.proc.returncode, 0)            # ended by itself after stdin closed
        self.assertFalse(sp.alive)

    def test_a_dead_speaker_fails_at_once_not_after_ten_seconds(self):
        sp = tts.Speaker(device="null", sleep=lambda s: None)
        sp.proc.kill()
        sp.proc.wait()
        t = time.monotonic()
        with self.assertRaises(tts.SpeakerError):
            sp.play(self.phrase(3.0))
        self.assertLess(time.monotonic() - t, 2.0)
        sp.close()

    def test_a_card_that_cannot_be_opened_is_a_clear_error(self):
        with self.assertRaises(tts.SpeakerError) as cm:
            tts.Speaker(device="no-such-device", sleep=lambda s: None)
        self.assertIn("no-such-device", str(cm.exception))

    def test_only_48k_stereo(self):
        sp = tts.Speaker(device="null", sleep=lambda s: None)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(bytes(320))
        with self.assertRaises(ValueError):
            sp.play(buf.getvalue())
        sp.close()

    def test_open_speaker_restarts_a_dead_one_with_its_log(self):
        real = tts.Speaker
        made = []

        def factory(card, log_path=None):
            made.append(log_path)
            return real(device="null", log_path=log_path, sleep=lambda s: None)
        d = tempfile.mkdtemp()
        logf = os.path.join(d, "speaker.log")
        with mock.patch.object(tts, "Speaker", factory), mock.patch.object(tts, "_speaker", None):
            a = tts.open_speaker(0, log_path=logf)
            self.assertIs(tts.open_speaker(0), a)            # kept open
            a.proc.kill()
            a.proc.wait()
            a._reader.join(2)
            b = tts.open_speaker(0)                          # dead: a new one, same log
            self.assertIsNot(a, b)
            self.assertEqual(made, [logf, logf])
            self.assertIsNotNone(a.proc.returncode)          # the old one was reaped
            b.play(self.phrase(0.05))
            tts.close_speaker()
        self.assertIn("speakerd", open(logf).read())

    def test_phrases_fade_in_and_out(self):
        x = np.ones(4800, np.float32) + np.sin(np.arange(4800)).astype(np.float32)
        b = tts.to_card(x, 48000, 0.5)
        y = np.frombuffer(b[44:], "<i2")[::2]
        self.assertLess(abs(int(y[0])), 200)
        self.assertLess(abs(int(y[-1])), 200)

    def test_busy_while_a_phrase_is_still_being_made(self):
        made = threading.Event()

        def slow_synth(text, key, model, voice, style):
            made.wait(2)
            return tone()
        v = tts.Voice(key="k", cache_dir=tempfile.mkdtemp(), synth=slow_synth, player=lambda b: None,
                      clock=at(14), fallback_models=())
        v.say("Merhaba")
        time.sleep(0.05)                                    # out of the queue, being made
        self.assertTrue(v.busy)
        made.set()
        v.close()
        self.assertFalse(v.busy)

    def test_one_voice_thread_even_when_two_threads_say_the_first_phrase(self):
        v = tts.Voice(key="k", cache_dir=tempfile.mkdtemp(), synth=lambda *a: tone(), player=lambda b: None,
                      clock=at(14), fallback_models=())
        go = threading.Barrier(8)

        def first():
            go.wait()
            v.say("Merhaba")
        ts = [threading.Thread(target=first) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        voices = [t for t in threading.enumerate() if t.name == "voice"]
        self.assertEqual(len(voices), 1)
        v.close()


def wav_answer(pcm16, trailer=b"C2PA" + bytes(6068)):
    """What gemini-3.8-flash-tts sends: a whole WAV file, a C2PA signature chunk after the audio."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue() + trailer


class WavAnswerTest(unittest.TestCase):
    """The fallback model's WAV answers: only the audio is played (it was a noise burst at every phrase end)."""

    def post_with(self, data, mime):
        return lambda req, timeout: {"candidates": [{"content": {"parts": [{"inlineData": {
            "mimeType": mime, "data": base64.b64encode(data).decode()}}]}}]}

    def test_a_wav_answer_gives_only_its_audio(self):
        pcm = (np.sin(np.arange(24000) / 7) * 8000).astype("<i2")
        for mime in ("audio/wav", "audio/L16;codec=pcm;rate=24000"):     # the header decides, not the mime type
            x, rate = tts.gemini_pcm("Merhaba", "k", post=self.post_with(wav_answer(pcm), mime))
            self.assertEqual(rate, 24000)
            self.assertEqual(len(x), 24000)
            self.assertLess(np.abs(x * 32768 - pcm).max(), 1)

    def test_a_raw_answer_is_still_raw(self):
        pcm = (np.arange(100) * 10).astype("<i2")
        x, rate = tts.gemini_pcm("x", "k", post=self.post_with(pcm.tobytes(), "audio/L16;codec=pcm;rate=16000"))
        self.assertEqual((len(x), rate), (100, 16000))

    def poisoned_cache_file(self, path, pcm16):
        """How the bad answers were cached: the WAV bytes taken as samples, then written by write_wav."""
        raw = wav_answer(pcm16)
        as_samples = np.frombuffer(raw[:len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768
        tts.write_wav(path, as_samples, 24000)

    def test_a_poisoned_cache_file_is_healed_without_a_new_request(self):
        d = tempfile.mkdtemp()
        calls = []
        v = tts.Voice(key="k", cache_dir=d, synth=lambda *a: calls.append(a) or tone(), player=lambda b: None,
                      clock=at(14), fallback_models=(), level_file=os.path.join(d, "none.json"))
        pcm = (np.sin(np.arange(12000) / 7) * 9000).astype("<i2")
        path = v.cache_path(tts.spoken("Merhaba Selim"))
        self.poisoned_cache_file(path, pcm)
        x, rate, src = v.audio("Merhaba Selim")
        self.assertEqual((src, rate, calls), ("cache", 24000, []))
        self.assertLess(abs(len(x) - 12000), 2)                 # no header, no 6 KB signature
        self.assertLess(np.abs(x[:11990] * 32768 - pcm[:11990]).max(), 3)
        y, _ = tts.read_wav(str(path))                          # saved back clean
        self.assertIs(tts.unwrap_wav(y), y)
        self.assertLess(abs(len(y) - 12000), 2)

    def test_a_long_poisoned_phrase_keeps_all_its_audio(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "x.wav")
        for n in (48000, 70001, 100000):                 # data size over 64 KB: its high half was damaged
            pcm = (np.sin(np.arange(n) / 7) * 9000).astype("<i2")
            self.poisoned_cache_file(path, pcm)
            x, _ = tts.read_wav(path)
            self.assertLess(abs(len(tts.unwrap_wav(x)) - n), 2, n)

    def test_normal_audio_is_left_alone(self):
        x = tone()[0]
        self.assertIs(tts.unwrap_wav(x), x)


class QuotaTest(unittest.TestCase):
    def voice(self, synth, clock=None):
        d = tempfile.mkdtemp()
        return tts.Voice(key="k", cache_dir=d, synth=synth, player=lambda b: None, clock=clock or at(14),
                         fallback_models=("backup",), espeak=False, level_file=os.path.join(d, "none.json"))

    def test_a_per_day_429_rests_the_model_until_midnight_in_california(self):
        asked = []

        def synth(text, key, model, voice, style):
            asked.append(model)
            if model == config.TTS_MODEL:
                raise tts.Refused(429, "quota", retry_s=30, per_day=True)
            return tone()
        v = self.voice(synth)
        self.assertEqual(v.audio("bir")[2], "gemini")
        self.assertEqual(v.audio("iki")[2], "gemini")
        self.assertEqual(asked, [config.TTS_MODEL, "backup", "backup"])    # not asked again the same day
        until = datetime.fromtimestamp(v.paused[config.TTS_MODEL], ZoneInfo("America/Los_Angeles"))
        self.assertEqual((until.hour, until.minute), (0, 1))
        self.assertEqual(v.usage()[1], 2)                    # the refused request was not counted

    def test_a_per_minute_429_waits_as_asked(self):
        def synth(text, key, model, voice, style):
            raise tts.Refused(429, "slow down", retry_s=30)
        v = self.voice(synth)
        v.audio("bir")
        self.assertAlmostEqual(v.paused[config.TTS_MODEL] - at(14)(), 30)

    def test_a_model_that_is_gone_rests_for_the_day(self):
        def synth(text, key, model, voice, style):
            if model == "backup":
                raise tts.Refused(404, "models/backup is not found")
            raise tts.Refused(429, "x", retry_s=5)
        v = self.voice(synth)
        self.assertEqual(v.audio("bir")[2], "silent")
        self.assertGreater(v.paused["backup"] - at(14)(), 3600)

    def test_per_day_is_read_from_googles_answer(self):
        import urllib.error
        body = json.dumps({"error": {"message": "Quota exceeded", "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]}}).encode()

        class Err(urllib.error.HTTPError):
            def read(self):
                return body

        def urlopen(req, timeout):
            raise Err("u", 429, "x", {}, None)
        with mock.patch("urllib.request.urlopen", urlopen):
            with self.assertRaises(tts.Refused) as cm:
                tts._post_json(None, 1)
        self.assertTrue(cm.exception.per_day)
        self.assertEqual(cm.exception.retry_s, 12)


class PipelineTest(unittest.TestCase):
    """A reply in sentences: the next one is made while the one before plays, in order, busy all along."""

    def voice(self, synth, player):
        d = tempfile.mkdtemp()
        return tts.Voice(key="k", cache_dir=d, synth=synth, player=player, clock=at(14), fallback_models=(),
                         espeak=False, level_file=os.path.join(d, "none.json"))

    def test_sentence_two_is_made_while_sentence_one_plays(self):
        log, playing = [], threading.Event()

        def synth(text, key, model, voice, style):
            log.append(("make", text))
            return tone()

        def player(wav):
            log.append(("play", len(wav)))
            playing.set()
            time.sleep(0.2)
        v = self.voice(synth, player)
        first = {}
        with v.hold():
            v.say("Bir.", mark=first)
            self.assertTrue(playing.wait(2))
            v.say("İki.")                               # the mind wrote it while "Bir." plays
            time.sleep(0.05)
            self.assertEqual(log[-1], ("make", "İki."))  # made before "Bir." has finished
        v.close()
        self.assertEqual([k for k, _ in log], ["make", "play", "make", "play"])
        self.assertEqual(first["src"], "gemini")
        self.assertLessEqual(first["made"], first["play"])

    def test_busy_between_sentences_while_the_reply_is_still_written(self):
        v = self.voice(lambda *a: tone(), lambda w: None)
        with v.hold():
            v.say("Bir.")
            time.sleep(0.1)                             # spoken and done; the next sentence not written yet
            self.assertTrue(v.busy)
        self.assertFalse(v.busy)
        v.close()

    def test_order_is_kept_when_a_later_phrase_is_quicker_to_make(self):
        played = []

        def synth(text, key, model, voice, style):
            time.sleep(0.15 if text == "Uzun." else 0)
            return tone(6000 if text == "Uzun." else 2000)
        v = self.voice(synth, played.append)
        for t in ("Uzun.", "Kısa."):
            v.say(t)
        v.close()
        self.assertEqual(len(played), 2)
        self.assertGreater(len(played[0]), len(played[1]))

    def test_cached_says_whether_a_phrase_costs_a_request(self):
        v = self.voice(lambda *a: tone(), lambda w: None)
        self.assertFalse(v.cached("Hmm, bakalım."))
        v.prepare(["Hmm, bakalım."], sleep=lambda s: None)
        self.assertTrue(v.cached("Hmm, bakalım."))

    def test_two_429s_in_a_row_rest_the_model_for_the_day(self):
        now = [at(14)()]
        asked = []

        def synth(text, key, model, voice, style):
            asked.append(model)
            raise tts.Refused(429, "You exceeded your current quota", retry_s=20)
        d = tempfile.mkdtemp()
        v = tts.Voice(key="k", cache_dir=d, synth=synth, player=lambda b: None, clock=lambda: now[0],
                      fallback_models=(), espeak=False, level_file=os.path.join(d, "none.json"))
        v.audio("bir")
        self.assertAlmostEqual(v.paused[config.TTS_MODEL] - now[0], 20)
        now[0] += 21
        v.audio("iki")
        self.assertGreater(v.paused[config.TTS_MODEL] - now[0], 3600)
        now[0] += 60
        v.audio("üç")
        self.assertEqual(len(asked), 2)                 # the third reply did not wait for a refusal


class LimitNoticeTest(unittest.TestCase):
    def test_says_its_limit_is_full_from_the_cache_now_and_then(self):
        d = tempfile.mkdtemp()
        played = []
        v = tts.Voice(key="k", cache_dir=d, daily_requests=1, synth=lambda *a: tone(), player=played.append,
                      streamer=None, clock=at(14), fallback_models=(), espeak=False,
                      level_file=os.path.join(d, "none.json"))
        self.assertEqual(v.prepare([config.TTS_LIMIT_PHRASE], sleep=lambda s: None), ["gemini"])   # the 1 request
        self.assertEqual(v.say_now("Merhaba."), "limit")      # limit full: says so
        self.assertEqual(v.say_now("Nasılsın?"), "silent")    # ...not again right away
        self.assertEqual(len(played), 1)

    def test_turkish_without_turkish_letters_is_turkish(self):
        self.assertEqual(tts.guess_lang("Arkbi bana kendini anlat"), "tr")
        self.assertEqual(tts.guess_lang("What is your name?"), "en")


class StreamTest(unittest.TestCase):
    """3.x models: the phrase plays while Gemini still makes it; whole = cached; broken / refused = next model."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.played = []

    def tearDown(self):
        self.dir.cleanup()

    def voice(self, streamer, models=("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts")):
        return tts.Voice(key="k", cache_dir=self.dir.name, model=models[0], fallback_models=models[1:],
                         synth=lambda *a: tone(), player=lambda b: self.played.append(("whole", len(b))),
                         stream_player=lambda pieces: self.played.append(("stream", sum(len(p) for p in pieces))),
                         streamer=streamer, clock=at(14), espeak=False,
                         level_file=os.path.join(self.dir.name, "none.json"))

    def test_streams_then_plays_from_the_cache(self):
        asked = []

        def streamer(text, key, model, voice, style):
            asked.append((text, model, style))
            for _ in range(5):
                yield np.full(2400, 0.3, np.float32)
        v = self.voice(streamer)
        mark = {}
        v.say("<chuckle> Olur.", mark=mark)
        v.close()
        self.assertEqual(asked[0][0], "<chuckle> Olur.")
        self.assertEqual(self.played[0][0], "stream")
        self.assertAlmostEqual(self.played[0][1], 5 * 2400 * 2 * 4, delta=8)   # 0.5 s at 48 kHz stereo, 16-bit
        self.assertEqual(mark["src"], "stream")
        v2 = self.voice(streamer)
        v2.say("<chuckle> Olur.")
        v2.close()
        self.assertEqual(self.played[1][0], "whole")                   # cached: no second request
        self.assertEqual(len(asked), 1)

    def test_sentence_two_is_made_while_sentence_one_still_streams(self):     # no pause between sentences
        events, release = [], threading.Event()

        def streamer(text, key, model, voice, style):
            events.append(("start", text))
            yield np.zeros(2400, np.float32)
            if text == "Bir.":
                release.wait(2)                     # sentence 1 is still coming...
            events.append(("end", text))
            yield np.zeros(2400, np.float32)
        v = self.voice(streamer)
        v.say("Bir.")
        v.say("İki.")
        deadline = time.time() + 2
        while ("start", "İki.") not in events and time.time() < deadline:
            time.sleep(0.01)
        self.assertIn(("start", "İki."), events)    # ...and sentence 2 has started already
        self.assertNotIn(("end", "Bir."), events)
        release.set()
        v.close()
        self.assertEqual([p[0] for p in self.played], ["stream", "stream"])

    def test_refused_model_the_next_one_streams(self):
        asked = []

        def streamer(text, key, model, voice, style):
            asked.append(model)
            if model == "gemini-3.8-flash-tts":
                raise tts.Refused(429, "per day quota", per_day=True)
            yield np.zeros(2400, np.float32)
        v = self.voice(streamer)
        v.say("Merhaba.")
        v.close()
        self.assertEqual(asked, ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"])
        self.assertEqual(self.played[0][0], "stream")

    def test_broken_off_stream_is_played_but_not_cached(self):
        def streamer(text, key, model, voice, style):
            yield np.zeros(2400, np.float32)
            raise OSError("connection reset")
        v = self.voice(streamer)
        v.say("Merhaba.")
        v.close()
        self.assertEqual(self.played[0][0], "stream")
        self.assertFalse(any(Path(self.dir.name).rglob("*.wav")))

    def test_never_louder_than_the_cap_whatever_asks(self):                    # the speaker, 2026-11-01
        x = np.sin(np.arange(24000) / 7).astype(np.float32)
        wav = tts.to_card(x, 24000, 1.0)
        y = np.frombuffer(wav[44:], "<i2").astype(np.float64) / 32767
        self.assertLessEqual(np.abs(y).max(), tts.config.TTS_LEVEL_MAX + 1e-3)
        cs = tts.CardStream(24000, level=1.0, peak=1.0)
        y = np.frombuffer(cs.push(x) + cs.finish(), "<i2").astype(np.float64) / 32767
        self.assertLessEqual(np.abs(y).max(), tts.config.TTS_LEVEL_MAX + 1e-3)

    def test_card_stream_has_no_seam_between_pieces(self):
        x = (0.5 * np.sin(np.arange(24000) / 7)).astype(np.float32)
        cs = tts.CardStream(24000, level=0.6, peak=1.0)
        out = b"".join(cs.push(x[i:i + 1000]) for i in range(0, len(x), 1000)) + cs.finish()
        y = np.frombuffer(out, "<i2")[::2].astype(np.float64) / 32767
        self.assertAlmostEqual(len(y), 48000, delta=2)
        ref = np.interp(np.arange(len(y)) / 2, np.arange(len(x)), x * 0.6)
        self.assertLess(np.abs(y[300:-300] - ref[300:-300]).max(), 1e-3)   # same as one piece, apart from fades
        self.assertLess(abs(y[0]), 0.01)
        self.assertLess(abs(y[-1]), 0.05)

    def test_gemini_stream_reads_the_events(self):
        import base64 as b64
        pcm = (np.arange(10, dtype="<i2") * 100).tobytes()
        lines = [b"data: " + json.dumps({"event_type": "step.delta", "delta": {"type": "audio",
                                                                                "data": b64.b64encode(pcm[:7]).decode()}}).encode(),
                 b"data: " + json.dumps({"delta": {"type": "audio", "data": b64.b64encode(pcm[7:]).decode()}}).encode(),
                 b"data: [DONE]"]
        seen = {}

        class R(list):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def urlopen(req, timeout):
            seen["body"] = json.loads(req.data)
            return R(lines)
        got = np.concatenate(list(tts.gemini_stream("<sigh> Peki.", "k", "gemini-3.8-flash-tts", "Charon",
                                                    "calm", urlopen=urlopen)))
        np.testing.assert_allclose(got * 32768, np.arange(10) * 100)
        content = seen["body"]["input"][0]["content"][0]
        self.assertEqual(content["text"], "<sigh> Peki.")
        self.assertEqual(content["annotations"][0]["style"], "calm")        # never in the text
        self.assertTrue(seen["body"]["stream"])

    def test_the_old_model_gets_no_tags(self):
        self.assertTrue(tts.interactions_model("gemini-3.8-flash-tts"))
        self.assertFalse(tts.interactions_model("gemini-2.5-flash-preview-tts"))


class CloudTtsTest(unittest.TestCase):
    def test_charon_in_the_texts_language_without_tags(self):
        seen = {}
        pcm = (np.arange(240, dtype="<i2") * 10).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)

        def post(req, timeout):
            seen["body"] = json.loads(req.data)
            seen["auth"] = req.get_header("Authorization")
            return {"audioContent": base64.b64encode(buf.getvalue()).decode()}
        x, rate = tts.cloud_pcm("<chuckle> Merhaba Selim.", "cloud:Charon", token="t", post=post)
        self.assertEqual(rate, 24000)
        self.assertEqual(len(x), 240)
        self.assertEqual(seen["body"]["voice"]["name"], "tr-TR-Chirp3-HD-Charon")
        self.assertEqual(seen["body"]["input"]["text"], "Merhaba Selim.")       # no tags
        self.assertEqual(seen["auth"], "Bearer t")
        tts.cloud_pcm("What is your name?", "cloud:Charon", token="t", post=post)
        self.assertEqual(seen["body"]["voice"]["name"], "en-US-Chirp3-HD-Charon")
        self.assertFalse(tts.interactions_model("cloud:Charon"))


class AudioTagTest(unittest.TestCase):
    def test_allowed_tags_stay_others_go(self):
        self.assertEqual(tts.audio_tags("[uhm] Bir saniye.", keep=True), "<breath> Bir saniye.")   # old name
        self.assertEqual(tts.audio_tags("[Short  Pause] Tamam.", keep=True), "<short pause> Tamam.")
        self.assertEqual(tts.audio_tags("Tamam <Chuckle>, olur.", keep=True), "Tamam <chuckle>, olur.")
        self.assertEqual(tts.audio_tags("Bak <b>bu</b> <smiles>.", keep=True), "Bak bu.")
        self.assertEqual(tts.audio_tags("Tamam [gülümser].", keep=True), "Tamam.")
        self.assertEqual(tts.audio_tags("[sigh] Peki [laughing] olur.", keep=False), "Peki olur.")

    def test_spoken_checks_the_tags(self):
        self.assertEqual(tts.spoken("<sigh> Ben Arcby [robot sesi].", tags=True), "<sigh> Ben Arkbi.")

    def test_only_a_tag_is_never_a_request(self):
        calls = []
        d = tempfile.mkdtemp()
        v = tts.Voice(key="k", cache_dir=d, synth=lambda *a: calls.append(a) or tone(), player=lambda b: None,
                      clock=at(14), fallback_models=(), espeak=False, level_file=os.path.join(d, "none.json"))
        self.assertEqual(v.audio("[sigh]")[2], "silent")
        self.assertEqual(v.audio("[robot bip]")[2], "silent")
        self.assertEqual(calls, [])


class CacheLimitTest(unittest.TestCase):
    def test_the_least_recently_used_phrases_go_first(self):
        d = tempfile.mkdtemp()
        v = tts.Voice(key="k", cache_dir=d, synth=lambda *a: tone(24000), player=lambda b: None, clock=at(14),
                      fallback_models=(), level_file=os.path.join(d, "none.json"))
        for i, t in enumerate(("bir", "iki", "üç")):
            v.audio(t)
            p = v.cache_path(t)
            os.utime(p, (1000 + i, 1000 + i))
        size = v.cache_path("bir").stat().st_size
        v._trim_cache(max_bytes=int(size * 2.5))
        self.assertFalse(v.cache_path("bir").exists())
        self.assertTrue(v.cache_path("iki").exists() and v.cache_path("üç").exists())


if __name__ == "__main__":
    unittest.main()
