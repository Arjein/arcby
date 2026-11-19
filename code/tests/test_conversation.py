"""robot/conversation.py: who is listened to, what the robot answers, meeting people (it never forgets anyone)."""
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np

from robot import config
from robot.conversation import Conversation, extract_name, is_yes, plain_words
from robot.people import PeopleStore


class FakeVoice:
    busy = False

    def __init__(self):
        self.said = []

    def say(self, text, level=None, mark=None):
        self.said.append(text)


class FakeStt:
    def __init__(self):
        self.answers, self.calls, self.last_error = [], [], None

    def transcribe(self, pcm, prompt=""):
        self.calls.append(prompt)
        return self.answers.pop(0) if self.answers else ""


def track(tid, person=None, looking=True, looked=5.0):
    return SimpleNamespace(id=tid, person=person, looking=looking, looked_for=lambda t: looked if looking else 0.0)


class FakeMind:
    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def reply(self, text, **scene):
        self.asked.append((text, scene))
        return self.answers.pop(0) if self.answers else None


class ConversationTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        rng = np.random.default_rng(0)
        self.selim = self.store.enrol([rng.normal(size=8) / 3])
        self.store.set_name(self.selim, "Selim")
        self.voice, self.stt = FakeVoice(), FakeStt()
        self.now = [100.0]
        self.c = Conversation(self.voice, self.stt, self.store, clock=lambda: self.now[0], wall=lambda: 1793490000.0,
                              start=False, sleep=lambda s: None)

    def tearDown(self):
        self.dir.cleanup()

    def say_to_it(self, text, looking=((1, "selim"),)):
        """Someone says text now - looking at the robot as they start (looking=...) or not (looking=())."""
        self.c.update([track(i, self.selim if p == "selim" else p) for i, p in looking], [], self.now[0])
        self.stt.answers.append(text)
        self.c._utterance(np.zeros(8000, np.float32), self.now[0], self.now[0] + 1)
        self.now[0] += 2

    def test_not_looking_at_it_nothing_is_sent(self):
        self.say_to_it("Napıyorsun?", looking=())              # looking at the phone
        self.assertEqual(self.stt.calls, [])
        self.assertEqual(self.c.dropped, 1)
        self.assertEqual(self.voice.said, [])

    def test_keeps_what_it_heard_as_wav(self):
        self.c.save_dir = self.dir.name + "/heard"
        self.say_to_it("Napıyorsun?", looking=())
        self.say_to_it("Saat kaç?")
        names = sorted(p.name for p in __import__("pathlib").Path(self.c.save_dir).glob("*.wav"))
        self.assertEqual(len(names), 2)
        self.assertTrue(any(n.endswith("-dropped.wav") for n in names))
        self.assertTrue(any("-heard-" in n and n.endswith(".wav") for n in names))   # + what the STT made of it

    def test_talked_over_and_only_its_own_words_heard_it_was_its_own_echo(self):
        # 2026-11-02: its own echo cut every reply and the words heard back were answered - a loop. Only its own
        # words after a cut = nobody talked over it: not answered, and the barge-in detector is told
        told = []
        self.c.on_false_barge = lambda: told.append(1) or np.zeros(9)
        self.c._recent_said.append((time.monotonic(), "İyiyim Selim, teşekkür ederim."))
        self.c._cut_at, self.c._cut_clock = time.monotonic(), self.now[0]
        self.say_to_it("İyiyim Selim.")                          # ends 0.55 s after the cut: with its voice
        self.assertEqual(self.voice.said, [])
        self.assertEqual(told, [1])
        kinds = [e["kind"] for e in self.c.events()]
        self.assertIn("own_echo", kinds)
        self.assertNotIn("heard", kinds)

    def test_its_greeting_heard_back_after_a_cut_is_not_answered(self):   # alive.log 2026-11-02 15:38:09
        self.voice.spoken = [(time.monotonic(), "Merhaba Selim!")]          # the greeter said it, not the conversation
        self.c._cut_at, self.c._cut_clock = time.monotonic(), self.now[0]
        self.say_to_it("Merhaba Selim.")
        self.assertEqual(self.voice.said, [])

    def test_talked_over_their_sound_went_on_but_only_its_words_understood_nothing_learnt(self):   # scenario 11a
        told = []
        self.c.on_false_barge = lambda: told.append(1) or np.zeros(9)
        self.c._recent_said.append((time.monotonic(), "İyiyim Selim, teşekkür ederim."))
        self.c._cut_at, self.c._cut_clock = time.monotonic(), self.now[0] - 1.0     # went on 1.5 s after the cut
        self.say_to_it("teşekkür")
        self.assertEqual(self.voice.said, [])                   # it stopped: that is the answer
        self.assertEqual(told, [])                              # a person was there: nothing learnt

    def test_talked_over_and_something_of_theirs_heard_it_answers(self):
        self.c._recent_said.append((time.monotonic(), "İyiyim Selim, teşekkür ederim."))
        self.c._cut_at = time.monotonic()
        self.say_to_it("İyiyim Selim. Saat kaç?")
        self.assertTrue(self.voice.said and self.voice.said[-1].startswith("Saat "))

    def test_the_echo_check_is_only_right_after_a_cut(self):
        self.c._recent_said.append((time.monotonic(), "Merhaba Selim!"))
        self.say_to_it("Merhaba")                               # no barge-in: Selim greets it - answered
        self.assertTrue(self.voice.said)

    def test_own_echo_words(self):
        from robot.conversation import own_echo
        self.assertTrue(own_echo("çağırma.", ["Anneni çağıramam, bu odadan çıkamıyorum."]))     # STT on its echo
        self.assertTrue(own_echo("Merhaba Selim.", ["Merhaba Selim!"]))
        self.assertTrue(own_echo("Hmm", ["Hmm, bakalım."]))
        self.assertFalse(own_echo("Dur!", ["Bugün hava çok güzel."]))
        self.assertFalse(own_echo("Merhaba, saat kaç?", ["Merhaba Selim!"]))
        self.assertFalse(own_echo("", ["Merhaba Selim!"]))
        self.assertFalse(own_echo("Merhaba", []))

    def test_looking_at_it_is_talking_to_it_no_wake_word(self):
        self.say_to_it("Napıyorsun?")
        self.assertIn("dinliyorum", self.voice.said[-1])

    def test_a_conversation_goes_on_without_looking_then_ends(self):
        self.say_to_it("Merhaba")
        self.assertEqual(self.voice.said[-1], "Merhaba Selim!")
        self.now[0] += 10
        self.say_to_it("saat kaç", looking=())                 # looking away now: still the conversation
        self.assertTrue(self.voice.said[-1].startswith("Saat "))
        self.now[0] += 25                                      # quiet for longer than CONV_SESSION_S
        self.say_to_it("napıyorsun", looking=())
        self.assertEqual(len(self.stt.calls), 2)               # not sent any more
        self.assertTrue(any(e["kind"] == "session_end" for e in self.c.events()))

    def test_goodbye_closes_the_conversation(self):
        self.say_to_it("Görüşürüz")
        self.assertEqual(self.voice.said[-1], "Görüşürüz Selim.")
        self.say_to_it("saat kaç", looking=())
        self.assertEqual(len(self.stt.calls), 1)

    def test_called_by_name(self):
        self.say_to_it("Arkbi")
        self.assertEqual(self.voice.said, ["Efendim, Selim?"])
        self.say_to_it("saat kaç")                             # the command after "Efendim?"
        self.assertTrue(self.voice.said[-1].startswith("Saat "))
        self.assertIn("Selim", self.stt.calls[0])                # the names go to the recogniser as a hint


    def test_commands(self):
        self.say_to_it("Arkbi, kendini tanıt")
        self.assertIn("Ben Arcby", self.voice.said[-1])
        self.say_to_it("Arcby, teşekkürler")
        self.assertEqual(self.voice.said[-1], "Rica ederim.")
        self.say_to_it("Arkbi, uçak yap")
        self.assertIn("bilmiyorum", self.voice.said[-1])

    def meet(self, looked=5.0):
        new = self.store.enrol([np.ones(8) / 8 ** 0.5])
        self.c.update([track(5, new, looked=looked)], [], self.now[0])
        return new

    def test_meeting_a_new_face(self):
        new = self.meet()
        self.c._ask_name(self.c._todo.get()[1])
        self.assertIn("Adın ne?", self.voice.said[-1])
        self.say_to_it("Benim adım doğu", looking=((5, new),))
        self.assertEqual(self.voice.said[-1], "Doğu, doğru mu?")
        self.say_to_it("Evet", looking=((5, new),))
        self.assertEqual(self.store.name(new), "Doğu")
        self.assertIn("Memnun oldum Doğu", self.voice.said[-1])

    def test_a_new_face_is_asked_only_after_looking_a_while(self):
        self.meet(looked=0.3)
        self.assertTrue(self.c._todo.empty())                  # a passing look: not yet
        self.meet(looked=1.5)
        self.assertEqual(self.c._todo.get()[0], "ask_name")

    def test_not_asked_while_talking_with_someone(self):
        self.say_to_it("Merhaba")                              # a conversation with Selim is open
        self.meet()
        self.assertTrue(self.c._todo.empty())

    def test_a_picture_is_never_asked(self):
        new = self.store.enrol([np.ones(8) / 8 ** 0.5])
        self.store.index[new]["picture"] = True
        self.c.update([track(5, new)], [], self.now[0])
        self.assertTrue(self.c._todo.empty())

    def test_not_asked_when_gone_by_then(self):
        new = self.meet()
        self.now[0] += 5                                       # out of view before the worker got to it
        self.c._ask_name(self.c._todo.get()[1])
        self.assertEqual(self.voice.said, [])

    def test_only_the_new_face_answers_the_name_question(self):
        new = self.meet()
        self.c._ask_name(self.c._todo.get()[1])
        self.say_to_it("Ne yapıyorsun?")                       # Selim, looking: not the answer
        self.assertNotIn("doğru mu", self.voice.said[-1])
        self.assertIsNotNone(self.c.expect)                     # still waiting for the new face
        self.say_to_it("Ahmet", looking=((5, new),))
        self.assertEqual(self.voice.said[-1], "Ahmet, doğru mu?")
        self.say_to_it("Tamam", looking=((1, "selim"),))        # Selim's "tamam" does not confirm it
        self.assertIsNone(self.store.name(new))
        self.say_to_it("Evet", looking=((5, new),))
        self.assertEqual(self.store.name(new), "Ahmet")

    def test_a_known_name_is_not_taken_twice(self):
        new = self.meet()
        self.c._ask_name(self.c._todo.get()[1])
        self.say_to_it("Selim", looking=((5, new),))
        self.assertIn("zaten tanıyorum", self.voice.said[-2])
        self.assertIn("bir daha", self.voice.said[-1])

    def test_answers_after_the_wait_are_not_answers(self):
        new = self.meet()
        self.c._ask_name(self.c._todo.get()[1])
        self.now[0] += config.CONV_ANSWER_S + 1
        self.say_to_it("Ahmet", looking=((5, new),))
        self.assertIsNone(self.store.name(new))
        self.assertNotIn("Ahmet, doğru mu?", self.voice.said)

    def test_beni_unut_forgets_nobody(self):                    # Selim 2026-11-01: never by itself
        self.say_to_it("Arkbi beni unut")
        self.say_to_it("evet")
        self.assertIn(self.selim, self.store.index)
        self.assertNotIn("emin misin", " ".join(self.voice.said))

    def test_a_sentence_started_in_the_conversation_counts_though_it_ends_later(self):
        self.say_to_it("Merhaba")
        start = self.c.session["until"] - 1                  # starts inside the session...
        self.now[0] = start + 6                                # ...processed after it (a long sentence + STT)
        self.stt.answers.append("napıyorsun")
        self.c._utterance(np.zeros(8000, np.float32), start, start + 5, scene=self.c._scene(start, start + 5))
        self.assertEqual(len(self.stt.calls), 2)

    def test_looking_is_taken_when_the_sentence_was_said(self):
        self.c.update([track(1, self.selim)], [], self.now[0])
        self.c.heard(np.zeros(8000, np.float32), self.now[0], self.now[0] + 0.5, onset=self.now[0] + 0.2)
        self.now[0] += 20                                      # the worker was busy; nobody looks any more
        self.c.update([], [], self.now[0])
        self.stt.answers.append("Merhaba")
        self.c._utterance(*self.c._todo.get()[1:])
        self.assertEqual(self.voice.said[-1], "Merhaba Selim!")

    def test_the_conversation_partner_expires(self):
        self.say_to_it("Merhaba")
        self.assertEqual(self.c.partner(), self.selim)
        self.now[0] += config.CONV_SESSION_S + 1
        self.assertIsNone(self.c.partner())

    def test_its_name_alone(self):
        self.say_to_it("Arkbi")
        self.assertEqual(self.voice.said, ["Efendim, Selim?"])


class NameFixTest(unittest.TestCase):
    def test_the_misheard_robot_name(self):
        from robot.conversation import fix_name
        self.assertEqual(fix_name("Nasılsın Arif Bey?"), "Nasılsın Arkbi?")
        self.assertEqual(fix_name("Efendim, Arif Bey?"), "Efendim, Arkbi?")
        self.assertEqual(fix_name("Ark bi bana bak"), "Arkbi bana bak")
        self.assertEqual(fix_name("Ark, saat kaç?"), "Arkbi, saat kaç?")
        self.assertEqual(fix_name("Arka üstündekini görebiliyor musun?"), "Arka üstündekini görebiliyor musun?")
        self.assertEqual(fix_name("Ark'ta sınıfta"), "Ark'ta sınıfta")
        self.assertEqual(fix_name("Arif geldi", known=["Arif"]), "Arif geldi")    # a real Arif at home


class OwnWordsTest(unittest.TestCase):
    def test_its_own_words_at_the_start_are_taken_out(self):          # talked over, 2026-11-01
        from robot.conversation import strip_own_words, fix_name
        said = ["<chuckle> İyiyim Selim, teşekkür ederim.", "Sen nasılsın?"]
        self.assertEqual(strip_own_words("İyiyim Selim. Hava nasıl?", said), "Hava nasıl?")
        self.assertEqual(strip_own_words("İyiyim Selim.", said), "")
        self.assertEqual(strip_own_words("Hava nasıl?", said), "Hava nasıl?")
        self.assertEqual(fix_name("Arka nasılsın?"), "Arkbi nasılsın?")
        self.assertEqual(fix_name("Arka bahçeye bak"), "Arka bahçeye bak")

    def test_only_its_words_in_its_order_not_theirs(self):           # scenario simulator 2026-11-01
        from robot.conversation import strip_own_words
        said = ["Bugün hava çok güzel, dışarı çıkabilirsin."]
        self.assertEqual(strip_own_words("güzel, dışarı çıkabilirsin. Hava yarın da güzel mi?", said),
                         "Hava yarın da güzel mi?")                  # their "Hava" is kept
        self.assertEqual(strip_own_words("Bugün çok güzel. Peki yarın?", said), "Peki yarın?")   # STT missed one
        self.assertEqual(strip_own_words("Doğru.", ["Doğu, doğru mu?"]), "")                     # all its own

    def test_a_word_it_said_twice_is_tried_at_every_place(self):              # review 2026-11-01
        from robot.conversation import strip_own_words
        said = ["Selim, bugün hava güzel.", "Selim, yarın yağmur var."]
        self.assertEqual(strip_own_words("Selim, yarın yağmur var. Peki ya cuma?", said), "Peki ya cuma?")


class NoiseTest(unittest.TestCase):
    def test_sounds_are_not_answered_words_are(self):
        from robot.conversation import is_noise
        for t in ("Ahem", "Ahem ahem.", "Hmmmm", "Hapşu!", "Hahaha", "şu", "Eee...", "Mhm.", "[öksürük]", ""):
            self.assertTrue(is_noise(t), t)
        for t in ("Evet", "Hayır", "Tamam", "Merhaba", "Saat kaç?", "Arkbi", "Şu kitabı ver"):
            self.assertFalse(is_noise(t), t)


class WordsTest(unittest.TestCase):
    def test_names_and_words(self):
        self.assertEqual(extract_name("Benim adım ahmet."), "Ahmet")
        self.assertEqual(extract_name("ismim irem"), "İrem")
        self.assertEqual(extract_name("Ben Selim"), "Selim")
        self.assertEqual(extract_name("Irmak"), "Irmak")
        self.assertEqual(extract_name("My name is John"), "John")
        self.assertEqual(extract_name("Evet"), None)
        self.assertEqual(plain_words("Arcby, Hayır!"), ["arkbi", "hayir"])

    def test_what_is_not_a_name(self):                         # all heard on the robot or in the review probe
        for text in ("Ne yapıyorsun?", "Yapıyorsun", "Saat kaç?", "Bilmiyorum", "Ne diyor? Bu 12 gün.",
                     "Bir dakika bekle", "Dur"):
            self.assertIsNone(extract_name(text), text)

    def test_yes_is_a_whole_yes(self):
        self.assertTrue(is_yes(plain_words("Evet, doğru")))
        self.assertTrue(is_yes(plain_words("Tamam")))
        self.assertFalse(is_yes(plain_words("Doğru mu bilmiyorum")))
        self.assertFalse(is_yes(plain_words("Evetten çok hayır")))
        self.assertFalse(is_yes(plain_words("Hayır, değil")))


class MindConversationTest(ConversationTest):
    """With a mind (every rule test above runs again, the mind giving no answer): free talk, silence, failures."""

    def setUp(self):
        super().setUp()
        self.mind = FakeMind()
        self.c.mind = self.mind

    def test_free_talk_is_answered_by_the_mind_with_the_scene(self):
        self.mind.answers.append("Bugün sakin bir gün.")
        self.c.update([track(2, None, looking=False)], [], self.now[0])
        self.say_to_it("Bugün nasıl geçti?")
        self.assertEqual(self.voice.said, ["Bugün sakin bir gün."])
        text, scene = self.mind.asked[0]
        self.assertEqual(scene["speaker"], "Selim")
        self.assertTrue(scene["looked"])
        self.assertEqual((scene["visible"], scene["strangers"]), (["Selim"], 0))

    def test_said_to_someone_else_stays_quiet_and_does_not_keep_talking(self):
        self.mind.answers.append("")
        self.say_to_it("Emre, yemeğe gel!")
        self.assertEqual(self.voice.said, [])
        self.assertIsNone(self.c.session)
        self.assertIn("not_for_me", [e["kind"] for e in self.c.events()])

    def test_no_answer_from_the_mind_falls_back_to_the_rules(self):
        self.say_to_it("Saat kaç?")                          # the mind gives None
        self.assertTrue(self.voice.said[0].startswith("Saat "))

    def test_goodbye_closes_the_conversation(self):
        self.mind.answers.append("Görüşürüz Selim, iyi geceler.")
        self.say_to_it("İyi geceler Arkbi")
        self.assertIsNone(self.c.session)


class FakeVoices:
    """A voice-print store that answers what it is told to."""

    def __init__(self, answer=(None, 0.0)):
        self.answer, self.learnt, self.forgotten, self.own = answer, [], [], {}

    def scores(self, vec):
        return dict(self.own)

    def identify(self, vec, among=None):
        return self.answer

    def learn(self, pid, vec):
        self.learnt.append(pid)
        return True

    def count(self, pid):
        return self.learnt.count(pid)

    def forget(self, pid):
        self.forgotten.append(pid)


class VoiceConversationTest(unittest.TestCase):
    """Who speaks, from the voice: learnt only from seen speakers, used when nobody is seen, sent to the head."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        rng = np.random.default_rng(0)
        self.selim = self.store.enrol([rng.normal(size=8) / 3])
        self.store.set_name(self.selim, "Selim")
        self.emre = self.store.enrol([rng.normal(size=8) / 3])
        self.store.set_name(self.emre, "Emre")
        self.voice, self.stt, self.mind, self.voices = FakeVoice(), FakeStt(), FakeMind(), FakeVoices()
        self.now = [100.0]
        self.c = Conversation(self.voice, self.stt, self.store, self.mind, self.voices, voice_emb=object(),
                              clock=lambda: self.now[0], wall=lambda: 1793490000.0, start=False, sleep=lambda s: None)

    def tearDown(self):
        self.dir.cleanup()

    def say(self, text, tracks, vid=None, seconds=1.5):
        self.c.update(tracks, [], self.now[0])
        self.stt.answers.append(text)
        self.c._utterance(np.zeros(int(seconds * 16000), np.float32), self.now[0], self.now[0] + seconds,
                          vid, np.ones(4, np.float32))
        self.now[0] += 2

    def test_the_looking_face_did_not_say_it_when_the_voice_is_someone_elses(self):     # Selim 2026-11-01
        self.say("Bugün ne yapsam", [track(1, self.emre), track(2, self.selim, looking=False)], vid=self.selim)
        self.assertEqual(self.mind.asked, [])                      # Emre looked, Selim spoke at his computer
        self.assertEqual(self.stt.calls, [])                       # nothing sent
        dropped = [e for e in self.c.events() if e["kind"] == "dropped"]
        self.assertEqual(dropped[0]["person"], self.selim)
        self.assertEqual(self.voices.learnt, [])                   # Selim's voice never learnt as Emre's

    def test_an_unsure_voice_clearly_more_someone_elses_is_not_the_looking_face(self):    # Selim 2026-11-01
        self.voices.learnt = [self.emre] * 10 + [self.selim] * 20       # both have prints
        self.voices.own = {self.selim: 0.47, self.emre: 0.26}           # "Su çarptı." - under 0.5: nobody's for sure
        self.say("Su çarptı.", [track(1, self.emre), track(2, self.selim, looking=False)])
        self.assertEqual(self.mind.asked, [])                          # not taken as Emre's
        dropped = [e for e in self.c.events() if e["kind"] == "dropped"]
        self.assertEqual(dropped[0]["reason"], "voice not the looking face")
        self.assertEqual(self.voices.learnt.count(self.emre), 10)      # never learnt as Emre's voice

    def test_the_looking_face_speaking_stays_theirs(self):
        self.voices.learnt = [self.emre] * 10 + [self.selim] * 20
        self.voices.own = {self.selim: 0.30, self.emre: 0.38}           # Emre's own, unsure
        self.mind.answers.append("Tamam.")
        self.say("Bugün ne yapsam", [track(1, self.emre)])
        heard = [e for e in self.c.events() if e["kind"] == "heard"]
        self.assertEqual(heard[-1]["person"], self.emre)

    def test_in_a_conversation_with_the_voice_it_is_theirs(self):
        self.say("Merhaba", [track(2, self.selim)])                 # Selim talks with the robot...
        self.mind.answers.append("Tamam.")
        self.say("Bir şey soracağım", [track(1, self.emre)], vid=self.selim)   # ...goes on while Emre looks
        heard = [e for e in self.c.events() if e["kind"] == "heard"]
        self.assertEqual(heard[-1]["person"], self.selim)

    def test_two_looking_is_not_the_conversation_partner_by_default(self):     # Meltem, 2026-11-01
        meltem = self.store.enrol([np.random.default_rng(5).normal(size=8) / 3])
        self.store.set_name(meltem, "Meltem")
        self.mind.answers.append("Merhaba.")
        self.say("Merhaba", [track(1, self.selim)])                 # Selim talks with the robot
        self.mind.answers.append("Hoş geldiniz.")
        self.say("Merhaba canım, hoş buldum", [track(1, self.selim), track(2, meltem)])   # Meltem came, both look
        heard = [e for e in self.c.events() if e["kind"] == "heard"]
        self.assertIsNone(heard[-1]["person"])                     # not "Selim" because Selim's talk was open
        self.assertEqual(sorted(self.mind.asked[-1][1]["speaker"].split(" ya da ")), ["Meltem", "Selim"])
        self.assertTrue(self.mind.asked[-1][1]["unsure"])

    def test_a_face_recognised_late_counts_as_who_it_was(self):
        self.c.update([track(7, None)], [], self.now[0])           # looks, not recognised yet
        self.now[0] += 1.5
        self.c.update([track(7, self.emre, looking=False)], [], self.now[0])    # ...now it is Emre
        self.stt.answers.append("Merhaba")
        self.c._utterance(np.zeros(24000, np.float32), self.now[0] - 1.6, self.now[0], None, np.ones(4, np.float32),
                          scene=self.c._scene(self.now[0] - 1.6, self.now[0], self.now[0] - 1.5))
        heard = [e for e in self.c.events() if e["kind"] == "heard"]
        self.assertEqual(heard[-1]["person"], self.emre)

    def test_a_side_remark_to_the_partner_is_not_for_the_robot(self):         # Selim 2026-11-01
        self.mind.answers.append("Merhaba Emre Bey.")
        self.say("Merhaba", [track(1, self.emre)])                 # Emre talks with the robot
        n = len(self.mind.asked)
        self.say("Emre, yemeğe gel", [track(1, self.emre, looking=False)], vid=self.selim)   # Selim, to Emre
        self.assertEqual(len(self.mind.asked), n)                  # not answered...
        dropped = [e for e in self.c.events() if e["kind"] == "dropped"]
        self.assertEqual(dropped[-1]["reason"], "a side remark, not to the robot")
        self.assertEqual(self.c.partner(), self.emre)              # ...and the talk with Emre goes on
        self.mind.answers.append("Tabii.")
        self.say("Bir şey daha soracağım", [track(1, self.emre, looking=False)], vid=self.emre)
        self.assertEqual(len(self.mind.asked), n + 1)

    def test_a_voice_unlike_the_partners_is_not_theirs(self):                # TikTok on full volume
        self.voices.count = lambda pid: 20
        self.voices.scores = lambda vec: {self.selim: 0.1}
        self.mind.answers.append("Merhaba.")
        self.say("Merhaba", [track(1, self.selim)])
        n = len(self.mind.asked)
        self.say("Bu videoyu beğenmeyi unutmayın", [track(1, self.selim, looking=False)])
        self.assertEqual(len(self.mind.asked), n)

    def test_a_voice_is_learnt_from_the_one_person_seen_saying_it(self):
        self.say("Merhaba", [track(1, self.selim)])
        self.assertEqual(self.voices.learnt, [self.selim])
        self.assertIn("voice_learnt", [e["kind"] for e in self.c.events()])

    def test_not_learnt_when_they_looked_away_while_someone_else_spoke(self):
        t = self.now[0]
        for i in range(10):                                         # Selim looks as it starts, then turns away
            self.c.update([track(1, self.selim, looking=i < 2)], [], t + i * 0.15)
        self.stt.answers.append("Merhaba")
        self.c._utterance(np.zeros(24000, np.float32), t, t + 1.5, None, np.ones(4, np.float32))
        self.assertEqual(self.voices.learnt, [])

    def test_not_learnt_when_it_sounds_unlike_their_own_prints(self):
        self.voices.own = {self.selim: 0.2}                          # nobody's voice, and far from Selim's
        self.say("Merhaba", [track(1, self.selim)])
        self.assertEqual(self.voices.learnt, [])
        self.voices.own = {self.selim: 0.45}
        self.say("Merhaba", [track(1, self.selim)])
        self.assertEqual(self.voices.learnt, [self.selim])

    def test_an_unseen_voice_does_not_take_the_conversation_over(self):
        self.mind.answers += ["Merhaba Selim.", "Tamam."]
        self.say("Merhaba", [track(1, self.selim)])
        self.say("Bir şey soracağım", [], vid=self.emre)            # the session goes on; by voice: Emre
        self.assertEqual(self.c.session["person"], self.selim)

    def test_not_learnt_when_unsure_who_spoke(self):
        self.say("Merhaba", [track(1, self.selim), track(2, None)])          # someone else looks too
        self.say("Merhaba", [track(1, self.selim)], vid=self.emre)           # the voice is clearly Emre's
        self.say("Evet", [track(1, self.selim)], seconds=0.4)                # too short to be a good print
        self.assertEqual(self.voices.learnt, [])

    def test_the_voice_tells_the_mind_who_speaks_when_nobody_is_seen(self):
        self.say("Merhaba", [track(1, None)], vid=self.emre)                # a stranger's face looks...
        self.mind.answers.append("Merhaba Emre.")
        self.say("Nasılsın?", [], vid=self.emre)                            # ...the session goes on, unseen
        text, scene = self.mind.asked[-1]
        self.assertEqual((scene["speaker"], scene["by_voice"]), ("Emre", True))

    def test_its_name_is_an_event_for_the_head(self):
        self.say("Arkbi, bak", [track(1, self.selim)])
        called = [e for e in self.c.events() if e["kind"] == "called"]
        self.assertEqual(called[0]["person"], self.selim)

    def test_whom_it_talks_with(self):
        self.assertIsNone(self.c.partner())
        self.mind.answers.append("İyiyim.")
        self.say("Nasılsın?", [track(1, self.selim)])
        self.assertEqual(self.c.partner(), self.selim)

    def test_the_voices_thread_then_the_conversation_thread(self):
        class Emb:
            def embed(self, pcm):
                return np.ones(4, np.float32)
        self.voices.answer = (self.emre, 0.7)
        c = Conversation(self.voice, self.stt, self.store, self.mind, self.voices, voice_emb=Emb())
        c.heard(np.zeros(16000, np.float32), 0.0, 1.0)
        c.close()                                       # both threads finish what was queued
        kinds = [(e["kind"], e.get("person")) for e in c.events()]
        self.assertEqual(kinds, [("voice", self.emre), ("dropped", self.emre)])

    def test_a_dropped_sentence_still_says_whose_voice_it_was(self):          # local only: the head can turn
        self.say("...", [], vid=self.emre)
        dropped = [e for e in self.c.events() if e["kind"] == "dropped"]
        self.assertEqual((dropped[0]["person"], self.stt.calls), (self.emre, []))


class StreamVoice(FakeVoice):
    """A voice that knows its cache and its hold() (busy while a reply is still written)."""

    def __init__(self, cached=()):
        super().__init__()
        self.cache, self.holds, self.marks = set(cached), [], []

    def say(self, text, level=None, mark=None):
        self.said.append(text)
        self.marks.append(mark)
        if mark is not None:
            mark.update(made=1.0, play=1.0, src="cache" if text in self.cache else "gemini")

    def cached(self, text):
        return text in self.cache

    def hold(self):
        import contextlib
        voice = self

        @contextlib.contextmanager
        def held():
            voice.holds.append("on")
            yield
            voice.holds.append("off")
        return held()


class StreamMind:
    """Writes its reply sentence by sentence, as the real streamed mind does."""

    def __init__(self, *sentences):
        self.sentences, self.seen = list(sentences), []

    def reply(self, text, on_sentence=None, **scene):
        for s in self.sentences:
            self.seen.append(list(self.voice.said))       # what had been said when it wrote this one
            on_sentence(s)
        return " ".join(self.sentences)


class FastReplyTest(unittest.TestCase):
    """Delay from the end of a sentence to the robot's voice: parallel voice-ID, streamed mind, fillers."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        self.selim = self.store.enrol([np.random.default_rng(0).normal(size=8) / 3])
        self.store.set_name(self.selim, "Selim")
        self.now = [100.0]

    def tearDown(self):
        self.dir.cleanup()

    def conv(self, voice, mind, filler_p=1.0, **kw):
        self.voice, self.stt = voice, FakeStt()
        mind.voice = voice
        return Conversation(voice, self.stt, self.store, mind, clock=lambda: self.now[0], wall=lambda: 1793490000.0,
                            start=False, sleep=lambda s: None, filler_p=filler_p, **kw)

    def say(self, c, text):
        c.update([track(1, self.selim)], [], self.now[0])
        self.stt.answers.append(text)
        c._utterance(np.zeros(16000, np.float32), self.now[0], self.now[0] + 1)
        self.now[0] += 2

    def test_sentence_one_is_spoken_while_the_mind_still_writes_and_one_said_event(self):
        mind = StreamMind("İyiyim.", "Sen nasılsın?")
        c = self.conv(StreamVoice(cached=["Hmm."]), mind, filler_p=0)
        self.say(c, "Nasılsın?")
        self.assertEqual(mind.seen, [[], ["İyiyim."]])          # sentence 2 written after 1 went to the voice
        self.assertEqual(self.voice.said, ["İyiyim.", "Sen nasılsın?"])
        self.assertEqual(self.voice.holds, ["on", "off"])
        said = [e for e in c.events() if e["kind"] == "said"]
        self.assertEqual([(e["text"], e["by"]) for e in said], [("İyiyim. Sen nasılsın?", "mind")])

    def test_a_barge_in_while_the_reply_still_plays_marks_it_cut(self):    # scenario simulator 2026-11-01
        voice = StreamVoice()
        voice.busy = True                                       # the reply plays...
        voice.interrupt = lambda: setattr(voice, "busy", False)
        c = self.conv(voice, StreamMind("Bir.", "İki."), filler_p=0)
        c._sleep = lambda s: c.barge_in()                       # ...and Selim (looking) talks over it
        self.say(c, "Anlat")
        said = [e for e in c.events() if e["kind"] == "said"]
        self.assertTrue(said[-1].get("cut"))                    # the mind had written it all before the barge-in

    def test_a_filler_first_when_the_reply_still_has_to_be_made(self):
        c = self.conv(StreamVoice(cached=["Hmm.", "Hmm, bakalım."]), StreamMind("Bakayım.", "Tamam."))
        self.say(c, "Bugün ne yaptın?")
        self.assertIn(self.voice.said[0], ("Hmm.", "Hmm, bakalım."))
        self.assertEqual(self.voice.said[1:], ["Bakayım.", "Tamam."])

    def test_no_filler_for_a_ready_reply_nor_one_that_would_cost_a_request(self):
        c = self.conv(StreamVoice(cached=["Hmm.", "Merhaba Selim."]), StreamMind("Merhaba Selim."))
        self.say(c, "Merhaba")
        self.assertEqual(self.voice.said, ["Merhaba Selim."])     # cached: plays at once anyway
        c = self.conv(StreamVoice(cached=[]), StreamMind("Güzel."))
        self.say(c, "Bak")
        self.assertEqual(self.voice.said, ["Güzel."])            # no filler is cached: none made now

    def test_fillers_vary_and_can_be_switched_off(self):
        import random
        fillers = ["Hmm.", "Hmm, bakalım.", "Şey..."]
        c = self.conv(StreamVoice(cached=fillers), StreamMind("Tamam."), rand=random.Random(3))
        firsts = []
        for _ in range(6):
            self.voice.said.clear()
            self.say(c, "Bir şey soracağım")
            firsts.append(self.voice.said[0])
        self.assertTrue(all(a != b for a, b in zip(firsts, firsts[1:])))     # never the same twice in a row
        c = self.conv(StreamVoice(cached=fillers), StreamMind("Tamam."), filler_p=0)
        self.say(c, "Bir şey soracağım")
        self.assertEqual(self.voice.said, ["Tamam."])

    def test_english_gets_an_english_filler(self):
        c = self.conv(StreamVoice(cached=["Let me see.", "Şey..."]), StreamMind("Sure."))
        self.say(c, "What is your name?")
        self.assertEqual(self.voice.said[0], "Let me see.")

    def test_a_timing_line_per_answered_utterance(self):
        c = self.conv(StreamVoice(), StreamMind("Tamam."))
        with self.assertLogs("robot.conversation", "INFO") as logs:
            self.say(c, "Merhaba")
        line = [l for l in logs.output if "timing" in l]
        self.assertEqual(len(line), 1)
        for part in ("stt done", "mind 1st sentence", "1st audio", "(gemini)", "playing"):
            self.assertIn(part, line[0])

    def test_the_rules_are_timed_too(self):
        c = self.conv(StreamVoice(), StreamMind())
        c.mind = None
        with self.assertLogs("robot.conversation", "INFO") as logs:
            self.say(c, "Saat kaç?")
        self.assertIn("1st audio", [l for l in logs.output if "timing" in l][0])
        self.assertIsNotNone(self.voice.marks[0])


class ParallelVoiceTest(unittest.TestCase):
    """The STT does not wait for the voice recognition (only the 'who speaks' decision does)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        rng = np.random.default_rng(0)
        self.selim = self.store.enrol([rng.normal(size=8) / 3])
        self.store.set_name(self.selim, "Selim")
        self.emre = self.store.enrol([rng.normal(size=8) / 3])
        self.store.set_name(self.emre, "Emre")

    def tearDown(self):
        self.dir.cleanup()

    def test_stt_runs_while_the_voice_is_still_being_recognised(self):
        import threading
        stt_started, order = threading.Event(), []

        class Emb:
            def embed(self, pcm):
                stt_started.wait(3)                # finishes only once the STT has started
                order.append("embed done")
                return np.ones(4, np.float32)

        class Stt(FakeStt):
            def transcribe(self, pcm, prompt=""):
                order.append("stt")
                stt_started.set()
                return "Merhaba"
        voices = FakeVoices((self.emre, 0.7))
        mind = FakeMind("Merhaba Emre Bey.")
        c = Conversation(FakeVoice(), Stt(), self.store, mind, voices, voice_emb=Emb())
        c.update([track(1, None)], [], time.monotonic())        # a stranger's face looks: it is sent
        c.heard(np.zeros(16000, np.float32), time.monotonic() - 1, time.monotonic())
        c.close()
        self.assertEqual(order, ["stt", "embed done"])
        self.assertEqual(mind.asked[0][1]["speaker"], "Emre")    # the voice still decides who it was

    def test_nobody_looking_and_only_the_voice_can_tell_the_answer_it_waits_for_the_voice_first(self):
        from robot.conversation import _VoiceResult
        stt = FakeStt()
        c = Conversation(FakeVoice(), stt, self.store, None, FakeVoices(), voice_emb=object(),
                         clock=lambda: 100.0, start=False, sleep=lambda s: None)
        c.expect = dict(kind="name", person=self.selim, until=200.0)
        pending = _VoiceResult()
        pending.set(self.emre, np.ones(4, np.float32))           # Emre's voice, nobody in view
        scene = c._scene(99.0, 100.0)
        c._utterance(np.zeros(8000, np.float32), 99.0, 100.0, scene=scene, pending=pending)
        self.assertEqual(stt.calls, [])                          # not Selim's answer: never sent
        self.assertEqual(c.dropped, 1)


class MuteVoiceTest(unittest.TestCase):
    """alive.py --mute: ears + mind on, voice off - replies only as text (2026-11-02, the speaker broken)."""

    def setUp(self):
        from robot.tts import MuteVoice
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        rng = np.random.default_rng(0)
        self.selim = self.store.enrol([rng.normal(size=8) / 3])
        self.store.set_name(self.selim, "Selim")
        self.voice, self.stt, self.now = MuteVoice(), FakeStt(), [100.0]
        self.c = Conversation(self.voice, self.stt, self.store, FakeMind("Selam Selim, buradayım."),
                              clock=lambda: self.now[0], wall=lambda: 1793490000.0, start=False, sleep=lambda s: None)

    def tearDown(self):
        self.dir.cleanup()

    def test_replies_become_text_only(self):
        self.c.update([track(1, self.selim)], [], self.now[0])
        self.stt.answers.append("Napıyorsun?")
        with self.assertLogs("robot.tts", level="INFO") as logs:
            self.c._utterance(np.zeros(8000, np.float32), self.now[0], self.now[0] + 1)
        said = [e for e in self.c.events() if e["kind"] == "said"]
        self.assertTrue(said and "buradayım" in said[-1]["text"])          # the dashboard sees it
        self.assertTrue(any("would say: Selam Selim" in line for line in logs.output))
        self.assertFalse(self.voice.busy)                                   # the ears are never muted by it
        self.assertTrue(self.voice.spoken)                                  # its own words known (own_echo)

    def test_the_voice_interface(self):
        mark = {}
        self.voice.say("Merhaba", mark=mark)
        self.assertEqual(mark["src"], "mute")
        self.assertEqual(self.voice.prepare(["a", "b"]), ["mute", "mute"])
        self.assertTrue(self.voice.cached("Hmm."))
        with self.voice.hold():
            pass
        self.assertEqual(self.voice.usage()[1], 0)


class ShowingTest(unittest.TestCase):
    """robot/show.py + the conversation: asked ("bu ne?") sends the picture with the question; unasked showing is a
    turn of its own."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        rng = np.random.default_rng(0)
        self.selim = self.store.enrol([rng.normal(size=8) / 3])
        self.store.set_name(self.selim, "Selim")
        self.voice, self.stt, self.now = FakeVoice(), FakeStt(), [100.0]
        self.mind = FakeMind("Bir telefon görüyorum.")
        self.c = Conversation(self.voice, self.stt, self.store, self.mind, clock=lambda: self.now[0],
                              wall=lambda: 1793490000.0, start=False, sleep=lambda s: None)
        self.asked = []
        outer = self

        class Show:
            def ask(self, t, person=None):
                outer.asked.append(person)

            def picture(self):
                return b"\xff\xd8pic", "cell phone"
        self.c.show = Show()

    def tearDown(self):
        self.dir.cleanup()

    def test_asked_what_this_is_looks_and_sends_the_picture(self):
        self.c.update([track(1, self.selim)], [], self.now[0])
        self.stt.answers.append("Bu ne?")
        self.c._utterance(np.zeros(8000, np.float32), self.now[0], self.now[0] + 1)
        self.assertEqual(self.asked, [self.selim])
        _, scene = self.mind.asked[0]
        self.assertEqual(scene["image"], b"\xff\xd8pic")
        self.assertEqual(scene["shown"], {"what": "cell phone", "asked": True})
        self.assertEqual(self.voice.said[-1], "Bir telefon görüyorum.")

    def test_a_plain_sentence_sends_no_picture(self):
        self.c.update([track(1, self.selim)], [], self.now[0])
        self.stt.answers.append("Saat kaç?")
        self.c._utterance(np.zeros(8000, np.float32), self.now[0], self.now[0] + 1)
        self.assertEqual(self.asked, [])

    def test_shown_unasked_is_answered_and_opens_a_talk(self):
        self.c.shown(dict(kind="showing", what="cell phone", person=self.selim, asked=False))
        text, scene = self.mind.asked[0]
        self.assertEqual((text, scene["speaker"], scene["shown"]["asked"]), ("", "Selim", False))
        self.assertEqual(self.voice.said, ["Bir telefon görüyorum."])
        self.assertIsNotNone(self.c.session)
        self.assertIn("shown", [e["kind"] for e in self.c.events()])

    def test_shown_while_it_speaks_waits_for_nothing(self):
        self.voice.busy = True
        self.c.shown(dict(kind="showing", what="cell phone", person=self.selim, asked=False))
        self.assertEqual(self.mind.asked, [])


if __name__ == "__main__":
    unittest.main()


class FakeLegs:
    def __init__(self, active=True):
        self.active, self.stops, self.starts = active, [], []

    def stop(self, why):
        was, self.active = self.active, False
        self.stops.append(why)
        return was

    def start(self, mode, why="", max_s=None):
        self.starts.append((mode, why, max_s))
        self.active = True
        return True, ""


class DriveWordsTest(unittest.TestCase):
    """robot/legs.py inside alive.py: "dur" stops a drive (anyone, looking or not), "Arcby gez" starts one."""
    tearDown = ConversationTest.tearDown
    say_to_it = ConversationTest.say_to_it

    def setUp(self):
        ConversationTest.setUp(self)
        self.legs = self.c.drive = FakeLegs()

    def test_dur_while_driving_stops_even_when_nobody_looks(self):
        self.say_to_it("Arcby dur!", looking=())
        self.assertEqual(len(self.legs.stops), 1)
        self.assertIn("dur", self.legs.stops[0])
        self.assertEqual(self.voice.said, ["Durdum."])
        self.assertIn("drive_stop", [e["kind"] for e in self.c.events()])

    def test_other_talk_while_driving_is_transcribed_but_never_answered(self):
        self.say_to_it("Bugün akşam ne yiyoruz?", looking=())
        self.assertEqual(len(self.stt.calls), 1)
        self.assertEqual(self.voice.said, [])
        self.assertEqual(self.legs.stops, [])
        self.assertFalse(any(e["kind"] == "heard" for e in self.c.events()))

    def test_not_driving_the_listening_rule_holds(self):
        self.legs.active = False
        self.say_to_it("Dur!", looking=())
        self.assertEqual(self.stt.calls, [])
        self.assertEqual(self.legs.stops, [])

    def test_hear_all_off_dur_works_only_when_said_to_it(self):
        old = config.DRIVE_STOP_HEAR_ALL
        config.DRIVE_STOP_HEAR_ALL = False
        self.addCleanup(setattr, config, "DRIVE_STOP_HEAR_ALL", old)
        self.say_to_it("Dur!", looking=())
        self.assertEqual(self.stt.calls, [])
        self.say_to_it("Dur!")
        self.assertEqual(len(self.legs.stops), 1)

    def test_its_own_durdum_is_not_a_stop_word(self):
        self.say_to_it("Durdum.", looking=())
        self.assertEqual(self.legs.stops, [])

    def test_arif_gez_starts_a_wander(self):
        self.legs.active = False
        self.say_to_it("Arcby biraz gez")
        self.assertEqual(self.legs.starts[0][0], "explore")
        self.assertEqual(self.legs.starts[0][2], config.WANDER_MAX_S)
        self.assertEqual(self.voice.said[-1], "Tamam, biraz geziyorum.")

    def test_gez_in_a_sentence_without_the_name_is_just_talk(self):
        self.legs.active = False
        self.say_to_it("Dün parkta gezdik")
        self.assertEqual(self.legs.starts, [])
