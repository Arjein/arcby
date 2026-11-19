"""
Conversation: what the robot does with what it hears. Free talk goes to the mind (Claude, robot/mind.py); meeting a
new face and every answer when the mind cannot be reached are rule-based. The robot never forgets anyone by itself
(Selim, 2026-11-01: no "beni unut" by voice) - people_cli.py forget is the only way.

Listening rule (privacy, Selim 2026-11-01): an utterance goes to speech-to-text only when someone LOOKED at the robot
as the sentence STARTED (Track.looking within CONV_GAZE_START_S of the speech onset) - Selim too. After that the
conversation stays open (CONV_SESSION_S since the last exchange, counted to the START of the next sentence): people
may turn their heads and keep talking. A question the robot asked is answered only by the person it asked. Everything
else is dropped unheard. No wake word: looking at the robot is how you talk to it; its name works too then.

    conv = Conversation(voice, stt, store, mind, voices, voice_emb)  # mind=None: rules only; voices: who speaks
    conv.update(tracks, events, t)      # camera loop: who looks at the robot; a new face looking -> ask the name
    conv.heard(pcm, t0, t1, onset)      # ears callback (any thread): the scene is noted NOW, the rest is queued
    for e in conv.events(): ...         # voice / called / heard / said / met / dropped (recorder, attention)
    conv.partner()                      # who it is talking with now (or None)
    conv.drive = legs                   # robot/legs.py (alive.py, battery on): "dur" stops a drive, "Arcby gez" starts one
    conv.close()

While it drives, a stop word ("dur", "Arcby dur", "yeter", "stop") from anyone stops it at once - a local rule, never
the mind - and with DRIVE_STOP_HEAR_ALL every sentence said during a drive goes to speech-to-text for that (nobody
looks at a robot driving with its head down at the floor); such a sentence is used for the stop word only, never
answered.

Who speaks: the face looking at the robot as the sentence started; else the voice (robot/voices.py, local - every
utterance, also the ones never sent anywhere, so the head can turn to a voice: a "voice" event); else whom it was
talking with. A voice is learnt only from sentences whose speaker it SAW the whole time (one known person looking,
nobody else) and whose sound is not someone else's.

Two worker threads, so the camera loop never waits: "voices" (who is it, ~0.4 s) and "conversation" (Gemini STT,
the mind, speaking) work on the same utterance AT ONCE - only the "who speaks" decision after the STT waits for the
voice (and, when nobody looked and only the voice can say whether it is the answer to a question, the decision to
send it at all). The mind's reply is streamed: its first sentence is being spoken while the second is being made.
A reply that is not cached (Gemini TTS ~3 s) often starts with a short cached filler ("Hmm, bakalım.", made at
start-up). Speech goes through Voice.say (one queue with the greeter) and the worker waits until it is spoken.
Every answered utterance logs one "timing:" line (seconds after the end of speech).
"""
import collections
import contextlib
import difflib
import logging
import queue
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import config
from .bargein import prune
from .stt import wav_bytes

log = logging.getLogger(__name__)

# yes / no: whole words only ("doğru mu bilmiyorum" is not a yes, "hayırlı olsun" not a no)
YES = {"evet", "dogru", "aynen", "tabii", "tabi", "tamam", "evt", "he", "yes", "yeah", "yep", "correct", "right"}
NO = {"hayir", "yanlis", "degil", "yok", "no", "nope", "wrong"}
UNSURE = {"mi", "mu", "bilmiyorum", "bilmem", "emin", "galiba", "sanirim", "belki", "maybe", "not"}
NAME_WORD = "arkbi"                 # the robot's name (Arcby, said "Arkbi") as plain_words gives it
WAKE = ("arkbi", "arcby", "arkby", "arkibi")
BYE = ("gorusuruz", "hoscakal", "hosca", "geceler")
STOP_WORDS = {"dur", "durr", "durrr", "dursana", "durun", "durdur", "stop", "yeter", "dur dur"}   # whole words only
WANDER_WORDS = ("gez", "dolas")     # "Arcby gez" / "Arcby biraz dolaş": explore for a while (with the name only)
FILLER = {"benim", "ben", "adim", "ismim", "isim", "evet", "merhaba", "selam", "ad", "bu", "o", "sey", "efendim",
          "hayir", "degil", "yanlis", "imm", "hmm", "ee", "eee", "aa", "yani", "tamam", "ne", "ki", "arkbi",
          "hello", "hi", "my", "name", "is", "i", "am", "im", "it", "s", "call", "me", "the", "a", "yes", "no",
          "ok", "okay", "well", "oh", "uh", "um", "peki", "hadi", "iste", "sen", "siz", "da", "de", "ve"}
# words that are never a name: question words, common words, short commands
NOT_NAMES = {"ne", "neden", "nasil", "kim", "kac", "nerede", "nereye", "niye", "niçin", "hangi", "saat", "bugun",
             "yarin", "dun", "simdi", "iyi", "kotu", "guzel", "bak", "dur", "gel", "git", "yap", "soyle", "anlat",
             "tesekkurler", "tesekkur", "sagol", "nasilsin", "napiyorsun", "robot", "kimse", "hic", "bir", "iki",
             "uc", "what", "who", "why", "how", "when", "where", "thanks", "thank", "you", "nothing", "nobody",
             "anne", "baba", "abi", "abla", "kardes", "mi", "mu", "var", "yok", "olur", "olmaz", "lutfen"}
# Turkish verb / question endings: "yapıyorsun", "bilmiyorum", "gelecek", "yapmak", "misin"
VERB_ENDS = ("yor", "yorum", "yorsun", "yoruz", "yorlar", "acagim", "ecegim", "misin", "musun", "miyim", "muyum",
             "iyorum", "iyorsun")           # not -mak / -dim / -lim: Irmak, Selim are names
TR_UPPER = str.maketrans("iıçğöşü", "İIÇĞÖŞÜ")
# said at once while a reply is still being made (only when cached: made at start-up, never at run time)
FILLERS = {"tr": ("Hmm.", "Hmm, bakalım.", "Şey...", "<breath> Bir saniye.", "Bir bakayım."),
           "en": ("Hmm.", "Let me see.", "<breath> One moment.")}


class _VoiceResult:
    """Who the voice is (from the "voices" thread), for the conversation thread that is already transcribing."""

    def __init__(self):
        self._done = threading.Event()
        self.vid, self.vec, self.at = None, None, None

    def set(self, vid, vec):
        self.vid, self.vec, self.at = vid, vec, time.monotonic()
        self._done.set()

    def get(self, timeout=10.0):
        self._done.wait(timeout)
        return self.vid, self.vec


# what the live STT (no word list) makes of "Arkbi" - not a word, so it writes the nearest it knows (2026-11-01:
# "Nasılsın Arif Bey?", "Efendim, Arif Bey?", "Ark bir sınıfta..."). Nobody at home is called Arif; "Ark" alone is
# no Turkish word ("arka" = back, left alone)
_NAME_FIXES = [("arif", re.compile(r"\bAr(?:i|ı)f\s+Bey\b", re.I), "Arkbi"),
               ("alp", re.compile(r"\bAlp\s+Bey\b", re.I), "Arkbi"),
               ("arif", re.compile(r"\bAr(?:i|ı)f\b(?!['’])", re.I), "Arkbi"),
               ("ark", re.compile(r"\bArk\s+bi\b", re.I), "Arkbi"),
               ("ark", re.compile(r"\bArk\b(?!['’\w])", re.I), "Arkbi"),
               # "Arka nasılsın?" - "arka" (back) is a word, but not before a greeting / question at the start
               ("arka", re.compile(r"^\s*Arka(?=[\s,]+(?:nas[ıi]l|naber|bak|beni|bana|sen|saat|ne|dur|merhaba|"
                                   r"selam|hey|s[öo]yle|anlat|duy|g[öo]r)\w*)", re.I), "Arkbi")]


def fix_name(text, known=()):
    """The robot's name as the STT misheard it -> "Arkbi" (not when a known person really has that name)."""
    if not text:
        return text
    known = {plain_words(n)[0] for n in known if n and plain_words(n)}
    for name, rx, to in _NAME_FIXES:
        if name not in known:
            text = rx.sub(to, text)
    return text


def strip_own_words(text, said, skip=2):
    """Heard text without the robot's own words at its start (its voice still in the mic as someone talked over
    it): "İyiyim Selim. Hava nasıl?" while it said "İyiyim Selim, teşekkür ederim" -> "Hava nasıl?". Only words in
    the order it said them (up to `skip` of its words missed by the STT in between): "güzel. Hava yarın?" after
    "Bugün hava çok güzel." keeps their "Hava" (any word it said lately was taken out before - scenario simulator).
    Every place a word was said is tried ("Selim" twice in its sentences, review 2026-11-01). A first word of theirs
    that the robot also said is still taken out ("Hava yarın nasıl?" after "Bugün hava çok güzel." -> "yarın
    nasıl?"): the STT mostly catches exactly one word of the robot's tail ("teşekkür Hava nasıl peki?",
    "veriyorum, Dur bir dakika" - scenario simulator 11b / 13a), the commoner case."""
    own = [w for t in said for w in plain_words(t)]
    if not own or not text:
        return text
    tokens = text.split()
    words = [plain_words(t) for t in tokens]
    where = collections.defaultdict(list)   # word -> every place in its own words
    for j, o in enumerate(own):
        where[o].append(j)

    def places(ws, at):
        """The last place of each way to find the words ws in its own words, in order, after `at`."""
        ends = [at]
        for w in ws:
            ends = {j for e in ends for j in where.get(w, ()) if e is None or e < j <= e + 1 + skip}
            if not ends:
                return ()
        return ends

    memo = {}

    def longest(i, at):                     # how many tokens from i on are its own words, in its order
        if i >= len(tokens) or not words[i]:
            return 0
        if (i, at) not in memo:
            memo[i, at] = max((1 + longest(i + 1, e) for e in places(words[i], at)), default=0)
        return memo[i, at]
    n = longest(0, None)
    return " ".join(tokens[n:]).lstrip(",.;:!? ")


def asked_to_look(text):
    from .show import asked_to_look as asked     # show.py uses plain_words from here
    return asked(text)


def own_echo(text, said, fuzz=None):
    """Was every word heard one the robot just said (or nearly: the STT on its distorted echo makes "çağırma" of
    "çağıramam", "Merhaba Selim." of "Merhaba Selim!")? Then nobody talked over it - the barge-in was its own voice
    (2026-11-02: every reply cut, its own words answered back in a loop). Substrings count from 3 letters on."""
    fuzz = config.CONV_ECHO_FUZZ if fuzz is None else fuzz
    words = plain_words(text or "")
    own = {w for t in said for w in plain_words(t)}
    if not words or not own:
        return False

    def like(w):
        return w in own or any((len(w) >= 3 and w in o) or difflib.SequenceMatcher(None, w, o).ratio() >= fuzz
                               for o in own)
    return all(like(w) for w in words)


def plain_words(text):
    """Turkish text -> lower-case ASCII words ("Hayır!" -> ["hayir"]); the robot's name in one spelling."""
    t = text.replace("I", "ı").replace("İ", "i").lower().translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    words = "".join(c if "a" <= c <= "z" else " " for c in t).split()
    return [NAME_WORD if w in WAKE else w for w in words]


# sounds that are not words: coughs, sneezes, throat-clearing, laughs, hesitations, half-words (as plain_words gives
# them). Nothing is answered for them (Selim 2026-11-01: "Ahem" got a reply). Not "evet" / "hayir" / "tamam".
NOISE_WORDS = {"ahem", "ehem", "hm", "hmm", "mhm", "mm", "ım", "im", "ee", "e", "eh", "a", "aa", "ah", "oh", "o", "of",
               "off", "uf", "ooo", "ay", "vay", "hapsu", "hapsuu", "hapsiu", "hapsir", "hapsirik", "aksiru", "atsu",
               "achoo", "atchoo", "haha", "hahaha", "hehe", "hihi", "kih", "ohu", "ohoo", "ohho", "ehe", "ihi",
               "sey", "su", "bu", "ya", "yani", "ii", "ı", "uh", "um", "uhm", "huh", "pff", "tsk", "cik", "ck", "hik"}


def is_noise(text):
    """Only coughs / sneezes / laughs / hesitations (or a single half-word like "şu"): nothing to answer. A sound
    the STT described in brackets ("[öksürük]", "(gülüşme)") is no word either."""
    words = plain_words(re.sub(r"[\[(<][^\])>]*[\])>]", " ", text))
    if not words:
        return True
    squeezed = [re.sub(r"(.)\1+", r"\1\1", w) for w in words]      # "hmmmm" -> "hmm", "ahemmm" -> "ahemm"
    return all(w in NOISE_WORDS or w.rstrip(w[-1]) + w[-1] in NOISE_WORDS or re.fullmatch(r"(h[ae])+h?", w)
               for w in squeezed)


def has(words, *keys):
    """A word starting with one of keys (commands: "teşekkürler" has "tesekkur")."""
    return any(w.startswith(k) for w in words for k in keys)


def is_yes(words):
    w = set(words)
    return bool(w & YES) and not (w & NO) and not (w & UNSURE)


def _name_like(word, plain):
    return (len(plain) >= 2 and plain not in FILLER and plain not in NOT_NAMES
            and not any(plain.endswith(e) and len(plain) > len(e) + 1 for e in VERB_ENDS))


def extract_name(text):
    """'Benim adım ahmet.' / 'Adım Ahmet' / 'Ahmet' / 'My name is John' -> the name; None if no name-like word
    (a question, a verb, a common word: "Ne yapıyorsun?", "Saat kaç?", "Bilmiyorum")."""
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşüâîû]+", text)
    plain = plain_words(" ".join(words))
    if len(plain) != len(words) or not words:
        return None
    pick = None
    for key in (("adim",), ("ismim",), ("ben",), ("name", "is"), ("i", "am"), ("im",), ("call", "me")):
        for i in range(len(plain) - len(key)):
            if tuple(plain[i:i + len(key)]) == key:
                pick = i + len(key)
                break
        if pick is not None:
            break
    if pick is None:
        if "?" in text or any(p in ("mi", "mu", "ne", "kim", "nasil", "kac") for p in plain):
            return None                     # a question back, not a name
        rest = [i for i, p in enumerate(plain) if p not in FILLER]
        if len(rest) != 1:                  # a name answer is one word (plus fillers): "Ahmet", "Ben Ahmet"
            return None
        pick = rest[0]
    if pick >= len(words) or not _name_like(words[pick], plain[pick]):
        return None
    low = words[pick].replace("I", "ı").replace("İ", "i").lower()
    return low[0].translate(TR_UPPER).upper() + low[1:]


class _Stream:
    """A sentence on its way to the live STT: state None = undecided (speech kept in buf), True = streaming
    (session), False = not said to the robot (nothing sent)."""

    def __init__(self, onset, pcm):
        self.onset, self.buf, self.state, self.session = onset, [pcm], None, None

    def cancel(self):
        if self.session is not None:
            self.session.cancel()


class Conversation:
    def __init__(self, voice, stt, store, mind=None, voices=None, voice_emb=None, clock=time.monotonic,
                 wall=time.time, answer_s=config.CONV_ANSWER_S, ask_again_s=config.CONV_ASK_NAME_AGAIN_S, start=True,
                 sleep=time.sleep, rand=None, filler_p=config.CONV_FILLER_P, save_dir=None, live=None):
        self.voice, self.stt, self.store, self.mind = voice, stt, store, mind
        self.live = live                    # robot/live_stt.py LiveSTT: the sentence streams while it is said
        self._stream = None                 # the sentence being heard now (ears thread) -> _Stream
        self._cut = 0                       # barge-ins so far: a reply still being written stops at the next one
        self._cut_at = -1e9                 # when the last barge-in was (time.monotonic())
        self._cut_clock = -1e9              # ...and on the conversation's clock (as the ears time utterances)
        self._recent_said = collections.deque(maxlen=6)     # (time, sentence) the robot spoke lately
        self.on_false_barge = None          # () -> dB added: robot/bargein.py BargeIn.false_alarm (alive.py)
        self.show = None                    # robot/show.py ShowSense (alive.py): "bak" / "bu ne" + pictures
        self.drive = None                   # robot/legs.py Legs (alive.py, battery on): stop words, "Arcby gez"
        self._streams = {}                  # onset -> _Stream of sentences that ended, until the worker takes them
        self.save_dir = save_dir            # None: utterances are not kept (tests)
        self._rand, self.filler_p, self._last_filler = rand or random.Random(), filler_p, None
        self._mark = None                   # timing of the utterance being answered (its first spoken phrase)
        self.voices, self.voice_emb = voices, voice_emb
        self._clock, self._wall, self._sleep = clock, wall, sleep
        self.answer_s, self.ask_again_s = answer_s, ask_again_s
        self._lock = threading.Lock()
        self._track_person = {}             # track id -> the person it was last recognised as
        self._frames = collections.deque()  # (frame time, {track id: person} looking at the robot) - every frame
        self._seen = ([], 0)                # faces in view now: (names of the people it knows, strangers)
        self.session = None                 # an open conversation: {"person", "until"}
        self.expect = None                  # what the robot waits for: {"kind", "person", "until", ...}
        self._asked = {}                    # person id -> when its name was last asked
        self._ask_pending = None            # person id whose name question is queued
        self._events = []
        self._todo = queue.Queue()
        self._voice_q = queue.Queue()
        self.dropped = 0                    # utterances not sent (nobody looking, no question)
        self._threads = []
        if start:
            for name, run in (("conversation", self._run), ("voices", self._run_voices)):
                self._threads.append(threading.Thread(target=run, name=name, daemon=True))
                self._threads[-1].start()
            if filler_p > 0 and hasattr(voice, "prepare"):      # cached once, then free for good
                threading.Thread(target=self.prepare_fillers, name="filler-prepare", daemon=True).start()

    def prepare_fillers(self):
        try:
            texts = list(dict.fromkeys([config.TTS_LIMIT_PHRASE, "Efendim?"]
                                       + [t for group in FILLERS.values() for t in group]))
            done = self.voice.prepare(texts)
            log.info("conversation: fillers ready: %d of %d", sum(1 for d in done if d != "silent"), len(texts))
        except Exception:  # noqa: BLE001 - no fillers is no reason to stop
            log.exception("conversation: preparing the fillers failed")

    # ---------- from the camera loop and the ears ----------

    def update(self, tracks, events, t):
        """Who looks at the robot now; an unnamed face that looks at it for CONV_ASK_LOOK_S gets its name asked."""
        with self._lock:
            self._frames.append((t, {tr.id: tr.person for tr in tracks if tr.looking}))
            for tr in tracks:               # who each track turned out to be (a new face is known ~1-2 s later)
                if tr.person:
                    self._track_person[tr.id] = tr.person
            while self._frames and t - self._frames[0][0] > config.CONV_LOOK_KEEP_S:
                self._frames.popleft()
            names = [self.store.name(tr.person) for tr in tracks if tr.person]
            self._seen = (sorted({n for n in names if n}), sum(1 for tr in tracks if not self.store.name(tr.person)))
            busy = self._busy()
        if busy:
            return
        best = {}                           # unnamed person -> longest-looking track (two tracks can share an id)
        for tr in tracks:
            pid = tr.person
            if not pid or self.store.name(pid) or self._is_picture(pid) or not tr.looking:
                continue
            look = tr.looked_for(t) if hasattr(tr, "looked_for") else config.CONV_ASK_LOOK_S
            if look >= config.CONV_ASK_LOOK_S and look > best.get(pid, -1.0):
                best[pid] = look
        for pid in sorted(best, key=lambda p: -best[p]):
            if t - self._asked.get(pid, -1e9) < self.ask_again_s:
                continue
            self._asked[pid] = t
            self._ask_pending = pid
            self._todo.put(("ask_name", pid))
            break

    # ---------- live speech to text (robot/mic.py Ears.listener; the ears thread) ----------

    def speech_started(self, onset, pcm):
        if self.live is None:
            return
        self._stream = _Stream(onset, pcm)
        self._decide(self._stream)

    def speech_chunk(self, pcm):
        s = self._stream
        if s is None:
            return
        if s.session is not None:
            s.session.feed(pcm)
        elif s.state is None:
            s.buf.append(pcm)
            self._decide(s)

    def speech_ended(self, kept):
        s, self._stream = self._stream, None
        if s is None:
            return
        if not kept:                        # a click / its own voice: never an utterance
            if s.session is not None:
                s.session.cancel()
            return
        if s.state is None:
            self._decide(s, final=True)
        if s.session is not None:
            s.session.end()
        with self._lock:
            now = self._clock()
            for k in [k for k in self._streams if now - k > 60]:     # never taken (the worker was gone)
                self._streams.pop(k).cancel()
            self._streams[s.onset] = s

    def _decide(self, s, final=False):
        """Stream this sentence? Only when it is said to the robot - someone looked at it as it started, or a
        conversation / question is open - like the clip STT: nothing else ever leaves the robot. Undecided until
        CONV_GAZE_START_S after the onset (a look may still come); the speech heard meanwhile is sent at once."""
        now, w = self._clock(), config.CONV_GAZE_START_S
        with self._lock:
            looked = any(look for ft, look in self._frames if s.onset - w <= ft <= s.onset + w)
        sess, exp = self.session, self.expect
        open_ = (sess is not None and s.onset <= sess["until"]) or (exp is not None and s.onset <= exp["until"])
        if looked or open_ or self._drive_hears():
            s.session = self.live.open()
            s.state = s.session is not None
            if s.session is not None:
                for piece in s.buf:
                    s.session.feed(piece)
            s.buf = []
        elif final or now > s.onset + w:
            s.state, s.buf = False, []

    # ---------- barge-in (robot/bargein.py via robot/mic.py; the ears thread) ----------

    def barge_in(self):
        """Someone talks while the robot speaks: stop and listen - only when someone looks at the robot or a
        conversation is open (a door, a TV, a knock in the room must not cut it off). True = it stopped,
        "listen" = it had finished speaking already: listen now, but nothing was cut off (not a barge-in), False = no."""
        now = self._clock()
        with self._lock:
            looking = any(look for ft, look in self._frames if now - ft <= config.BARGE_LOOK_S)
        sess, exp = self.session, self.expect
        if not (looking or (sess is not None and now <= sess["until"]) or (exp is not None and now <= exp["until"])):
            return False
        if not getattr(self.voice, "busy", True):
            # it had finished speaking (the ears were in their hold after its voice): a quick answer - listen now,
            # but nothing was cut off: its own words are not taken out of it ("Doğru." to "Doğu, doğru mu?" became
            # "" -> "Efendim?", scenario simulator 2026-11-01)
            log.info("conversation: an answer right after the robot spoke - listening")
            return "listen"
        self._cut += 1
        self._cut_at = time.monotonic()
        self._cut_clock = now               # the same moment on the ears' clock (an utterance's t0 / t1)
        interrupt = getattr(self.voice, "interrupt", None)
        if interrupt is not None:
            interrupt()
        self._event(kind="interrupted")
        log.info("conversation: interrupted - someone talks over the robot, listening")
        return True

    def _driving(self):
        d = self.drive
        return bool(d is not None and getattr(d, "active", False))

    def _drive_hears(self):
        """While driving, every sentence is transcribed - for a stop word only (DRIVE_STOP_HEAR_ALL)."""
        return config.DRIVE_STOP_HEAR_ALL and self._driving()

    def _drive_stop_word(self, text):
        """A stop word in a sentence heard while driving: stop now (no mind, any voice). True if it stopped."""
        words = plain_words(text or "")
        if not (set(words) & STOP_WORDS) or not self._driving():
            return False
        stopped = self.drive.stop(f"voice: {text.strip()!r}")
        self._event(kind="drive_stop", text=text, by="voice")
        log.info("conversation: %r while driving - stopped", text)
        if stopped:
            self._say("Durdum.", wait=False)
        return True

    def _wander(self, text, who):
        """'Arcby gez': explore for WANDER_MAX_S."""
        ok, why = self.drive.start("explore", why=f"voice: {text.strip()!r}", max_s=config.WANDER_MAX_S)
        self._event(kind="drive_start" if ok else "drive_refused", text=text, person=who, why=why)
        self._say("Tamam, biraz geziyorum." if ok else "Şu an gezemem.")

    def _take_stream(self, onset):
        with self._lock:
            return self._streams.pop(onset, None) if onset is not None else None

    def heard(self, pcm, t0, t1, onset=None, **_):
        """An utterance from the ears (t0 = start incl. pre-roll, onset = first speech chunk). Who looked, the open
        conversation and the question asked are taken NOW, as the sentence was said - not when the worker gets to it."""
        scene = self._scene(t0, t1, onset)
        scene["arrived"] = time.monotonic()
        pending = None
        if self.voice_emb is not None and self.voices is not None:
            pending = _VoiceResult()            # who speaks: worked out next to the STT, not before it
            self._voice_q.put((pcm, t0, t1, scene, pending))
        self._todo.put(("utterance", pcm, t0, t1, None, None, scene, pending))

    def partner(self):
        """Whom it is talking with now: whom it waits for an answer from, or the open conversation's person."""
        now = self._clock()
        exp, sess = self.expect, self.session
        if exp is not None and exp.get("person") and now <= exp["until"]:
            return exp["person"]
        return sess["person"] if sess and now <= sess["until"] else None

    def events(self):
        with self._lock:
            out, self._events = self._events, []
        return out

    def close(self):
        if self._threads:                   # the voices thread first: it hands its last utterances on
            conv, voices = self._threads
            self._voice_q.put(None)
            voices.join(timeout=10)
            self._todo.put(None)
            conv.join(timeout=10)

    # ---------- what was going on as a sentence started ----------

    def _busy(self):
        """Talking or waiting for an answer (or about to ask): no new name question now."""
        now = self._clock()
        exp, sess = self.expect, self.session   # read once: the conversation thread may set them to None meanwhile
        return (self._ask_pending is not None or (exp is not None and now <= exp["until"])
                or (sess is not None and now <= sess["until"]))

    def _is_picture(self, pid):
        p = self.store.index.get(pid)
        return bool(isinstance(p, dict) and p.get("picture"))

    def _scene(self, t0, t1, onset=None):
        """{who, looked, sure, whole, session, expect, seen, onset}: who = the one known person looking at the robot as
        the sentence started (onset +- CONV_GAZE_START_S), sure = theirs was the only face looking, whole = they
        looked (and nobody else) through the whole sentence."""
        onset = t0 if onset is None else onset
        w = config.CONV_GAZE_START_S
        with self._lock:
            frames = list(self._frames)
            seen = self._seen
        with self._lock:
            latest = dict(self._track_person)
        # a face not recognised yet as the sentence started is taken as who it turned out to be (Meltem's first
        # sentence went to Selim, 2026-11-01)
        near = {(tid, p or latest.get(tid)) for ft, look in frames if onset - w <= ft <= onset + w
                for tid, p in look.items()}
        known = {p for _, p in near if p}
        who = next(iter(known)) if len(known) == 1 else None
        sure = who is not None and len({tid for tid, _ in near}) == 1
        during = [look for ft, look in frames if onset <= ft <= t1]
        whole = sure and bool(during) and \
            sum(1 for look in during if set(look.values()) == {who}) >= config.CONV_WHOLE_LOOK * len(during)
        exp, sess = self.expect, self.session
        return dict(who=who, looked=bool(near), sure=sure, whole=whole, onset=onset, seen=seen, driving=self._driving(),
                    lookers=sorted(known), others_looked=len({tid for tid, _ in near}) > (1 if who else 0),
                    session=dict(sess) if sess is not None and onset <= sess["until"] else None,
                    expect=dict(exp) if exp is not None and onset <= exp["until"] else None)

    def shown(self, event):
        """robot/show.py: someone holds something up to the robot, unasked (it is sure) - the mind reacts, in turn
        with what is being heard. Asked ("bak", "bu ne") goes through the sentence itself (_mind_reply)."""
        if event.get("asked"):
            return
        if self._threads:
            self._todo.put(("shown", event))
        else:
            self._shown(event)

    def _wait_picture(self):
        """The picture robot/show.py takes of what is shown (the next camera frames), or None after a short wait."""
        show = self.show
        if show is None:
            return None
        deadline = time.monotonic() + config.CONV_PICTURE_WAIT_S
        while True:
            pic = show.picture()
            if pic is not None or time.monotonic() >= deadline:
                return pic
            self._sleep(0.05)

    def _shown(self, event):
        if self.mind is None or getattr(self.voice, "busy", False):
            return                          # nothing to say it with / it is talking: not now
        pic = self._wait_picture()
        who = event.get("person")
        names, strangers = self._seen
        with self._hold():
            reply = self.mind.reply("", speaker=self.store.name(who) if who else None, looked=True, visible=names,
                                    strangers=strangers, image=pic[0] if pic else None,
                                    shown=dict(what=event.get("what"), asked=False))
        if reply:
            self._say(reply, by="mind")
            self._keep_talking(who, seen_face=who is not None)   # they may answer without looking again
        self._event(kind="shown", what=event.get("what"), person=who, picture=pic is not None, answered=bool(reply))

    # ---------- the worker ----------

    def _run(self):
        while True:
            item = self._todo.get()
            if item is None:
                return
            try:
                if item[0] == "ask_name":
                    self._ask_name(item[1])
                elif item[0] == "shown":
                    self._shown(item[1])
                else:
                    self._utterance(*item[1:])
            except Exception:  # noqa: BLE001 - one bad turn must not end the conversation thread
                log.exception("conversation: %s failed", item[0])

    def _known_ids(self):
        lock = getattr(self.store, "_lock", None)
        if lock is None:
            return set(self.store.index)
        with lock:
            return set(self.store.index)

    def _run_voices(self):
        while True:
            item = self._voice_q.get()
            if item is None:
                return
            pcm, t0, t1, scene, pending = item
            vid, vsim, vec = None, 0.0, None
            try:
                vec = self.voice_emb.embed(pcm)
                vid, vsim = self.voices.identify(vec, among=self._known_ids())
                self._event(kind="voice", person=vid, sim=round(vsim, 3), seconds=round(len(pcm) / 16000, 2))
            except Exception:  # noqa: BLE001 - not knowing the voice is no reason to stop listening
                log.exception("conversation: voice recognition failed")
            finally:
                pending.set(vid, vec)

    def _save(self, pcm, kind, text=""):
        """Keep the utterance as a WAV (CONV_SAVE_DIR, the newest CONV_SAVE_N) - what the robot heard, to listen to.
        Never stops the conversation."""
        if not self.save_dir:
            return
        try:
            d = Path(self.save_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            words = re.sub(r"[^\w]+", "-", (text or "").lower()).strip("-")[:40]
            name = time.strftime("%Y%m%d-%H%M%S") + f"-{kind}" + (f"-{words}" if words else "") + ".wav"
            (d / name).write_bytes(wav_bytes(pcm))
            prune(d, config.CONV_SAVE_N)        # by time written (names sorted wrong across days)
        except Exception as e:             # a full disk must not stop the robot hearing
            log.warning("conversation: could not save the utterance: %s", e)

    def _event(self, **e):
        e.setdefault("t", round(self._clock(), 4))
        with self._lock:
            self._events.append(e)

    def _said_lately(self, seconds):
        """What the robot said in the last `seconds`: its own sentences and fillers, and whatever else its voice
        said (the greeter's "Merhaba Selim!" - heard back as Selim's "Merhaba Selim.", 2026-11-02)."""
        now = time.monotonic()
        out = [t for at, t in self._recent_said if now - at < seconds]
        for at, t in list(getattr(self.voice, "spoken", ())):
            if now - at < seconds and t not in out:
                out.append(t)
        return out

    def _false_barge(self):
        """Tell the barge-in detector its last alarm was its own echo (it predicts those bands louder from now)."""
        cb = self.on_false_barge
        if cb is None:
            return
        try:
            add = cb()
            if add is not None and len(add) and max(add) > 0:
                log.info("barge-in: own echo - bands predicted louder by %s dB",
                         " ".join(f"{a:.0f}" for a in add))
        except Exception:  # noqa: BLE001 - tuning must never stop the conversation
            log.exception("conversation: telling the barge-in detector failed")

    def _say(self, text, by="conversation", mark=None, wait=True):
        """Speak (one queue with the greeter) and wait until it is spoken."""
        self._recent_said.append((time.monotonic(), text))
        if by is not None:
            self._event(kind="said", text=text, by=by)
            if mark is None and self._mark is not None and not self._mark:
                mark = self._mark               # the first thing said for this utterance: its timing
        if mark is not None:
            self.voice.say(text, mark=mark)
        else:
            self.voice.say(text)
        if wait:
            self._wait_spoken()

    def _wait_spoken(self):
        deadline = self._clock() + 30           # Voice.say counts the phrase at once: busy from now on
        while self.voice.busy and self._clock() < deadline:
            self._sleep(0.02)

    def _hold(self):
        hold = getattr(self.voice, "hold", None)
        return hold() if hold is not None else contextlib.nullcontext()

    def _filler(self, first, heard):
        """A short cached "Hmm." before a reply whose first sentence still has to be made (~3 s of Gemini TTS) -
        not always, never the same twice in a row, never one that would cost a request."""
        cached = getattr(self.voice, "cached", None)
        if cached is None or self.filler_p <= 0 or cached(first) or self._rand.random() >= self.filler_p:
            return None
        from .tts import guess_lang
        pool = [f for f in FILLERS[guess_lang(heard)] if f != self._last_filler and cached(f)]
        if not pool:
            return None
        self._last_filler = f = self._rand.choice(pool)
        self._say(f, by=None, wait=False)
        return f

    def _wait_for(self, kind, **kw):
        self.expect = dict(kind=kind, until=self._clock() + self.answer_s, **kw)

    def _learn_voice(self, who, vid, vec, seconds, whole):
        """The face says who spoke: one more print of their voice - only when they were seen the whole sentence and
        the sound is theirs (or nobody's yet, and close enough to theirs). Selim, Emre: brothers, one mic."""
        if self.voices is None or vec is None or seconds < config.VOICE_MIN_S or not whole:
            return
        if vid is not None and vid != who:
            return
        if vid is None:
            scores = getattr(self.voices, "scores", None)
            own = scores(vec).get(who) if scores is not None else None
            if own is not None and own < config.VOICE_LEARN_SIM:
                return                      # has prints and this sounds unlike them: maybe someone off camera
        if self.voices.learn(who, vec):
            self._event(kind="voice_learnt", person=who, prints=self.voices.count(who))

    def _clearly_not(self, who, vec):
        """The known person whose voice this clearly is rather than `who`'s (the face looking), or None. For a voice
        too unsure to name (Selim's prints are only ~0.47 alike, under VOICE_MATCH_SIM): Selim talking while Emre
        looked scored Selim 0.47 / Emre 0.26 and was taken as Emre's (2026-11-01). Only when `who` has enough prints
        to compare with."""
        scores = getattr(self.voices, "scores", None)
        if vec is None or scores is None or self.voices.count(who) < config.VOICE_PARTNER_MIN_PRINTS:
            return None
        known = self._known_ids()
        s = {p: v for p, v in scores(vec).items() if p in known}
        if who not in s:
            return None
        best = max((p for p in s if p != who), key=s.get, default=None)
        if best is not None and s[best] >= config.VOICE_LEARN_SIM and s[best] - s[who] >= config.VOICE_OTHER_MARGIN:
            log.info("conversation: the voice is more %s's (%.2f) than %s's (%.2f)", self.store.name(best) or best,
                     s[best], self.store.name(who) or who, s[who])
            return best
        return None

    def _session_open(self, scene):
        """The conversation open as the sentence started (None: it had ended)."""
        if self.session is not None and self._clock() > self.session["until"] and scene["session"] is None:
            self._event(kind="session_end", person=self.session["person"])
            self.session = None
        return scene["session"]

    def _keep_talking(self, who, seen_face):
        """An exchange happened: the conversation stays open CONV_SESSION_S from now. Whom it is with changes only
        when a face says so (an unseen voice does not take the conversation over)."""
        prev = self.session["person"] if self.session else None
        person = who if (seen_face or prev is None) else prev
        self.session = dict(person=person, until=self._clock() + config.CONV_SESSION_S)

    def _utterance(self, pcm, t0, t1, vid=None, vec=None, scene=None, pending=None):
        if scene is None:                   # called directly (tests, old callers): the scene now
            scene = self._scene(t0, t1)
        tm = self._timing_start(t1, scene)

        def voice_id():                     # who the voice is - waits for the voices thread only when needed
            nonlocal vid, vec, pending
            if pending is not None:
                vid, vec = pending.get()
                tm["voice"] = pending.at
                pending = None
            return vid
        seconds = round(len(pcm) / 16000, 2)
        who, looked, sure = scene["who"], scene["looked"], scene["sure"]
        exp = scene["expect"]
        if exp is not None and self.expect is not None and self.expect.get("kind") == exp["kind"] \
                and self.expect.get("person") == exp["person"]:
            exp = self.expect               # still the question it asked (not answered by an earlier sentence)
        else:
            exp = None
        if exp is not None and exp.get("person") is not None and scene["who"] is None and not looked:
            voice_id()                      # only the voice can tell whether this is the answer
        if exp is not None and not self._from(exp.get("person"), scene, vid):
            exp = None                      # someone else spoke while it waited: not the answer
        sess = self._session_open(scene)
        other = None                        # the voice is clearly someone else's than the face that looked
        if who is not None and (pending is not None or vid is not None or vec is not None):
            v = voice_id()
            if v is not None and v != who:
                other = v
            elif v is None:                 # nobody's for sure - but maybe clearly more someone else's than theirs
                other = self._clearly_not(who, vec)
        if other is not None:
            # Emre looked at the robot while Selim, at his computer, spoke (2026-11-01): the face did not say it.
            # The speaker did not look - only an open conversation with them (or a question to them) keeps it
            partner = (sess or {}).get("person") if sess is not None else None
            asked = exp.get("person") if exp is not None else None
            log.info("conversation: %s looked, but the voice is %s's", self.store.name(who) or who,
                     self.store.name(other) or other)
            if other not in (partner, asked):
                exp, sess, looked = None, None, False
            who, sure, looked = None, False, False   # theirs by voice from here on (by_voice below)
            scene = dict(scene, who=None, sure=False, whole=False)
        side = None                         # in a conversation with Emre, Selim (not looking) says something to Emre
        if sess is not None and not looked and other is None and \
                (pending is not None or vid is not None or vec is not None):
            v = voice_id()
            partner = sess.get("person")
            asked = exp.get("person") if exp is not None else None
            known_partner = partner is not None and self.voices is not None and \
                self.voices.count(partner) >= config.VOICE_PARTNER_MIN_PRINTS
            own = None
            if v is None and known_partner and vec is not None and hasattr(self.voices, "scores"):
                own = self.voices.scores(vec).get(partner)
            if v is not None and partner is not None and v != partner and v != asked:
                side = v
                log.info("conversation: %s's voice while talking with %s, not looking at the robot - a side remark",
                         self.store.name(v) or v, self.store.name(partner) or partner)
            elif own is not None and own < config.VOICE_LEARN_SIM and partner != asked:
                # a voice clearly not the partner's (TikTok, the TV, a guest): not part of their talk
                side = "unknown"
                log.info("conversation: a voice unlike %s's (%.2f) while talking with them - not answered",
                         self.store.name(partner) or partner, own)
            if side is not None:
                exp, sess = None, None
        drive_only = exp is None and sess is None and not looked and bool(scene.get("driving")) \
            and config.DRIVE_STOP_HEAR_ALL      # not said to the robot, but it drives: only for a stop word
        if exp is None and sess is None and not looked and not drive_only:
            self.dropped += 1                   # not said to the robot: never sent anywhere
            self._mark = None
            self._event(kind="dropped", seconds=seconds, person=voice_id(),
                        **({"reason": "voice not the looking face"} if other is not None else
                           {"reason": "a side remark, not to the robot"} if side is not None else {}))
            self._save(pcm, "dropped")
            stream = self._take_stream(scene.get("onset"))
            if stream is not None:
                stream.cancel()
            return
        names = [n for _, n, _ in self.store.people() if n]
        text, stream = None, self._take_stream(scene.get("onset"))
        if stream is not None and stream.session is not None:
            text = stream.session.finish()  # streamed while it was said: ready ~0.4 s after the end
            tm["stt_by"] = "live" if text else f"clip ({stream.session.error or 'live heard nothing'})"
        if not text:                        # the voice is worked out meanwhile
            text = self.stt.transcribe(pcm, prompt=", ".join(names + ["Arkbi"]))
            tm.setdefault("stt_by", "clip")
        tm["stt"] = time.monotonic()
        text = fix_name(text, names)
        if self._drive_stop_word(text):     # first of all: "dur" while it drives
            self._timing_log(tm, "stop word while driving")
            return
        if drive_only:                      # heard only because it drives - not answered, not kept as heard
            self._event(kind="dropped", seconds=seconds, reason="heard while driving, no stop word")
            self._timing_log(tm, "while driving, not to the robot")
            return
        barged = time.monotonic() - self._cut_at < config.CONV_BARGE_WINDOW_S
        if barged and text:                 # talked over it: its own words still in the mic are not theirs
            said = self._said_lately(15)
            # its own words and nothing else: not answered - it has stopped already (all someone talking over it
            # wanted), and "Efendim?" here fed a loop: its own echo cut every reply, the words heard back were
            # answered, that answer was cut again (alive.log 2026-11-02 15:37-15:38). If the sound also ended with
            # its own voice (speakerd fades ~0.2 s, the room rings ~0.15 s, the mic lags 0.11 s) nobody was there:
            # a false barge-in - the detector learns that echo. Someone who goes on after it stops (scenario 11a)
            # teaches it nothing.
            if own_echo(text, said):
                echo_only = t1 - config.MIC_END_S - self._cut_clock <= config.CONV_ECHO_END_S
                log.info("conversation: %r was only the robot's own words - not answered%s", text,
                         " (a false barge-in: its own echo)" if echo_only else "")
                self._event(kind="own_echo", text=text, learnt=echo_only)
                self._save(pcm, "own")
                if echo_only:
                    self._false_barge()
                self._timing_log(tm, "only its own words after a cut")
                return
            mine = strip_own_words(text, said)
            if mine != text:
                log.info("conversation: %r was partly the robot's own voice -> %r", text, mine)
            text = mine
            if not text or is_noise(text):
                self._timing_log(tm, "talked over it, nothing of theirs understood - asking")
                self._say("Efendim?")
                self.expect = None
                return
        voice_id()
        if sure:
            self._learn_voice(who, vid, vec, seconds, scene["whole"])
        names_seen, strangers_seen = scene["seen"]
        # several people there and the voice did not say it was the face that looked: maybe someone else spoke
        scene["unsure"] = who is not None and vid != who and len(names_seen) + strangers_seen >= 2
        by_voice = who is None and vid is not None
        if by_voice:
            who = vid
        maybe = None                        # several looked as it started: one of them (the mind is told so)
        if who is None and len(scene.get("lookers", [])) > 1:
            maybe = scene["lookers"]
        elif who is None and sess is not None and not scene.get("others_looked"):
            who = sess["person"]            # their conversation goes on (they looked away) - nobody else looked
        scene["maybe"] = maybe
        self._event(kind="heard", text=text, person=who, seconds=seconds, error=getattr(self.stt, "last_error", None))
        self._save(pcm, "heard", text)
        log.info("conversation: heard %r (looking: %s)", text, self.store.name(who) if who else who)
        if not text:
            self._timing_log(tm, "nothing understood")
            return
        if exp is None and is_noise(text):      # a cough, "ahem", a sneeze, a laugh: not something to answer
            self._event(kind="noise", text=text, person=who)
            self._timing_log(tm, f"only a sound ({text!r}) - not answered")
            return
        words = plain_words(text)
        if NAME_WORD in words:                    # called by its name: the head turns to whoever said it
            self._event(kind="called", person=who)
        if exp is not None:
            self.expect = None
            self._answer(exp, text, words)
        else:
            rest = [w for w in words if w != NAME_WORD]
            reply = None
            # NOT opened at the start of the answer (tried 2026-11-01 for scenario 14): the barge-in detector still
            # fires on the robot's own echo, and an open conversation let it cut every first reply - the voice
            # sounded broken. The conversation opens after the reply (below) until barge-in is tuned.
            if not rest:                        # just its name
                call = self.store.call(who) if who else None
                self._say(f"Efendim{', ' + call if call else ''}?")
            elif self.drive is not None and NAME_WORD in words and has(rest, *WANDER_WORDS) and len(rest) <= 3:
                self._wander(text, who)
                self._keep_talking(who, seen_face=scene["who"] is not None)
                self._timing_log(tm, "wander")
                return
            elif self.mind is not None:
                cut = self._cut
                reply = self._mind_reply(text, who, looked, by_voice, scene, tm)
                if reply == "":                 # said to someone else: no answer, the conversation does not go on
                    self._event(kind="not_for_me", text=text, person=who)
                    self._timing_log(tm, "not for me")
                    return
                if not reply and self._cut != cut:      # talked over before it said a word: no rule answer now
                    self._keep_talking(who, seen_face=scene["who"] is not None)
                    self._timing_log(tm, "talked over before the reply")
                    return
            if rest and reply is None:          # no mind, or it could not answer: the rules
                bye = self._command(rest, text, who) == "bye"
            else:
                bye = bool(reply) and has(rest, *BYE)
            if bye:
                self.session = None
                self._event(kind="session_end", person=who)
                self._timing_log(tm)
                return
        self._keep_talking(who, seen_face=scene["who"] is not None)
        self._timing_log(tm)

    def _mind_reply(self, text, who, looked, by_voice, scene, tm):
        """The mind's reply, streamed: sentence 1 goes to the voice as soon as it is written (after a filler when
        it is not cached), sentence 2 is made while 1 plays. The voice stays busy (ears muted) until the end."""
        names, strangers = scene["seen"]
        spoken = []
        cut = self._cut

        def on_sentence(sentence):
            if self._cut != cut:                # talked over: the rest of the reply is not said (nor remembered)
                return False
            if not spoken:
                tm["mind"] = time.monotonic()
                tm["filler"] = self._filler(sentence, text)
                self._say(sentence, by=None, mark=tm["first"], wait=False)
            else:
                self._say(sentence, by=None, wait=False)
            spoken.append(sentence)
        with self._hold():                      # busy between the sentences while the mind is still writing
            maybe = scene.get("maybe")
            speaker = self.store.name(who) if who else \
                " ya da ".join(n for n in (self.store.name(p) for p in maybe) if n) if maybe else None
            extra = {}
            if self.show is not None and asked_to_look(text):     # "bak", "bu ne?": sure something is shown
                self.show.ask(self._clock(), person=who)
                pic = self._wait_picture()
                extra = dict(image=pic[0] if pic else None,
                             shown=dict(what=pic[1] if pic else None, asked=True))
            reply = self.mind.reply(text, speaker=speaker or None, looked=looked,
                                    visible=names, strangers=strangers, by_voice=by_voice, on_sentence=on_sentence,
                                    unsure=scene.get("unsure", False) or bool(maybe), **extra)
            if reply and not spoken and self._cut == cut:   # a mind that does not stream: the whole reply at once
                tm["mind"] = time.monotonic()
                tm["filler"] = self._filler(reply, text)
                self._say(reply, by=None, mark=tm["first"], wait=False)
                spoken.append(reply)
        if spoken:
            self._wait_spoken()             # first: a barge-in while it is still speaking cuts this reply too
        if reply and self._cut != cut:
            log.info("conversation: reply cut off after %d sentence(s): %r", len(spoken), " ".join(spoken))
        if reply:
            self._event(kind="said", text=reply, by="mind", **({"cut": True} if self._cut != cut else {}))
        return reply

    # ---------- timing: end of speech -> STT -> mind -> sound -> playing ----------

    def _timing_start(self, t1, scene):
        """Times in time.monotonic(). The end of speech = the utterance's end (the ears' clock is monotonic too)
        minus the silence that ended it; for any other clock (tests) its arrival minus that silence."""
        arrived = scene.get("arrived", time.monotonic())
        end = t1 if abs(arrived - t1) < 5 else arrived
        tm = {"end": end - config.MIC_END_S, "heard": arrived, "first": {}}
        self._mark = tm["first"]
        return tm

    def _timing_log(self, tm, outcome=None):
        self._mark = None

        def at(t):
            return f"{t - tm['end']:+.2f}" if t else "-"
        first = tm["first"]
        parts = [f"utterance in {at(tm['heard'])}", f"stt done {at(tm.get('stt'))} ({tm.get('stt_by', '-')})",
                 f"voice-id {at(tm.get('voice'))}"]
        if outcome:
            parts.append(outcome)
        else:
            parts += [f"mind 1st sentence {at(tm.get('mind'))}",
                      f"1st audio {at(first.get('made'))} ({first.get('src', '-')})",
                      f"playing {at(first.get('play'))}"]
            if tm.get("filler"):
                parts.append(f"filler {tm['filler']!r}")
        log.info("timing (s after end of speech): %s", ", ".join(parts))
    def _from(self, pid, scene, vid):
        """Did pid say it? Their face looking as the sentence started (the only one), or - nobody seen - their voice."""
        if pid is None:
            return True
        if scene["who"] is not None:
            return scene["who"] == pid and scene["sure"]
        return not scene["looked"] and vid == pid

    # ---------- answers to its own questions ----------

    def _ask_name(self, pid, again=False):
        if not again:
            self._ask_pending = None
            now = self._clock()
            still = any(pid in look.values() for ft, look in list(self._frames) if now - ft <= 1.5)
            if (pid not in self.store.index or self.store.name(pid) or self._is_picture(pid) or not still
                    or (self.session is not None and now <= self.session["until"])
                    or (self.expect is not None and now <= self.expect["until"])):
                return                      # gone, named, or talking with someone by now: not the moment
        self._say("Adını bir daha söyler misin?" if again else
                  "Merhaba! Ben Arcby. Seni daha önce görmedim. Adın ne?")
        self._wait_for("name", person=pid, tries=1 if again else 0)

    def _answer(self, exp, text, words):
        kind, pid = exp["kind"], exp.get("person")
        if kind == "name":
            name = extract_name(text)
            known = {(n or "").lower() for p, n, _ in self.store.people() if p != pid}
            if name and name.lower() in known:
                self._say(f"{name} adında birini zaten tanıyorum.")
                name = None
            if name:
                self._say(f"{name}, doğru mu?")
                self._wait_for("confirm_name", person=pid, name=name, tries=exp.get("tries", 0))
            elif exp.get("tries", 0) < 1:
                self._ask_name(pid, again=True)
            else:
                self._say("Tamam, sonra tekrar tanışırız.")
        elif kind == "confirm_name":
            if is_yes(words):
                if pid in self.store.index:
                    self.store.set_name(pid, exp["name"])
                self._event(kind="met", person=pid, name=exp["name"])
                self._say(f"Memnun oldum {exp['name']}! Seni artık tanıyacağım.")
            elif exp.get("tries", 0) < 1:
                self._ask_name(pid, again=True)
            else:
                self._say("Tamam, sonra tekrar tanışırız.")

    # ---------- what it can do when called ----------

    def _command(self, words, text, who):
        call = self.store.call(who) if who else None
        if has(words, "saat") and has(words, "kac"):
            now = datetime.fromtimestamp(self._wall(), ZoneInfo(config.TTS_TIMEZONE))
            self._say(f"Saat {now.hour}:{now.minute:02d}.")
        elif has(words, "kendini", "kimsin") or (has(words, "tanit") and not has(words, "beni")):
            self._say("Ben Arcby. Bu evde yaşayan küçük bir robotum. Yüzleri tanıyorum, Türkçe konuşuyorum, ve her "
                      "gün biraz daha öğreniyorum. Kahve yapamıyorum, henüz.")
        elif has(words, "nasilsin"):
            self._say(f"İyiyim, teşekkür ederim{' ' + call if call else ''}. Sen nasılsın?")
        elif has(words, "tesekkur", "sagol"):
            self._say("Rica ederim.")
        elif has(words, *BYE):
            self._say(f"Görüşürüz{' ' + call if call else ''}.")
            return "bye"
        elif has(words, "napiyorsun", "yapiyorsun"):
            self._say("Seni dinliyorum. Bir de etrafı izliyorum, her zamanki gibi.")
        elif has(words, "merhaba", "selam"):
            self._say(f"Merhaba{' ' + call if call else ''}!")
        else:
            self._say("Hmm, bunu henüz bilmiyorum. Ama öğreniyorum.")
