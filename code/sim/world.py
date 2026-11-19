"""
The simulated room around the real ears + conversation (robot/mic.py Ears, robot/bargein.py, robot/conversation.py
run unchanged). One clock, advanced 32 ms chunk by chunk:

    camera  -> conv.update(tracks)            every 2nd chunk (~15 fps), from the scenario's look schedule
    speaker -> SimVoice plays its queue; while it plays its echo is ADDED to the mic (closed loop) and registered in
               the EchoRef (what the barge-in detector predicts from)
    mic     -> ears._chunk(raw)               the mic buffer (people + robot echo + floor) as arecord bytes
    voices  -> the "voices" thread's step     (voice ID from the ground truth: whose sound fills the utterance)
    worker  -> the "conversation" thread      conv._utterance / _ask_name, run in place

The conversation worker blocks (STT, the mind writing, waiting until a phrase is spoken): every wait it makes goes
through `sleep` = World.advance, which keeps feeding the mic, the camera and the speaker meanwhile - so a barge-in
during a reply happens exactly as on the robot (the ears thread interrupts the worker's voice). Utterances heard
while the worker is busy queue up, as with the real single worker thread.
"""
import os
import collections
import contextlib
import re
import tempfile
import time as _time
import zlib
from types import SimpleNamespace

import numpy as np

from robot import config
from robot import conversation as conv_mod
from robot.bargein import BargeIn, EchoRef
from robot.conversation import Conversation
from robot.mic import CHUNK_S, Ears
from robot.people import PeopleStore

from . import audio

T0 = 1000.0                                 # sim time of mic sample 0
FRAME48 = 1536                              # one 32 ms chunk at 48 kHz
LAG_S = 0.108                               # speaker -> mic (measured; config.BARGE_LAG_S is what the robot assumes)


class EnergyVad:
    """Stand-in for Silero (robot/vad.py) when its ONNX model / onnxruntime is not on this machine: the speech
    probability from the level of the GAINED chunk (what Ears gives the VAD) - a logistic around mid_db. Unlike
    Silero it takes any loud sound (a cough, a knock, music) for speech: scenarios script what the STT makes of it."""

    name = "energy"

    def __init__(self, mid_db=-30.0, width_db=3.0):
        self.mid, self.width = mid_db, width_db

    def reset(self):
        pass

    def prob(self, chunk):
        x = np.asarray(chunk, np.float64)
        db = 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-12)
        return float(1 / (1 + np.exp(-(db - self.mid) / self.width)))


def load_vad():
    """EnergyVad - always, unless ARCBY_SIM_SILERO=1: the simulated voices are speech-LIKE buzz (sim/audio.py), and
    the real Silero hears no speech in them (on the Pi every scenario got 0 utterances, 2026-11-01; the Mac had no
    onnxruntime and always used the stand-in). The scenarios test the conversation logic, not the VAD."""
    if os.environ.get("ARCBY_SIM_SILERO"):
        from robot.vad import Vad
        return Vad()
    return EnergyVad()


class Source:
    """One stretch of sound in the room: a person's line, a cough, the robot's echo. text = what a speech-to-text
    writes for it (words spread evenly over its length)."""

    def __init__(self, who, text, start, seconds, kind, audio48=None, words=None):
        self.who, self.text, self.start, self.kind, self.audio = who, text, start, kind, audio48
        self.end = start + seconds
        self.cut_at = None
        if words is not None:               # [(seconds from the start, word)] as synthesised
            self.words = [(start + t, w) for t, w in words]
        else:                               # spread evenly
            ws = re.sub(r"<[^>]*>", " ", text or "").split()
            span = max(1e-3, self.end - self.start)
            self.words = [(start + (i + 0.5) / len(ws) * span, w) for i, w in enumerate(ws)]

    def heard_words(self, a, b):
        stop = self.cut_at if self.cut_at is not None else self.end
        return [w for t, w in self.words if a <= t <= b and t <= stop]

    def energy(self, a, b):
        a, b = max(a, self.start), min(b, self.end if self.cut_at is None else min(self.end, self.cut_at))
        if b <= a:
            return 0.0
        if self.audio is None:
            return (b - a) * 1e-3
        i, j = int((a - self.start) * audio.RATE), int((b - self.start) * audio.RATE)
        return float(np.sum(self.audio[i:j].astype(np.float64) ** 2))


class SimVec:
    """A 'voice embedding' that knows whose sound it came from (the sim's ground truth)."""

    def __init__(self, who):
        self.who = who


class SimVoices:
    """robot/voices.py VoiceStore as the scenario sets it: ids[who] = (person id or None, similarity),
    own[who] = {person id: similarity to their prints}, prints[person id] = prints kept."""

    def __init__(self):
        self.ids, self.own, self.prints, self.learnt = {}, {}, collections.Counter(), []

    def identify(self, vec, among=None):
        return self.ids.get(vec.who, (None, 0.2))

    def scores(self, vec):
        return dict(self.own.get(vec.who, {}))

    def learn(self, pid, vec):
        self.learnt.append((pid, vec.who))
        self.prints[pid] += 1
        return True

    def count(self, pid):
        return self.prints[pid]


class SimStt:
    """Speech-to-text from the ground truth: the words of every source whose word falls inside the utterance, in
    the order the sources started (the robot's own words first when it was talked over). Takes stt_s of sim time.
    real= a robot/stt.py STT (costs money, --real-stt only)."""

    name = "sim"

    def __init__(self, world, stt_s=0.8, real=None):
        self.world, self.stt_s, self.real = world, stt_s, real
        self.calls, self.last_error = [], None

    def transcribe(self, pcm, prompt=""):
        a, b = self.world._ctx
        if self.real is not None:
            text = self.real.transcribe(pcm, prompt=prompt)
        else:
            text = self.world.transcript(a, b)
        self.calls.append(dict(t0=a, t1=b, text=text, at=self.world.t))
        self.world.advance(self.stt_s)
        return text


class SimMind:
    """The mind (Claude): answer(text, scene) -> list of sentences ("" = not for me, None = cannot answer). Streams
    them like robot/mind.py: first_s of 'writing' before sentence 1, next_s before each further one."""

    name = "sim"

    def __init__(self, world, first_s=1.0, next_s=0.8):
        self.world, self.first_s, self.next_s = world, first_s, next_s
        self.asked = []
        self.answer = lambda text, scene: ["Tamam."]

    def reply(self, text, on_sentence=None, **scene):
        self.asked.append(dict(text=text, at=self.world.t, **scene))
        out = self.answer(text, scene)
        if out is None or out == "":
            self.world.advance(self.first_s)
            return out
        for i, s in enumerate(out):
            self.world.advance(self.first_s if i == 0 else self.next_s)
            if on_sentence is not None:
                on_sentence(s)
        return " ".join(out)


class SimVoice:
    """robot/tts.py Voice in sim time: say() queues (made in make_s, 0 = cached), phrases play in order for their
    real length; busy = playing (+ ECHO_S of room ring) or queued or held; interrupt() fades out in 0.2 s, drops the
    queue and the holds. Closed loop: what plays goes into the EchoRef and (synthetic mode) into the mic."""

    ECHO_S = 0.15
    FADE_S = 0.2

    def __init__(self, world, make_s=0.0):
        self.world, self.make_s = world, make_s
        self.q = collections.deque()
        self.playing = None
        self.played = []                    # dict(text, start, end, cut)
        self.pending, self._holds, self.gen = 0, 0, 0
        self.ring_until = -1.0
        self.interrupts = []                # sim times of interrupt()
        self.scripted = []                  # recording mode: (start, frames, seconds, text) played on schedule
        self._pid = 0

    @property
    def busy(self):
        return self.playing is not None or self.pending > 0 or self._holds > 0 or self.world.t < self.ring_until

    def say(self, text, level=None, mark=None):
        self.q.append(dict(text=text, ready=self.world.t + self.make_s, mark=mark))
        self.pending += 1

    def cached(self, text):
        return False

    @contextlib.contextmanager
    def hold(self):
        self._holds += 1
        try:
            yield self
        finally:
            self._holds = max(0, self._holds - 1)

    def interrupt(self):
        t = self.world.t
        self.gen += 1
        self._holds = 0
        self.pending -= len(self.q)
        self.q.clear()
        self.interrupts.append(t)
        p = self.playing
        if p is not None and p["end"] > t + self.FADE_S:
            p["end"] = t + self.FADE_S
            p["cut"] = True
            self.world.cut_echo(p, t + self.FADE_S)
            if not p.get("scripted"):           # a recording goes on playing what it played: so does its reference
                self.world.echo.cut(t + 0.25)   # as robot/tts.py Speaker.stop

    def tick(self):
        t = self.world.t
        p = self.playing
        if p is not None and t >= p["end"]:
            self.playing = None
            self.ring_until = p["end"] + self.ECHO_S
            if not p.get("scripted"):
                self.pending -= 1
            self.played.append(p)
        if self.playing is None and self.scripted and self.scripted[0][0] <= t:
            start, data, seconds, text = self.scripted.pop(0)
            self._pid += 1
            self.world.echo.add(data, pid=self._pid)
            self.world.echo.ended(self._pid, start + seconds)
            self.playing = dict(text=text, start=start, end=start + seconds, cut=False, scripted=True)
            # on the robot these sentences came through Conversation._say: it knows it said them
            self.world.conv._recent_said.append((t, text))
            self.world.robot_source(self.playing, None)
        if self.playing is None and self.q and self.q[0]["ready"] <= t:
            item = self.q.popleft()
            x, words = audio.voice(re.sub(r"<[^>]*>", " ", item["text"]), "robot")
            self._pid += 1
            self.world.echo.add(audio.frames(x), pid=self._pid)
            self.playing = dict(text=item["text"], start=t, end=t + len(x) / audio.RATE, cut=False, words=words)
            self.world.robot_source(self.playing, x)
            if item["mark"] is not None:
                item["mark"].update(made=t, play=t, src="sim")
            for trig in list(self.world.triggers):
                if item["text"].startswith(trig["prefix"]):
                    self.world.triggers.remove(trig)
                    trig["then"](t, self.playing["end"])


class World:
    """One scenario's room. mic=None: synthetic (floor + what is said + the robot's echo); mic=Recording: the real
    recording replayed (its echo already in it - nothing is added)."""

    def __init__(self, seconds=30.0, seed=0, recording=None, people=("Selim", "Emre", "Meltem", "Halil"),
                 vad=None, stt_s=0.8, mind_s=(1.0, 0.8), make_s=0.0, real_stt=None):
        self.seed, self.recording = seed, recording
        self.t = T0
        self.i = 0
        self.mic = recording.mic.copy() if recording is not None else audio.mic_floor(seconds + 2, seed)
        self.sources = []
        self.looks = []                     # dict(t0, t1, tid, person, looking)
        self.triggers = []                  # dict(prefix, then(t_start, t_end)) when the robot starts a phrase
        self.utts, self.barges, self.errors = [], [], []
        self.events = []
        self._dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self._dir.name, clock=lambda: 0.0)
        rng = np.random.default_rng(seed)
        self.pid = {}
        for name in people:
            pid = self.store.enrol([rng.normal(size=8) / 3])
            self.store.set_name(pid, name)
            self.pid[name] = pid
        self.echo = EchoRef(clock=self.clock, start_delay_s=0.0)
        self.voice = SimVoice(self, make_s=make_s)
        self.stt = SimStt(self, stt_s=stt_s, real=real_stt)
        self.mind = SimMind(self, *mind_s)
        self.voices = SimVoices()
        self.conv = Conversation(self.voice, self.stt, self.store, self.mind, self.voices, voice_emb=object(),
                                 clock=self.clock, wall=lambda: 1793490000.0, start=False, sleep=self.advance,
                                 filler_p=0)
        self.vad = vad or load_vad()
        self.ears = Ears(self.vad, muted=lambda: self.voice.busy, on_utterance=self._heard, clock=self.clock,
                         popen=None)
        self.ears.listener = self.conv
        self.ears.barge = BargeIn(self.echo)
        self.ears.on_barge = self._on_barge
        self._in_worker = False
        self._ctx = (T0, T0)

    # ---------- the scenario's script ----------

    def at(self, s):
        """Scenario seconds -> sim time."""
        return T0 + s

    def look(self, person, t0, t1, looking=True, tid=None):
        """`person` (a name, or None for a face not known) is in view from t0 to t1 (scenario seconds), looking at
        the robot or not."""
        pid = self.pid.get(person, person) if person else None
        tid = tid if tid is not None else zlib.crc32(f"{person}".encode()) % 1000
        self.looks.append(dict(t0=self.at(t0), t1=self.at(t1), tid=tid, person=pid, looking=looking))

    def say(self, who, text, at, seconds=None, stt=None, level=None, kind="person"):
        """`who` says `text` at scenario second `at` (stt = what the STT writes, if not the text). Sim-time start."""
        start = self.at(at) if at < T0 / 2 else at
        x, words = audio.voice(text, who, seed=self.seed, level=level, seconds=seconds)
        if stt is not None:                 # the STT writes something else for it: spread over its length
            words = None
        src = Source(who, text if stt is None else stt, start, len(x) / audio.RATE, kind, x, words)
        self._mix(x, start)
        self.sources.append(src)
        return src

    def say_virtual(self, who, text, start, end):
        """Recording mode: someone speaks in the recording from start to end (sim times) - for the STT and voice ID
        only, nothing is added to the mic."""
        src = Source(who, text, start, end - start, "person")
        self.sources.append(src)
        return src

    def when_robot_says(self, prefix, then):
        """then(t_start, t_end) as the robot starts playing a phrase that starts with prefix (once)."""
        self.triggers.append(dict(prefix=prefix, then=then))

    def schedule_robot(self, start, data, seconds, text):
        """Recording mode: the robot plays these frames at sim time start (as it did when it was recorded)."""
        self.voice.scripted.append((start, data, seconds, text))
        self.voice.scripted.sort(key=lambda s: s[0])

    # ---------- the room ----------

    def clock(self):
        return self.t

    def _mix(self, x, start):
        i = int(round((start - T0) * audio.RATE))
        if i < 0:
            x, i = x[-i:], 0
        j = min(len(self.mic), i + len(x))
        if j > i:
            self.mic[i:j] += x[:j - i]

    def robot_source(self, p, played):
        """The robot started a phrase: its echo reaches the mic LAG_S later (added to a synthetic mic)."""
        start = p["start"] + LAG_S
        if played is None:                  # the recording: already in the mic
            src = Source("robot", p["text"], start, p["end"] - p["start"], "robot")
        else:
            e = audio.couple(played, seed=self.seed + len(self.sources))
            src = Source("robot", p["text"], start, len(e) / audio.RATE, "robot", e, p.get("words"))
            self._mix(e, start)
        p["source"] = src
        self.sources.append(src)

    def cut_echo(self, p, at):
        """The robot stopped at `at`: what would have played after it leaves the mic (a 50 ms fade)."""
        src = p.get("source")
        if src is None:
            return
        src.cut_at = at + LAG_S
        if src.audio is None or self.recording is not None:
            return                          # a recording keeps what really played
        k = int((src.cut_at - src.start) * audio.RATE)
        if k >= len(src.audio):
            return
        rest = src.audio[max(0, k):].copy()
        fade = min(len(rest), int(0.05 * audio.RATE))
        rest[:fade] *= np.linspace(0, 1, fade)
        self._mix(-rest, src.start + max(0, k) / audio.RATE)

    def tracks_at(self, t):
        out = []
        for lk in self.looks:
            if lk["t0"] <= t < lk["t1"]:
                start = lk["t0"]
                out.append(SimpleNamespace(id=lk["tid"], person=lk["person"], looking=lk["looking"],
                                           looked_for=(lambda now, s=start, on=lk["looking"]: now - s if on else 0.0)))
        return out

    def dominant(self, a, b):
        best, who = 0.0, None
        for s in self.sources:
            e = s.energy(a, b)
            if e > best:
                best, who = e, s.who
        return who

    def transcript(self, a, b):
        parts = []
        for s in sorted(self.sources, key=lambda s: s.start):
            words = s.heard_words(a, b)
            if words:
                parts.append(" ".join(words))
        return " ".join(parts)

    # ---------- running ----------

    def _heard(self, pcm, t0, t1, onset=None):
        self.utts.append(dict(t0=t0, t1=t1, onset=onset, seconds=len(pcm) / 16000, who=self.dominant(onset, t1)))
        self.conv.heard(pcm, t0, t1, onset=onset)

    def _on_barge(self):
        stopped = self.conv.barge_in()
        self.barges.append(dict(t=self.t, stopped=stopped))
        return stopped

    def _voices_step(self):
        q = self.conv._voice_q
        while not q.empty():
            item = q.get()
            if item is None:
                continue
            pcm, t0, t1, scene, pending = item
            vec = SimVec(self.dominant(scene.get("onset", t0), t1))
            vid, sim = self.voices.identify(vec)
            self.conv._event(kind="voice", person=vid, sim=round(sim, 3), seconds=round(len(pcm) / 16000, 2))
            pending.set(vid, vec)

    def _work(self):
        q = self.conv._todo
        while not q.empty():
            item = q.get()
            if item is None:
                continue
            self._in_worker = True
            try:
                if item[0] == "ask_name":
                    self.conv._ask_name(item[1])
                else:
                    self._ctx = (item[2], item[3])
                    self.conv._utterance(*item[1:])
            except Exception as e:  # noqa: BLE001 - a crash is a scenario failure, not a sim crash
                import traceback
                self.errors.append(f"{item[0]}: {e!r}\n{traceback.format_exc()}")
            finally:
                self._in_worker = False

    def _feed(self):
        a = self.i * FRAME48
        seg = self.mic[a:a + FRAME48]
        if len(seg) < FRAME48:
            return False
        self.t = T0 + (self.i + 1) * CHUNK_S
        if self.i % 2 == 0:
            self.conv.update(self.tracks_at(self.t), [], self.t)
        self.voice.tick()
        self.ears._chunk(audio.raw_chunk(seg))
        self.i += 1
        self._voices_step()
        if not self._in_worker:
            self._work()
        return True

    def advance(self, dt):
        """sim time moves dt on (the conversation's sleep, the STT / mind working): the room goes on meanwhile."""
        target = self.t + dt
        while T0 + (self.i + 1) * CHUNK_S <= target + 1e-9:
            if not self._feed():
                break
        self.t = max(self.t, target)

    def run(self, until):
        """Run to scenario second `until` (or the end of the mic), then let the worker finish its queue."""
        with self._sim_time():
            end = self.at(until)
            while self.t < end and self._feed():
                pass
            self._work()
            self.events += self.conv.events()
        return self

    @contextlib.contextmanager
    def _sim_time(self):
        """robot/conversation.py reads time.monotonic() directly for a few things (the barge-in window, what it said
        lately, timing lines): in the sim those follow the sim clock."""
        world = self

        class SimTime:
            def monotonic(self):
                return world.t

            def __getattr__(self, name):
                return getattr(_time, name)
        old = conv_mod.time
        conv_mod.time = SimTime()
        try:
            yield
        finally:
            conv_mod.time = old

    def close(self):
        self._dir.cleanup()

    # ---------- what happened ----------

    def kinds(self, kind):
        return [e for e in self.events if e["kind"] == kind]

    def said(self):
        """The phrases that played (text, cut?)."""
        return [(p["text"], p["cut"]) for p in self.voice.played + ([self.voice.playing] if self.voice.playing else [])
                if not p.get("scripted")]

    def rel(self, t):
        return None if t is None else round(t - T0, 3)
