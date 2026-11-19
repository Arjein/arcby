"""
The scenarios: each builds a room (sim/world.py), runs it, and checks what the robot did. A scenario returns a Result
(ok, the failed checks, timing numbers). `known` = a gap the sim found that is not fixed yet (with its diagnosis): it
is reported XFAIL while it still fails, XPASS once it does not.
"""
import time

from . import audio
from .world import T0, World


class Result:
    def __init__(self, name, title, status, reasons=(), metrics=None, known=None, seconds=0.0):
        self.name, self.title, self.status = name, title, status
        self.reasons, self.metrics, self.known, self.seconds = list(reasons), dict(metrics or {}), known, seconds

    @property
    def ok(self):
        return self.status in ("PASS", "XFAIL", "SKIP")

    def reason(self):
        if self.status == "XFAIL":
            return f"{'; '.join(self.reasons)}\n   known: {self.known}"
        return "; ".join(self.reasons)


class Skip(Exception):
    pass


class Check:
    def __init__(self):
        self.fails, self.metrics = [], {}

    def that(self, cond, msg):
        if not cond:
            self.fails.append(msg)
        return bool(cond)


class Ctx:
    """What a scenario gets: a world factory (cleaned up after it), the recording (or Skip), the options."""

    def __init__(self, seed=0, real_stt=None, recording=None):
        self.seed, self.real_stt, self._recording = seed, real_stt, recording
        self.worlds = []

    def world(self, seconds=20.0, **kw):
        kw.setdefault("real_stt", self.real_stt)
        w = World(seconds=seconds, seed=self.seed, **kw)
        self.worlds.append(w)
        return w

    @property
    def recording(self):
        if self._recording is None:
            self._recording = audio.Recording.find() or False
        if not self._recording:
            raise Skip("the barge-in recording is not on this machine (mic48k.npy, ref0-3.wav, log16.json in "
                       "~/.local/share/arcby/sim/barge-2026-11-01 or $ARCBY_SIM_RECORDING) - family voice data, "
                       "kept local, never in git")
        return self._recording

    def close(self):
        for w in self.worlds:
            w.close()


SCENARIOS = []


def scenario(name, title, known=None):
    def wrap(fn):
        SCENARIOS.append(dict(name=name, title=title, fn=fn, known=known))
        return fn
    return wrap


def run_one(sc, seed=0, real_stt=None, recording=None):
    ctx = Ctx(seed, real_stt, recording)
    t = time.perf_counter()
    try:
        c = sc["fn"](ctx)
        for w in ctx.worlds:
            for e in w.errors:
                c.fails.append("worker crashed: " + e.splitlines()[0])
        status = "PASS" if not c.fails else ("XFAIL" if sc["known"] else "FAIL")
        if not c.fails and sc["known"]:
            status = "XPASS"
        return Result(sc["name"], sc["title"], status, c.fails, c.metrics, sc["known"], time.perf_counter() - t)
    except Skip as e:
        return Result(sc["name"], sc["title"], "SKIP", [str(e)], seconds=time.perf_counter() - t)
    finally:
        ctx.close()


def run_all(names=None, seed=0, real_stt=None):
    recording = None
    out = []
    for sc in SCENARIOS:
        if names and not any(n in sc["name"] for n in names):
            continue
        out.append(run_one(sc, seed, real_stt, recording))
    return out


# ---------- helpers ----------

def mind_texts(w):
    return [a["text"] for a in w.mind.asked]


def first_play_after(w, t):
    ps = [p for p in w.voice.played + ([w.voice.playing] if w.voice.playing else []) if p["start"] >= t]
    return min(ps, key=lambda p: p["start"]) if ps else None


def answer_with(*replies):
    """mind.answer that gives these replies in turn (each a list of sentences), then None."""
    todo = list(replies)
    return lambda text, scene: todo.pop(0) if todo else None


# ---------- 1-8: who is listened to ----------

@scenario("01_selim_asks", "Selim alone looks and asks -> exactly one reply, his")
def s01(ctx):
    w = ctx.world(14)
    selim = w.pid["Selim"]
    w.look("Selim", 0, 14)
    w.voices.ids["Selim"] = (selim, 0.75)
    w.mind.answer = answer_with(["Bugün hava güneşli, yirmi derece."])
    line = w.say("Selim", "Bugün hava nasıl?", at=2.0)
    w.run(14)
    c = Check()
    c.that(mind_texts(w) == ["Bugün hava nasıl?"], f"mind got {mind_texts(w)}")
    c.that(w.mind.asked and w.mind.asked[0]["speaker"] == "Selim", "the mind was not told it is Selim")
    c.that(w.said() == [("Bugün hava güneşli, yirmi derece.", False)], f"said {w.said()}")
    heard = w.kinds("heard")
    c.that(len(heard) == 1 and heard[0]["person"] == selim, f"heard events {[(h['text'], h['person']) for h in heard]}")
    c.that(len(w.utts) == 1, f"{len(w.utts)} utterances from one sentence")
    p = first_play_after(w, line.end)
    if p and w.utts:
        c.metrics["onset_offset_s"] = round(w.utts[0]["onset"] - line.start, 3)
        c.metrics["reply_after_end_s"] = round(p["start"] - line.end, 2)
    return c


@scenario("02_not_looking", "Nobody looks, no conversation -> nothing to the STT, no reply")
def s02(ctx):
    w = ctx.world(10)
    w.look("Selim", 0, 10, looking=False)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.75)
    w.say("Selim", "Napıyorsun bugün?", at=2.0)
    w.run(10)
    c = Check()
    c.that(w.stt.calls == [], f"sent to the STT: {[s['text'] for s in w.stt.calls]}")
    c.that(w.said() == [], f"said {w.said()}")
    c.that(len(w.kinds("dropped")) >= 1, "no dropped event (did the ears hear it at all?)")
    return c


@scenario("03_emre_looks_selim_talks", "Emre looks, Selim (known voice, not looking) talks to his computer -> dropped")
def s03(ctx):
    w = ctx.world(10)
    selim, emre = w.pid["Selim"], w.pid["Emre"]
    w.look("Emre", 0, 10)
    w.look("Selim", 0, 10, looking=False)
    w.voices.ids["Selim"] = (selim, 0.78)
    w.voices.ids["Emre"] = (emre, 0.78)
    w.say("Selim", "Bu dosyayı nereye kaydettim ya", at=2.0)
    w.run(10)
    c = Check()
    c.that(w.stt.calls == [], f"sent to the STT: {[s['text'] for s in w.stt.calls]}")
    c.that(w.said() == [], f"said {w.said()}")
    dropped = w.kinds("dropped")
    c.that(dropped and dropped[0].get("reason") == "voice not the looking face", f"dropped events {dropped}")
    c.that(not any(pid == emre for pid, _ in w.voices.learnt), f"voice prints learnt: {w.voices.learnt}")
    return c


@scenario("04_side_remark", "Talking with Emre; Selim (not looking) says something to Emre -> side remark, Emre goes on")
def s04(ctx):
    w = ctx.world(22)
    selim, emre = w.pid["Selim"], w.pid["Emre"]
    w.look("Emre", 0, 3.5)
    w.look("Emre", 3.5, 22, looking=False, tid=2)
    w.voices.ids["Selim"] = (selim, 0.8)
    w.voices.ids["Emre"] = (emre, 0.8)
    w.mind.answer = answer_with(["Merhaba Emre Bey."], ["Yarın güneşli olacak."])
    w.say("Emre", "Merhaba Arkbi", at=1.0)
    w.say("Selim", "Emre, yemek hazır gel", at=9.0)
    w.say("Emre", "Yarın hava nasıl olacak?", at=13.0)
    w.run(22)
    c = Check()
    c.that(mind_texts(w) == ["Merhaba Arkbi", "Yarın hava nasıl olacak?"], f"mind got {mind_texts(w)}")
    side = [e for e in w.kinds("dropped") if e.get("reason") == "a side remark, not to the robot"]
    c.that(len(side) == 1 and side[0]["person"] == selim, f"dropped events {w.kinds('dropped')}")
    c.that(len(w.mind.asked) == 2 and w.mind.asked[1]["speaker"] == "Emre",
           f"2nd speaker {w.mind.asked[1]['speaker'] if len(w.mind.asked) > 1 else None}")
    c.that(w.conv.session and w.conv.session["person"] == emre, f"session {w.conv.session}")
    return c


@scenario("05_two_look_unknown_voice", "Selim + Meltem both look as it starts, voice unknown -> 'Selim ya da Meltem', unsure")
def s05(ctx):
    w = ctx.world(18)
    w.look("Selim", 0, 18)
    w.look("Meltem", 6.0, 18)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    w.voices.ids["Meltem"] = (None, 0.3)
    w.mind.answer = answer_with(["Merhaba Selim."], ["Hoş geldiniz Meltem Hanım."])
    w.say("Selim", "Merhaba", at=1.0)
    w.say("Meltem", "Merhaba canım, hoş buldum", at=9.0)
    w.run(18)
    c = Check()
    c.that(len(w.mind.asked) == 2, f"mind got {mind_texts(w)}")
    if len(w.mind.asked) == 2:
        a = w.mind.asked[1]
        c.that(sorted((a["speaker"] or "").split(" ya da ")) == ["Meltem", "Selim"], f"speaker {a['speaker']!r}")
        c.that(a["unsure"], "not marked unsure")
        heard = w.kinds("heard")
        c.that(heard[-1]["person"] is None, f"attributed to {heard[-1]['person']} (the open talk's partner?)")
    return c


@scenario("06_noises", "Cough / 'Ahem' / sneeze transcribed -> not answered")
def s06(ctx):
    w = ctx.world(14)
    w.look("Selim", 0, 14)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    for i, (text, secs) in enumerate((("Ahem", 0.5), ("Hapşu!", 0.45), ("[öksürük]", 0.6))):
        w.say("cough", text, at=2.0 + 4 * i, seconds=secs)
    w.run(14)
    c = Check()
    c.that(w.mind.asked == [], f"mind got {mind_texts(w)}")
    c.that(w.said() == [], f"said {w.said()}")
    c.that(len(w.kinds("noise")) == 3, f"{len(w.kinds('noise'))} noise events of 3 ({len(w.utts)} utterances)")
    return c


@scenario("07_name_fix", "'Nasılsın Arif Bey?' / 'Arka nasılsın?' -> the mind gets 'Arkbi'")
def s07(ctx):
    w = ctx.world(16)
    w.look("Selim", 0, 16)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    w.mind.answer = answer_with(["İyiyim."], ["İyiyim, sağ ol."])
    w.say("Selim", "Nasılsın Arif Bey?", at=1.0)
    w.say("Selim", "Arka nasılsın?", at=9.0)
    w.run(16)
    c = Check()
    c.that(mind_texts(w) == ["Nasılsın Arkbi?", "Arkbi nasılsın?"], f"mind got {mind_texts(w)}")
    c.that(len(w.kinds("called")) == 2, f"{len(w.kinds('called'))} 'called' events (head turns) of 2")
    return c


@scenario("08_tiktok_in_talk", "Talking with Selim (20 prints); a TikTok voice (< 0.4 to his) -> not answered")
def s08(ctx):
    w = ctx.world(22)
    selim = w.pid["Selim"]
    w.look("Selim", 0, 3.5)
    w.look("Selim", 3.5, 22, looking=False, tid=2)
    w.voices.ids["Selim"] = (selim, 0.8)
    w.voices.prints[selim] = 20
    w.voices.ids["TikTok"] = (None, 0.15)
    w.voices.own["TikTok"] = {selim: 0.1}
    w.mind.answer = answer_with(["Merhaba Selim."], ["Saat on dokuz."])
    w.say("Selim", "Merhaba", at=1.0)
    w.say("TikTok", "Bu videoyu beğenmeyi unutmayın", at=8.0)
    w.say("Selim", "Neyse, saat kaç?", at=13.0)
    w.run(22)
    c = Check()
    c.that(mind_texts(w) == ["Merhaba", "Neyse, saat kaç?"], f"mind got {mind_texts(w)}")
    c.that(any(e.get("reason") == "a side remark, not to the robot" for e in w.kinds("dropped")),
           f"dropped events {w.kinds('dropped')}")
    return c


# ---------- 9-16: barge-in, echo, talking over the robot ----------

REC_TEXTS = ["Merhaba Selim, ben Arcby. Bugün seni dinlemeye hazırım, ne istersen sorabilirsin.",
             "Bu bir deneme cümlesi, sesimi mikrofonda ne kadar duyduğumu ölçüyoruz şimdi.",
             "Üçüncü cümle de benden geliyor, kimse konuşmuyor, sadece ben konuşuyorum burada.",
             "İyiyim Selim, teşekkür ederim. Sen nasılsın, bugün neler yaptın bakalım?"]


def recording_world(ctx, look_from=None):
    rec = ctx.recording
    w = ctx.world(recording=rec)
    for (start, data, secs), text in zip(rec.phrases, REC_TEXTS):
        w.schedule_robot(w.at(start), data, secs, text)
    if look_from is not None:
        w.look("Selim", look_from, rec.seconds)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.75)
    w.say_virtual("Selim", "Hava nasıl?", w.at(rec.TALK_OVER_S), w.at(45.4))   # his words: 44.0-45.4 s (levels)
    w.mind.answer = answer_with(["Hava güneşli."])
    return rec, w


def phrase_windows(rec):
    return [(s + 0.108, s + 0.108 + secs + 0.15) for s, _, secs in rec.phrases]


@scenario("09_barge_real", "REAL recording: Selim looks and talks over sentence 4 at 44.0 s -> stopped, listens")
def s09(ctx):
    rec, w = recording_world(ctx, look_from=43.0)
    w.run(rec.seconds)
    c = Check()
    # only up to 46 s: later the recording still holds what really played then (the robot did not stop on the day),
    # under the sim robot's own reply
    ints = [w.rel(t) for t in w.voice.interrupts if t < w.at(46.0)]
    c.that(len(ints) == 1, f"interrupts at {ints} s")
    if ints:
        lat = ints[0] - rec.TALK_OVER_S
        c.metrics["interrupt_latency_s"] = round(lat, 3)
        c.that(0.0 <= lat <= 0.5, f"interrupted {lat:+.2f} s after Selim started (want 0..0.5)")
    after = [u for u in w.utts if u["onset"] >= w.at(43.0)]
    c.that(after, "no utterance after the barge-in")
    if after:
        u = after[0]
        c.metrics["utt_t0_offset_s"] = round(u["t0"] - w.at(rec.TALK_OVER_S), 3)
        c.metrics["utt_seconds"] = round(u["seconds"], 2)
        c.that(-0.4 <= u["t0"] - w.at(rec.TALK_OVER_S) <= 0.3,
               f"utterance starts {u['t0'] - w.at(rec.TALK_OVER_S):+.2f} s from Selim's voice (want -0.4..+0.3)")
    c.that(mind_texts(w) == ["Hava nasıl?"] or ("Efendim?", False) in w.said(),
           f"mind got {mind_texts(w)}, said {w.said()} (STT heard {[s['text'] for s in w.stt.calls]})")
    early = [w.rel(b["t"]) for b in w.barges if b["t"] < w.at(43.0)]
    c.metrics["detector_fired_before_look_s"] = early
    c.that(not any(w.at(43.0) > t >= T0 for t in w.voice.interrupts), "stopped while nobody looked")
    return c


@scenario("10_barge_nobody_looks", "Robot speaks, Selim talks over it not looking, no conversation -> not interrupted")
def s10(ctx):
    w = ctx.world(14)
    w.look("Selim", 0, 14, looking=False)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    w.when_robot_says("Günaydın", lambda t0, t1: w.say("Selim", "Evet anne geliyorum, bir dakika", at=t0 + 1.5))
    w.voice.say("Günaydın Selim! Bugün hava çok güzel, dışarı çıkmayı düşünüyor musun acaba?")
    w.run(14)
    c = Check()
    c.that(w.voice.interrupts == [], f"interrupted at {[w.rel(t) for t in w.voice.interrupts]}")
    c.that(w.said() == [("Günaydın Selim! Bugün hava çok güzel, dışarı çıkmayı düşünüyor musun acaba?", False)],
           f"said {w.said()}")
    c.that(w.stt.calls == [], f"sent to the STT: {[s['text'] for s in w.stt.calls]}")
    c.metrics["detector_fired"] = len(w.barges)
    return c


def talk_over(ctx, robot_reply, selim_line, delay=1.0, selim_stt=None, seconds=None):
    """Selim asks (looking), the mind replies `robot_reply`; `delay` s into its first sentence Selim talks over it."""
    w = ctx.world(20)
    w.look("Selim", 0, 20)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    w.mind.answer = answer_with(robot_reply, ["Peki."])
    w.say("Selim", "Nasılsın?", at=1.0)
    lines = []
    w.when_robot_says(robot_reply[0][:8], lambda t0, t1: lines.append(
        w.say("Selim", selim_line, at=t0 + delay, stt=selim_stt, seconds=seconds)))
    w.run(20)
    return w, lines


@scenario("11a_own_words_only", "Talked over, the transcript is only its own words -> it stays stopped, no reply")
def s11a(ctx):
    # was "Efendim?" until 2026-11-02: on the robot its own echo cut every reply and "Efendim?" / the answers to
    # its own words heard back kept the loop going - only its own words after a cut are never answered now
    w, lines = talk_over(ctx, ["İyiyim Selim, teşekkür ederim, bugün çok iyi hissediyorum.", "Sen nasılsın?"],
                         "ııı", selim_stt="", seconds=1.0)
    c = Check()
    c.that(len(w.voice.interrupts) == 1, f"interrupts {len(w.voice.interrupts)}")
    c.that(not any(t == "Efendim?" for t, _ in w.said()) and len(w.said()) == 1,
           f"said {w.said()} (STT heard {[s['text'] for s in w.stt.calls]})")
    c.that(len(w.mind.asked) == 1, f"mind got {mind_texts(w)}")
    return c


@scenario("11b_own_plus_new_words", "Talked over: own words + new words -> only the new words go to the mind")
def s11b(ctx):
    w, lines = talk_over(ctx, ["İyiyim Selim, teşekkür ederim, bugün çok iyi hissediyorum.", "Sen nasılsın?"],
                         "Hava nasıl peki?")
    c = Check()
    c.that(len(w.voice.interrupts) == 1, f"interrupts {len(w.voice.interrupts)}")
    stt = [s["text"] for s in w.stt.calls]
    c.that(len(stt) == 2 and stt[1] != "Hava nasıl peki?", f"the STT got no robot words to strip: {stt}")
    c.that(mind_texts(w)[1:] == ["Hava nasıl peki?"], f"mind got {mind_texts(w)} (STT heard {stt})")
    if lines and w.voice.interrupts:
        c.metrics["interrupt_latency_s"] = round(w.voice.interrupts[0] - lines[0].start, 3)
        after = [u for u in w.utts if u["onset"] >= lines[0].start - 1.0]
        if after:
            c.metrics["utt_t0_offset_s"] = round(after[0]["t0"] - lines[0].start, 3)
    return c


@scenario("11c_new_words_share_a_word", "Talked over: their words START with a word the robot said -> kept")
def s11c(ctx):
    w, lines = talk_over(ctx, ["Bugün hava çok güzel, dışarı çıkabilirsin.", "Ne dersin?"], "Hava yarın da güzel mi?",
                         delay=1.6)
    c = Check()
    stt = [s["text"] for s in w.stt.calls]
    c.that(mind_texts(w)[1:] == ["Hava yarın da güzel mi?"], f"mind got {mind_texts(w)} (STT heard {stt})")
    return c


@scenario("12a_echo_long_reply", "Synthetic: its own echo through a 3-sentence reply -> no utterance, no self-interrupt")
def s12a(ctx):
    w = ctx.world(30)
    w.look("Selim", 0, 30)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    reply = ["Bir zamanlar küçük bir robot varmış, adı Arcby imiş ve bir evde yaşarmış.",
             "Her sabah evdekileri selamlar, yüzlerini tanır, onlarla sohbet edermiş.",
             "Bir gün kendi sesini duymayı öğrenmiş ve çok şaşırmış, işte hikaye bu kadar."]
    w.mind.answer = answer_with(reply)
    q = w.say("Selim", "Bana bir hikaye anlat", at=1.0)
    w.run(30)
    c = Check()
    c.that(w.voice.interrupts == [], f"interrupted itself at {[w.rel(t) for t in w.voice.interrupts]}")
    c.that(w.said() == [(s, False) for s in reply], f"said {w.said()}")
    c.that(len(w.utts) == 1, f"{len(w.utts)} utterances (its own echo heard?): "
           f"{[(w.rel(u['onset']), u['who']) for u in w.utts]}")
    c.metrics["detector_fired"] = [w.rel(b["t"]) for b in w.barges]
    c.metrics["reply_seconds"] = round(sum(p["end"] - p["start"] for p in w.voice.played), 1)
    return c


@scenario("12b_echo_real", "REAL recording, nobody looks: its echo (4 sentences) -> never an utterance, never stopped")
def s12b(ctx):
    rec, w = recording_world(ctx, look_from=None)
    w.run(rec.seconds)
    c = Check()
    wins = [(w.at(a), w.at(b)) for a, b in phrase_windows(rec)]
    inside = [w.rel(u["onset"]) for u in w.utts if any(a <= u["onset"] <= b for a, b in wins)]
    c.that(inside == [], f"utterances started while it spoke at {inside} s")
    c.that(w.voice.interrupts == [], f"stopped at {[w.rel(t) for t in w.voice.interrupts]}")
    fired = [w.rel(b["t"]) for b in w.barges]
    c.metrics["detector_fired_s"] = fired
    s4 = rec.phrases[3][0]
    stray = [t for t in fired if t < s4 and abs(t - rec.KNOCK_S) > 0.6]
    c.that(stray == [], f"the detector fired on its own echo at {stray} s (sentences 1-3, not the knock)")
    c.metrics["would_stop_on_knock"] = any(abs(t - rec.KNOCK_S) <= 0.6 for t in fired)
    c.metrics["fired_on_selim"] = any(rec.TALK_OVER_S <= t <= rec.TALK_OVER_S + 1.5 for t in fired)
    return c


def cut_reply(ctx, next_s):
    w = ctx.world(22)
    w.mind.next_s = next_s
    w.look("Selim", 0, 22)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    first = "Uzun bir cevap veriyorum, çünkü bu konu çok ilginç ve anlatacak çok şey var bence."
    w.mind.answer = answer_with([first, "Bu ikinci cümle hiç söylenmemeli."], ["Tamam, dinliyorum."])
    w.say("Selim", "Bana yapay zekayı anlat", at=1.0)
    w.when_robot_says("Uzun bir", lambda t0, t1: w.say("Selim", "Dur bir dakika", at=t0 + 1.2))
    w.run(22)
    c = Check()
    texts = [t for t, _ in w.said()]
    c.that(len(w.voice.interrupts) == 1, f"interrupts {len(w.voice.interrupts)}")
    c.that("Bu ikinci cümle hiç söylenmemeli." not in texts, f"said {w.said()}")
    c.that((first, True) in w.said(), f"sentence 1 not cut: {w.said()}")
    said = [e for e in w.kinds("said") if e.get("by") == "mind"]
    c.that(said and said[0].get("cut"), f"no 'cut' on the mind's said event: {said}")
    c.that(mind_texts(w)[1:] == ["Dur bir dakika"], f"mind got {mind_texts(w)}")
    if len(w.stt.calls) > 1 and len(w.utts) > 1:
        # the interrupting sentence waits until the worker is free: the cut reply's mind stream still runs on
        c.metrics["barge_sentence_to_stt_s"] = round(w.stt.calls[1]["at"] - w.utts[1]["t1"], 2)
    return c


@scenario("13a_cut_before_written", "Reply cut by a barge-in while sentence 2 is still being written -> never said")
def s13a(ctx):
    return cut_reply(ctx, next_s=3.0)


@scenario("13b_cut_while_queued", "Reply cut by a barge-in with sentence 2 already queued -> never said")
def s13b(ctx):
    return cut_reply(ctx, next_s=0.1)


@scenario("14_interrupt_first_reply_not_looking",
          "Selim asks (looking), looks away, talks over the FIRST reply -> stopped and heard",
          known="the conversation opens only AFTER the reply: opening it at the start let the barge-in detector's "
                "false alarms on the robot's own echo cut every first reply (on the robot, 2026-11-01) - needs "
                "barge-in tuned first")
def s14(ctx):
    w = ctx.world(22)
    w.look("Selim", 0, 3.0)
    w.look("Selim", 3.0, 22, looking=False, tid=2)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    w.mind.answer = answer_with(["Uzun bir hikaye: bir zamanlar küçük bir robot varmış ve çok konuşurmuş.",
                                 "Sonra bir gün susmayı öğrenmiş."], ["Tamam, susuyorum."])
    w.say("Selim", "Bir hikaye anlat", at=1.0)
    w.when_robot_says("Uzun bir", lambda t0, t1: w.say("Selim", "Dur, yeter", at=t0 + 1.5))
    w.run(22)
    c = Check()
    c.that(len(w.voice.interrupts) == 1, f"interrupts {len(w.voice.interrupts)} (barge-ins seen: "
           f"{[(w.rel(b['t']), b['stopped']) for b in w.barges]})")
    c.that(mind_texts(w)[1:] == ["Dur, yeter"], f"mind got {mind_texts(w)}, dropped {len(w.kinds('dropped'))}")
    return c


@scenario("15_meet_new_face", "A new face looks -> name asked, 'Benim adım Doğu', 'Evet' -> stored")
def s15(ctx):
    w = ctx.world(24)
    import numpy as np
    new = w.store.enrol([np.ones(8) / 8 ** 0.5])
    w.look(new, 0.5, 24, tid=5)
    w.voices.ids["Doğu"] = (None, 0.3)
    w.when_robot_says("Merhaba! Ben Arcby", lambda t0, t1: w.say("Doğu", "Benim adım Doğu", at=t1 + 1.4))
    w.when_robot_says("Doğu, doğru", lambda t0, t1: w.say("Doğu", "Evet", at=t1 + 1.4, seconds=0.6))
    w.run(24)
    c = Check()
    c.that(w.store.name(new) == "Doğu", f"name {w.store.name(new)!r}; said {[t for t, _ in w.said()]}; "
           f"STT heard {[s['text'] for s in w.stt.calls]}")
    c.that(w.kinds("met"), "no 'met' event")
    return c


def quick_answer(ctx, delay, answer, confirm=None):
    """A new face looks; after the name question it answers `answer` `delay` s after the robot's last sample (and
    `confirm` to 'X, doğru mu?')."""
    import numpy as np
    w = ctx.world(22)
    new = w.store.enrol([np.ones(8) / 8 ** 0.5])
    w.look(new, 0.5, 22, tid=5)
    w.voices.ids["Doğu"] = (None, 0.3)
    w.when_robot_says("Merhaba! Ben Arcby", lambda t0, t1: w.say("Doğu", answer, at=t1 + delay))
    if confirm:
        w.when_robot_says("Doğu, doğru", lambda t0, t1: w.say("Doğu", confirm, at=t1 + delay))
    w.run(22)
    return w, new


@scenario("16_quick_answer", "Name question; 'Benim adım Doğu' 0.2-1.3 s after it -> heard whole (deaf hold after it)")
def s16(ctx):
    c = Check()
    for d in (0.2, 0.5, 0.9, 1.3):
        w, new = quick_answer(ctx, d, "Benim adım Doğu")
        said = [t for t, _ in w.said()]
        c.that("Doğu, doğru mu?" in said, f"{d} s: said {said}, STT heard {[s['text'] for s in w.stt.calls]}")
        # heard through the deaf hold by the detector ("listen": it had finished, nothing cut - not a barge-in)
        c.metrics[f"via_barge_in_{d}s"] = any(b["stopped"] for b in w.barges)
        c.that(not w.ears.barged, f"{d} s: a quick answer counted as a barge-in ({w.ears.barged})")
    return c


@scenario("17_quick_yes_with_its_word", "'Doğu, doğru mu?' answered 'Doğru.' 0.4 s after it -> the name is stored",
          known=None)
def s17(ctx):
    w, new = quick_answer(ctx, 0.4, "Benim adım Doğu", confirm="Doğru.")
    c = Check()
    c.that(w.store.name(new) == "Doğu", f"name {w.store.name(new)!r}; said {[t for t, _ in w.said()]}; STT heard "
           f"{[s['text'] for s in w.stt.calls]}; interrupts at {[w.rel(t) for t in w.voice.interrupts]}")
    return c


@scenario("18_other_voice_over_reply", "Replying to Emre (looking); Selim, not looking, talks to his computer -> reply goes on",
          known="barge_in() (robot/conversation.py) decides from looks / the open talk only - whose voice it is is "
                "known ~0.4 s after the sentence ENDS: Emre looking is enough, the reply to him is cut for Selim's "
                "remark, which is then dropped ('voice not the looking face') - the robot just goes quiet")
def s18(ctx):
    w = ctx.world(24)
    w.look("Emre", 0, 24)
    w.look("Selim", 0, 24, looking=False)
    w.voices.ids["Emre"] = (w.pid["Emre"], 0.8)
    w.voices.ids["Selim"] = (w.pid["Selim"], 0.8)
    reply = ["Uzun bir hikaye anlatayım: bir zamanlar küçük bir robot varmış.", "Her gün yeni bir şey öğrenirmiş."]
    w.mind.answer = answer_with(reply)
    w.say("Emre", "Bana bir hikaye anlat", at=1.0)
    w.when_robot_says("Uzun bir", lambda t0, t1: w.say("Selim", "Bu kod neden çalışmıyor ya", at=t0 + 1.2))
    w.run(24)
    c = Check()
    c.that(w.voice.interrupts == [], f"cut at {[w.rel(t) for t in w.voice.interrupts]}; said {w.said()}; "
           f"dropped {[e.get('reason') for e in w.kinds('dropped')]}")
    return c
