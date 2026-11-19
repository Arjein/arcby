#!/usr/bin/env python3
"""
Ears test set: which speech-to-text understands the robot's people best (Turkish), and how fast on the Pi?

    python3 code/hwtest/earshots.py synth --backends whisper-tiny whisper-base whisper-small gemini
        # the robot's own cached TTS phrases (known text, family names) - clean speech, an upper bound
    python3 code/hwtest/earshots.py session      # the robot asks you to say sentences (1 m, 2 m); saved here
    python3 code/hwtest/earshots.py compare --backends ...   # the same comparison on the latest session

Scores: WER (word error rate: wrong / missing / extra words per reference word, Turkish-normalised), names right
(the family names and "Arkbi"), seconds per utterance. A session's recordings stay in
~/scratch/hwtest/earshots/<time>/ (a test set Selim records on purpose - not the robot's dataset, which has no audio).
gemini sends the clips to Google: only with Selim's OK for his voice (synth = the robot's own TTS voice).
"""
import argparse
import json
import os
import re
import sys
import time
import wave
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot import config                                  # noqa: E402
from robot.stt import GeminiSTT, WhisperSTT, timed        # noqa: E402
from hwtest.common import LOG_DIR, plain_tr               # noqa: E402

OUT = LOG_DIR / "earshots"
NAMES = ("selim", "emre", "meltem", "halil", "dogu", "arkbi")
SENTENCES = (
    "Arkbi", "Arkbi, bana bak", "Benim adım Selim", "Benim adım Doğu", "Evet", "Hayır", "Emre nerede",
    "Meltem ve Halil geldi", "Arkbi, saat kaç", "Biraz sağa dön", "Dur", "Bugün hava nasıl",
    "Arkbi, kendini tanıt", "Teşekkürler, iyi geceler", "Yarın sabah beni uyandır",
)


def words(text):
    w = plain_tr(text)
    return ["arkbi" if x in ("arcby", "arkby", "arkibi", "arkıbi") else x for x in w]


def wer(ref, hyp):
    """Word error rate (Levenshtein over words)."""
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def names_right(ref, hyp):
    """(names in the reference, of them heard) - family names and the robot's own."""
    r = [w for w in words(ref) if w in NAMES]
    h = set(words(hyp))
    return len(r), sum(w in h for w in r)


def load_backend(name):
    if name.startswith("whisper-"):
        return WhisperSTT(model=name.split("-", 1)[1])
    if name == "gemini":
        return GeminiSTT()
    if name.startswith("gemini-"):                         # a specific model: gemini-3.8-flash-lite ...
        return GeminiSTT(model=name)
    raise SystemExit(f"unknown backend {name}")


def evaluate(clips, backends, prompt):
    """clips = [(name, reference, pcm16k)] -> printed table + per-clip results."""
    rows, lines = {}, []
    for b in backends:
        stt = load_backend(b)
        stt.transcribe(np.zeros(8000, np.float32))          # warm-up
        res = []
        for name, ref, pcm in clips:
            hyp, s = timed(stt, pcm, prompt)
            res.append(dict(clip=name, ref=ref, hyp=hyp, wer=wer(ref, hyp), names=names_right(ref, hyp), s=s,
                            audio_s=len(pcm) / 16000))
        stt.close()
        rows[b] = res
        n_names = sum(r["names"][0] for r in res)
        lines.append(f"| {b} | {np.mean([r['wer'] for r in res]):.0%} | "
                     f"{sum(r['names'][1] for r in res)}/{n_names} | {np.median([r['s'] for r in res]):.1f} s | "
                     f"{np.mean([r['s'] / r['audio_s'] for r in res]):.2f} |")
        print(lines[-1], flush=True)
    table = ["| backend | WER | names heard | seconds / utterance (median) | x real time |", "|---|---|---|---|---|"]
    return "\n".join(table + lines), rows


def read16k(path):
    w = wave.open(str(path))
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    rate = w.getframerate()
    return x if rate == 16000 else np.interp(np.arange(0, len(x), rate / 16000), np.arange(len(x)), x).astype(np.float32)


def cmd_synth(args):
    from robot.greeter import Greeter
    from robot.people import PeopleStore
    from robot.tts import Voice, spoken
    store = PeopleStore(config.PEOPLE_DIR)
    voice = Voice(key="")                                   # only reads the cache
    g = Greeter(voice, store)
    clips = []
    for pid in list(store.index):
        for text in g.phrases_for(pid):
            p = voice.cache_path(spoken(text))
            if p.exists():
                clips.append((p.name, text, read16k(p)))
    if not clips:
        sys.exit("no cached phrases yet (alive.py prepares them)")
    print(f"{len(clips)} cached phrases with known text")
    table, rows = evaluate(clips, args.backends, args.prompt)
    out = OUT / ("synth-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (out / "report.md").write_text(f"# Ears on the robot's own TTS phrases ({len(clips)})\n\n{table}\n")
    worst = sorted((r for b in rows.values() for r in b), key=lambda r: -r["wer"])[:8]
    for r in worst:
        print(f"  {r['wer']:.0%}  {r['ref']!r} -> {r['hyp']!r}")
    print(f"saved {out}")


def cmd_session(args):
    from robot.mic import Ears
    from robot.tts import Voice
    from robot.vad import Vad
    import threading
    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    voice, got, ready = Voice(), [], threading.Event()

    def heard(pcm, t0, t1):
        got.append(pcm)
        ready.set()
    stop = threading.Event()
    ears = Ears(Vad(), muted=lambda: voice.busy, on_utterance=heard).start(stop)
    meta = []
    try:
        voice.say_now("Merhaba Selim. Şimdi seni dinlemeyi öğreneceğim. Ben bir cümle söyleyeceğim, sen tekrar et.")
        for d in (1, 2):
            voice.say_now(f"Lütfen benden {'bir' if d == 1 else 'iki'} metre uzakta dur.")
            time.sleep(args.walk)
            for i, text in enumerate(SENTENCES):
                voice.say_now(f"Şunu söyle: {text}.")
                time.sleep(0.3)                             # the robot's own echo is not the answer
                got.clear()
                ready.clear()
                if ready.wait(8.0):
                    time.sleep(0.8)                         # a second utterance may follow (a pause inside)
                    pcm = np.concatenate(got)
                    name = f"{d}m-{i:02d}.wav"
                    with wave.open(str(out / name), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(16000)
                        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
                    meta.append(dict(file=name, ref=text, metres=d, level=round(float(np.sqrt((pcm ** 2).mean())), 4)))
                    print(f"{name}: {text!r} ({len(pcm) / 16000:.1f} s)", flush=True)
                else:
                    print(f"{d} m {text!r}: nothing heard", flush=True)
                (out / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        voice.say_now("Teşekkürler, bitti.")
    finally:
        stop.set()
        ears.close()
        voice.close()
    print(f"saved {out}")


def cmd_compare(args):
    sessions = sorted(p for p in OUT.iterdir() if (p / "session.json").exists()) if OUT.exists() else []
    if not sessions:
        sys.exit("no session yet - run session first")
    folder = OUT / args.session if args.session else sessions[-1]
    meta = json.loads((folder / "session.json").read_text())
    clips = [(m["file"], m["ref"], read16k(folder / m["file"])) for m in meta]
    table, rows = evaluate(clips, args.backends, args.prompt)
    (folder / "compare.md").write_text(f"# Ears on {folder.name} ({len(clips)} utterances)\n\n{table}\n")
    (folder / "compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"saved {folder / 'compare.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("synth", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--backends", nargs="+", default=["whisper-tiny", "whisper-base"])
        s.add_argument("--prompt", default="", help="words to expect (names), given to the recogniser")
        if name == "compare":
            s.add_argument("--session")
    se = sub.add_parser("session")
    se.add_argument("--walk", type=float, default=6.0)
    args = ap.parse_args()
    {"synth": cmd_synth, "session": cmd_session, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
