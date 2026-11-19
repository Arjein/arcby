#!/usr/bin/env python3
"""
Barge-in calibration: how loud is THIS speaker's echo in the mic, band by band? Once per speaker (and after moving
it / changing the level a lot). Nobody talks while it runs (~30 s); the robot says a few cached phrases.

    pkill -f "code/[a]live.py"                      # it owns the speaker and the mic
    python3 code/hwtest/bargecal.py                 # at the dashboard's speaker level (--level 0.3 to choose)
    python3 code/hwtest/bargecal.py --dry-run       # measure and report, do not write the tuning file

What it does: 2 s of room (the noise floor per band), then the phrases with the mic recording, exactly as alive.py
hears them (robot/mic.py: 48 -> 16 kHz, high-pass) against what the speaker played (robot/bargein.py EchoRef, timed
by speakerd). Per band the 97th percentile of mic - played (+1 dB) becomes the coupling, so the robot's own voice
is almost never 10 dB over its prediction. The detector is then replayed on the recording with the old and the new
values: false barge-ins before -> after (should be 0 after). Written to config.BARGE_TUNE_FILE (alive.py reads it
at start). The recording (the robot's own voice only) stays in ~/scratch/hwtest/bargecal/<time>/ for tuning.

2026-11-02: written after a damaged speaker cut every reply at once (its distorted echo was far over the old
prediction, fitted on the speaker before it was damaged).
"""
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot import config                                  # noqa: E402
from robot import bargein as B                            # noqa: E402

OUT = Path.home() / "scratch" / "hwtest" / "bargecal"
QUIET_S = 2.0
GAP_S = 0.8
# cached already (fillers, greetings) or made once (one request each, then cached for good)
PHRASES = ("Hmm.", "Hmm, bakalım.", "Şey...", "<breath> Bir saniye.", "Bir bakayım.", "Let me see.",
           "Merhaba Selim!", "Tekrar hoş geldin Selim!",
           "Merhaba, ben Arkbi. Şimdi kendi sesimi dinliyorum, lütfen biraz sessiz olun.",
           "Bir, iki, üç, dört, beş. Altı, yedi, sekiz, dokuz, on.",
           "Bugün hava nasıl? Ben buradan pencereyi göremiyorum ama sesler güzel geliyor.")


def analyse(chunks, echo, lag_s, decay_db=None):
    """chunks = [(16 kHz high-passed mic chunk, start time)] -> (mic_db, tail_db) per chunk, the echo reference
    shifted by lag_s and decaying as robot/bargein.py's tail."""
    decay = config.BARGE_DECAY_DB if decay_db is None else decay_db
    tail = np.zeros(len(B.EDGES) - 1)
    mic_db, tail_db = [], []
    for c, t in chunks:
        ref = echo.segment(t - lag_s, len(c))
        tail = np.maximum(B.band_power(ref), tail * 10 ** (-decay / 10))
        tail[tail < 1e-9] = 0.0
        mic_db.append(B.db(B.band_power(c)))
        tail_db.append(B.db(tail))
    return np.array(mic_db), np.array(tail_db)


def best_lag(chunks, echo, around):
    """The lag (s) at which the mic's loudness follows what was played best (broadband, in dB)."""
    mic = np.array([B.db(np.sum(c.astype(np.float64) ** 2) + 1e-12) for c, _ in chunks])
    best, best_r = around, -2.0
    for lag in np.arange(max(0.0, around - 0.08), around + 0.08, 0.004):
        ref = np.array([B.db(np.sum(echo.segment(t - lag, len(c)).astype(np.float64) ** 2) + 1e-12)
                        for c, t in chunks])
        ok = ref > ref.max() - 40
        if ok.sum() < 20:
            continue
        r = float(np.corrcoef(mic[ok], ref[ok])[0, 1])
        if r > best_r:
            best, best_r = float(lag), r
    return best, best_r


def replay(chunks, echo, lag_s, coupling, noise):
    """False barge-ins the detector would have had on this recording (nobody talked: every one is false)."""
    det = B.BargeIn(echo, lag_s=lag_s, coupling=list(coupling), noise=list(noise))
    fires = 0
    for c, t in chunks:
        if det.feed(c, t):
            fires += 1
            det.reset()
    return fires


def record(args):
    from robot.mic import Ears, to_16k_mono
    from robot.power import AliveRunning, require_no_alive
    from robot.tts import Voice, set_echo_ref, SpeakerError

    try:
        require_no_alive()
    except AliveRunning as e:
        sys.exit(f"{e}\nstop it first: pkill -f \"code/[a]live.py\"")
    echo = B.EchoRef(keep_s=600.0)
    set_echo_ref(echo)
    voice = Voice()
    chunks, lock = [], threading.Lock()

    class Recorder(Ears):
        def _chunk(self, raw):                  # what alive.py's ears hear: 16 kHz, high-passed, un-gained
            t = self._chunk_time()
            c = self.highpass(to_16k_mono(raw, down=self.down))
            with lock:
                chunks.append((c, t))

    stop = threading.Event()
    ears = Recorder(vad=None).start(stop)
    print(f"Quiet please: {QUIET_S:.0f} s of the room, then the robot speaks (~30 s). Nobody talks.")
    time.sleep(QUIET_S + 0.5)
    with lock:
        quiet = [c for c, _ in chunks[8:]]      # the first chunks: arecord settling
    level = args.level if args.level is not None else voice.level()
    said = []
    for text in PHRASES:
        if not voice.cached(text) and not voice.key:
            continue
        try:
            src = voice.say_now(text, level=level)
        except SpeakerError as e:
            stop.set()
            sys.exit(f"speaker: {e}")
        if src != "silent":
            said.append((text, src))
            print(f"  {src:7} {text}")
        time.sleep(GAP_S)
    time.sleep(1.0)
    stop.set()
    ears.close()
    with lock:
        rec = list(chunks)
    return rec, quiet, echo, level, said


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=float, help="loudness 0..1 (default: the dashboard's speaker level)")
    ap.add_argument("--dry-run", action="store_true", help="measure and report only (no tuning file written)")
    args = ap.parse_args()
    if args.level is not None:
        args.level = min(max(args.level, 0.0), config.TTS_LEVEL_MAX)
    rec, quiet, echo, level, said = record(args)
    if len(said) < 3 or len(rec) < 200:
        sys.exit(f"too little was played ({len(said)} phrases, {len(rec)} chunks) - no Gemini key and nothing cached?")
    noise = np.median([B.db(B.band_power(c)) for c in quiet], axis=0)
    lag, r = best_lag(rec, echo, config.BARGE_LAG_S)
    print(f"echo lag: {lag * 1000:.0f} ms (config {config.BARGE_LAG_S * 1000:.0f} ms), loudness follows r = {r:.2f}")
    if abs(lag - config.BARGE_LAG_S) > 0.03 or r < 0.5:
        print("  the measured lag is uncertain - keeping the config's")
        lag = config.BARGE_LAG_S
    mic_db, tail_db = analyse(rec, echo, lag)
    old = B.load_tuning(config.BARGE_TUNE_FILE).get("coupling", list(config.BARGE_COUPLING_DB))
    coupling, used = B.fit_coupling(mic_db, tail_db, noise, default=old)
    before = replay(rec, echo, lag, old, noise)
    after = replay(rec, echo, lag, coupling, noise)
    edges = B.EDGES
    print("band (Hz)        " + " ".join(f"{a:>5}" for a in edges[:-1]))
    print("room floor (dB)  " + " ".join(f"{v:5.0f}" for v in noise))
    print("coupling before  " + " ".join(f"{v:5.1f}" for v in old))
    print("coupling now     " + " ".join(f"{v:5.1f}" for v in coupling))
    print("chunks used      " + " ".join(f"{v:5d}" for v in used))
    print(f"false barge-ins on this recording (nobody talked): {before} before -> {after} now")
    folder = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "recording.npz", mic=np.concatenate([c for c, _ in rec]),
                        t0=rec[0][1], played=echo.segment(rec[0][1], sum(len(c) for c, _ in rec)), lag_s=lag)
    summary = dict(level=level, lag_s=lag, lag_r=r, noise=[round(float(v), 2) for v in noise],
                   coupling_before=[round(float(v), 2) for v in old],
                   coupling=[round(float(v), 2) for v in coupling], used=[int(u) for u in used],
                   false_before=before, false_after=after, phrases=[t for t, _ in said])
    (folder / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"recording + summary: {folder}")
    if args.dry_run:
        print("--dry-run: tuning file not written")
        return
    ok = B.save_tuning(config.BARGE_TUNE_FILE, dict(
        coupling=summary["coupling"], noise=summary["noise"], lag_s=lag, how="calibrated", false_alarms=0,
        level=level, when=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(("written: " if ok else "could NOT write: ") + config.BARGE_TUNE_FILE +
          " (alive.py reads it at start: restart it)")


if __name__ == "__main__":
    main()
