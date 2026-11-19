#!/usr/bin/env python3
"""
Make the robot say something (robot/tts.py: Gemini voice, cached; silent without key / internet / quota).
The sound card can have one owner: stop alive.py first (`pkill -f code/alive.py`).

    python3 code/say.py "Merhaba Selim, ben Arcby."          # volume by the time of day (quiet at night)
    python3 code/say.py --level 0.5 "Good evening."          # this loudness (sample peak, 0..1)
    python3 code/say.py --prepare phrases.txt                # make one phrase per line ahead (nothing played)
    python3 code/say.py --usage                              # Gemini requests used today
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import config                                  # noqa: E402
from robot.tts import SpeakerError, Voice                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="what to say")
    ap.add_argument("--level", type=float, help="loudness 0..1 (default: day / night level)")
    ap.add_argument("--prepare", metavar="FILE", help="cache the phrases in FILE (one per line)")
    ap.add_argument("--usage", action="store_true", help="Gemini requests used today")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    voice = Voice()
    if not voice.key:
        print(f"no GEMINI_API_KEY in {config.SECRETS_FILE}: only phrases already cached can be said")
    if args.usage:
        day, n = voice.usage()
        print(f"{day}: {n} of {voice.daily} Gemini requests")
    if args.prepare:
        with open(args.prepare) as f:
            phrases = [line.strip() for line in f if line.strip()]
        for text, src in zip(phrases, voice.prepare(phrases)):
            print(f"{src:7} {text}")
    if args.text:
        try:
            print(voice.say_now(" ".join(args.text), level=args.level))
        except SpeakerError as e:
            sys.exit(f"{e}\n(is alive.py running? it owns the speaker: pkill -f code/alive.py)")


if __name__ == "__main__":
    main()
