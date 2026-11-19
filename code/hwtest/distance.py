#!/usr/bin/env python3
"""
Hardware test: HC-SR04 ultrasonic distance sensor.

SAFE first test - power the sensor from the Pi's 3.3 V. ECHO can never go
above its own supply, so at 3.3 V it is safe on a Pi pin WITHOUT the divider:

    HC-SR04 VCC  -> Pi 3.3 V (pin 1 - on the robot via breadboard col 4)
    HC-SR04 GND  -> Pi pin 30 (GND)
    HC-SR04 TRIG -> Pi pin 29 (GPIO5)
    HC-SR04 ECHO -> Pi pin 31 (GPIO6)

If it works at 3.3 V, it can stay like that (no divider, no 4.7 V rail).
If it does not answer at 3.3 V it is the 5 V-only version: then ONLY power it
from 4.7-5 V WITH the 1 kOhm / 2 kOhm divider on ECHO (docs/wiring.md).

  python3 code/hwtest/distance.py             # all tests (~2 min)
  python3 code/hwtest/distance.py --only live

Tests: alive (does it answer), distance (20 / 50 / 100 cm vs a ruler),
noise (spread of readings), live (show the distance for 15 s).
You need a flat object (book, box, wall) and a ruler or tape measure.
Measure from the FRONT of the two silver cans.
"""
import argparse
import statistics
import sys
import time

from gpiozero import DigitalInputDevice, DigitalOutputDevice

from common import (FAIL, INFO, PASS, SKIP, Report, cbreak, check_power, getkey,
                    heading, key_pressed, release_pins)

TRIG, ECHO = 5, 6            # GPIO numbers (pins 29 / 31), docs/pinout.md
SOUND_M_S = 343.0            # speed of sound at ~20 C
RISE_TIMEOUT_S = 0.03        # no echo start within this = sensor did not answer
ECHO_TIMEOUT_S = 0.04        # old HC-SR04 holds ECHO ~38 ms when nothing is in range
PING_GAP_S = 0.06            # wait between pings so old echoes die out
DISTANCES_CM = (20, 50, 100)
TOLERANCE_CM = 2.0


class Sensor:
    def __init__(self):
        self.trig = DigitalOutputDevice(TRIG, initial_value=False)
        self.echo = DigitalInputDevice(ECHO, pull_up=None, active_state=True)

    def ping(self):
        """Distance in cm, 'noanswer' (ECHO never rose) or 'nothing' (no object in range)."""
        echo = self.echo
        if echo.value:                       # still high from the last ping
            time.sleep(ECHO_TIMEOUT_S)
        self.trig.on()
        time.sleep(0.00001)                  # >= 10 us trigger pulse
        self.trig.off()
        t0 = time.perf_counter()
        while not echo.value:
            if time.perf_counter() - t0 > RISE_TIMEOUT_S:
                return "noanswer"
        start = time.perf_counter()
        while echo.value:
            if time.perf_counter() - start > ECHO_TIMEOUT_S:
                return "nothing"
        return (time.perf_counter() - start) * SOUND_M_S / 2 * 100

    def pings(self, n):
        out = []
        for _ in range(n):
            out.append(self.ping())
            time.sleep(PING_GAP_S)
        return out

    def close(self):
        self.trig.off()
        self.trig.close()
        self.echo.close()


def numbers(results):
    return [r for r in results if isinstance(r, float)]


def test_preflight(rep):
    heading("Preflight")
    ok, text = check_power()
    rep.add("Pi power (get_throttled)", SKIP if ok is None else PASS if ok else FAIL, text)
    print("Which supply is the HC-SR04 on?\n"
          "  3 = Pi 3.3 V (pin 17), ECHO straight to pin 31      <- safe first test\n"
          "  5 = 4.7-5 V, ECHO through the 1k/2k divider to pin 31\n"
          "  n = neither / not sure -> stop")
    k = getkey("Supply?", "35n")
    if k == "n":
        rep.add("Wiring", SKIP, "not sure - aborted (never put 5 V-powered ECHO straight on the Pi)")
        return None
    if k == "5" and getkey("Divider in place (1 kOhm from ECHO, 2 kOhm to GND)?") != "y":
        rep.add("Wiring", SKIP, "5 V without divider - aborted to protect the Pi")
        return None
    supply = "3.3 V" if k == "3" else "5 V + divider"
    rep.fact(f"supply: {supply}")
    try:
        return supply, Sensor()
    except Exception as e:
        rep.add("Pins GPIO5/6 free", FAIL, str(e))
        rep.hint("Another program holds GPIO5/6.")
        return None


def test_alive(rep, sensor, supply):
    heading("Alive")
    print("Point the sensor at something 20-100 cm away.")
    getkey("Ready?", "y")
    res = sensor.pings(20)
    answered = sum(r != "noanswer" for r in res)
    rep.add("Sensor answers", PASS if answered >= 18 else FAIL, f"{answered}/20 pings")
    if answered < 18:
        if supply == "3.3 V":
            rep.hint("No answer at 3.3 V: this is the 5 V-only HC-SR04. Power it from the "
                     "4.7 V rail (or Pi 5 V for a bench test) and ALWAYS use the 1k/2k divider on ECHO.")
        else:
            rep.hint("No answer: check VCC/GND, TRIG on pin 29, ECHO via the divider on pin 31.")
    return answered >= 18


def test_distance(rep, sensor, supply):
    heading("Distance vs ruler")
    for cm in DISTANCES_CM:
        print(f"\nPut a flat object (book, box) exactly {cm} cm in front of the sensor, square to it.")
        getkey("Placed?", "y")
        res = sensor.pings(15)
        vals = numbers(res)
        name = f"Distance {cm} cm"
        if len(vals) < 10:
            rep.add(name, FAIL, f"only {len(vals)}/15 readings ({res.count('nothing')} 'nothing', "
                    f"{res.count('noanswer')} no answer)")
            if supply == "3.3 V":
                rep.hint(f"Weak at {cm} cm on 3.3 V: the sensor runs, but too weakly - "
                         "use 4.7 V + divider for the robot.")
            continue
        med = statistics.median(vals)
        ok = abs(med - cm) <= TOLERANCE_CM
        rep.add(name, PASS if ok else FAIL, f"median {med:.1f} cm ({len(vals)}/15 readings)")
        if not ok:
            rep.hint("Distance off: measure from the front of the cans; object flat and square; "
                     "a soft object (cloth) swallows the sound.")


def test_noise(rep, sensor, supply):
    heading("Noise")
    print("Object at about 50 cm, nothing moving.")
    getkey("Ready?", "y")
    vals = numbers(sensor.pings(30))
    if len(vals) < 20:
        rep.add("Noise", FAIL, f"only {len(vals)}/30 readings")
        return
    spread = max(vals) - min(vals)
    rep.add("Noise", PASS if spread <= 2.0 else FAIL,
            f"spread {spread:.1f} cm, std {statistics.pstdev(vals):.2f} cm (want spread <= 2)")


def test_live(rep, sensor, supply):
    heading("Live (15 s, any key stops)")
    end = time.monotonic() + 15
    with cbreak():
        while time.monotonic() < end:
            r = sensor.ping()
            text = f"{r:6.1f} cm" if isinstance(r, float) else f"{r:>9}"
            print(f"\r   {text}   ", end="", flush=True)
            if key_pressed(PING_GAP_S):
                break
    print()
    rep.add("Live view", INFO, "done")


TESTS = {"alive": test_alive, "distance": test_distance, "noise": test_noise, "live": test_live}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma separated: " + ",".join(TESTS))
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        ap.error(f"unknown test(s): {', '.join(unknown)}")

    rep, sensor, got = Report("distance"), None, None
    try:
        got = test_preflight(rep)
        if got:
            supply, sensor = got
            for n in names:
                if not TESTS[n](rep, sensor, supply) and n == "alive":
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")
        rep.add("Run", SKIP, "aborted with Ctrl+C")
    finally:
        if sensor:
            sensor.close()
        release_pins([TRIG, ECHO])

    if got and got[0] == "3.3 V" and not rep.failed:
        rep.fact("Works on 3.3 V: it can stay on Pi 3.3 V with ECHO direct - no divider needed. "
                 "Tell Claude so the docs and the breadboard plan get updated.")
    print("\n" + rep.text())
    print(f"\nSaved: {rep.save()}")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
