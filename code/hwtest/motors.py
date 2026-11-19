#!/usr/bin/env python3
"""
Hardware test: L298N motor driver + both TT motors.

Put the robot ON A BOX so both wheels are in the air, battery switch ON.
Quit robot_drive.py / robot_test.py first (they hold the same pins).

  python3 code/hwtest/motors.py              # all tests (~6 min)
  python3 code/hwtest/motors.py --only start # one or more tests, comma separated

Tests (in order):
  wiring    each wheel, each direction, alone, full power   -> is every wire where it should be?
  enable    IN set but EN off must NOT spin; EN on must spin -> are ENA/ENB really connected?
  wiggle    both wheels run 15 s while you wiggle the jumpers -> loose Dupont contact?
  start     speed ramp, you press a key when the wheel starts -> lowest speed that works
  sequence  w/a/s/d-style command changes at full and at 0.5 -> reproduces robot_drive.py

"Forward" for a wheel in the air: the TOP of the wheel rolls towards the
FRONT of the robot (the caster ball side).

Ctrl+C stops the motors at any time. The report is printed and saved to
~/scratch/hwtest/.
"""
import argparse
import sys
from contextlib import contextmanager
from time import sleep

from gpiozero import DigitalOutputDevice, Motor, Robot

from common import (FAIL, INFO, PASS, SKIP, Report, ask_text, cbreak,
                    check_power, flush_input, getkey, heading, key_pressed,
                    release_pins, require_battery)

# Same wiring as robot_drive.py (docs/pinout.md).
PINS = {"ENA": 17, "IN1": 27, "IN2": 22, "IN3": 23, "IN4": 24, "ENB": 25}
WHEELS = {  # wheel: (enable, forward IN, backward IN)
    "LEFT": ("ENA", "IN1", "IN2"),
    "RIGHT": ("ENB", "IN3", "IN4"),
}
DRIVE_SPEED = 0.5      # the speed robot_drive.py starts with
DRIVE_MIN_SPEED = 0.2  # lowest speed robot_drive.py allows with '-'

RAMP_START, RAMP_STEP, RAMP_HOLD_S = 0.15, 0.05, 1.5

# gpiozero Robot: left = left wheel backward + right forward (spin on the spot).
COMMANDS = {
    "forward": "LEFT forward  + RIGHT forward",
    "backward": "LEFT backward + RIGHT backward",
    "left": "LEFT backward + RIGHT forward",
    "right": "LEFT forward  + RIGHT backward",
}
SEQUENCE = ["forward", "left", "forward", "right", "backward", "left", "right", "forward"]


# ---------- pin access ----------

@contextmanager
def raw_pins():
    """Each L298N input as a plain on/off pin (no PWM), all starting LOW."""
    devs = {}
    try:
        for name, pin in PINS.items():
            devs[name] = DigitalOutputDevice(pin, initial_value=False)
        yield devs
    finally:
        for d in devs.values():
            d.off()
            d.close()


def all_off(devs):
    for d in devs.values():
        d.off()


def drive_wheel(devs, wheel, direction, enable=True):
    en, fwd, bwd = WHEELS[wheel]
    devs[fwd].value = direction == "forward"
    devs[bwd].value = direction == "backward"
    devs[en].value = enable


def make_motor(wheel):
    """The motor exactly as robot_drive.py builds it (PWM on the IN pins)."""
    en, fwd, bwd = WHEELS[wheel]
    return Motor(forward=PINS[fwd], backward=PINS[bwd], enable=PINS[en])


# ---------- tests ----------

def test_preflight(rep):
    heading("Preflight")
    ok, text = check_power()
    if ok is None:
        rep.add("Pi power (get_throttled)", SKIP, text)
    else:
        rep.add("Pi power (get_throttled)", PASS if ok else FAIL, text)
        if not ok:
            rep.hint("Pi under-voltage: measure the Pi buck output (should be 5.1 V) "
                     "while the motors run; a sagging battery can drag it down.")

    try:
        with raw_pins():
            pass
        rep.add("Motor pins free", PASS)
    except Exception as e:
        rep.add("Motor pins free", FAIL, str(e))
        rep.hint("Another program holds the motor pins - quit robot_drive.py / robot_test.py.")
        return False

    print("\nRobot on a box, BOTH wheels in the air, battery switch ON, L298N red LED lit?")
    if getkey("Ready?") != "y":
        rep.add("Operator setup", SKIP, "not ready - aborted")
        return False
    v = ask_text("Battery voltage if you measured it (e.g. 7.9), or Enter to skip:")
    if v:
        rep.fact(f"Battery voltage at start: {v} V (2S LiPo: 8.4 full, 7.4 nominal, "
                 "below ~7.0 recharge)")
    return True


def test_wiring(rep):
    heading("Wiring: one wheel, one direction, full power (no PWM)")
    print("Watch which wheel moves and which way. 'r' repeats the move.")
    with raw_pins() as devs:
        for wheel in WHEELS:
            for direction in ("forward", "backward"):
                name = f"Wiring {wheel} {direction}"
                while True:
                    print(f"\n-> expect: ONLY the {wheel} wheel, {direction.upper()}")
                    drive_wheel(devs, wheel, direction)
                    sleep(1.5)
                    all_off(devs)
                    k = getkey(f"Only {wheel} spun {direction}?", "ynr")
                    if k != "r":
                        break
                if k == "y":
                    rep.add(name, PASS)
                else:
                    seen = ask_text("What did you see? (e.g. 'nothing', 'right wheel', 'backward')")
                    rep.add(name, FAIL, seen)
                    en, fwd, bwd = WHEELS[wheel]
                    out = "OUT1/OUT2" if wheel == "LEFT" else "OUT3/OUT4"
                    rep.hint(f"{wheel} {direction}: check jumpers {en}, {fwd}, {bwd} at both ends "
                             f"and the motor wires in {out}. Wrong direction -> swap the two "
                             f"motor wires in {out}. Other wheel moved -> motors swapped.")


def test_enable(rep):
    heading("Enable lines (ENA / ENB)")
    print("For each wheel: 2 s with IN set but EN OFF (must stay STILL),\n"
          "then 2 s with EN ON (must SPIN forward).")
    with raw_pins() as devs:
        for wheel in WHEELS:
            en = WHEELS[wheel][0]
            print(f"\n-> {wheel} wheel: phase 1, {en} OFF - must stay still")
            drive_wheel(devs, wheel, "forward", enable=False)
            sleep(2)
            print(f"-> {wheel} wheel: phase 2, {en} ON - must spin")
            devs[en].on()
            sleep(2)
            all_off(devs)
            k = getkey(f"Still in phase 1 AND spinning in phase 2?")
            if k == "y":
                rep.add(f"Enable {en}", PASS)
            else:
                seen = ask_text("What did you see? (e.g. 'spun in both', 'never spun')")
                rep.add(f"Enable {en}", FAIL, seen)
                rep.hint(f"{en}: spun in phase 1 -> the {en} jumper cap is still on, or the wire "
                         f"is on the 5 V pin instead of the signal pin (the one closer to IN1-IN4). "
                         f"Never spun -> {en} wire loose or on the wrong Pi pin "
                         f"(should be GPIO{PINS[en]}).")


def test_wiggle(rep):
    heading("Wiggle test (loose contacts)")
    print("Both wheels will run forward at full power for 15 s.\n"
          "Meanwhile gently wiggle each of the 7 jumper wires (Pi end and L298N end)\n"
          "and the motor wires in the screw terminals. Watch for any stutter or stop.")
    getkey("Start?", "y")
    with raw_pins() as devs:
        for wheel in WHEELS:
            drive_wheel(devs, wheel, "forward")
        for left in range(15, 0, -1):
            print(f"\r   {left:2d} s ", end="", flush=True)
            sleep(1)
        all_off(devs)
    print()
    if getkey("Did both wheels spin steadily the whole time?") == "y":
        rep.add("Wiggle", PASS)
    else:
        seen = ask_text("Which wire made it stutter? (Enter if unsure)")
        rep.add("Wiggle", FAIL, seen or "stutter, wire unknown")
        rep.hint("Loose contact: re-seat or replace that jumper (Dupont sockets wear out); "
                 "re-tighten the L298N screw terminals.")


def ramp(motor, direction):
    """Raise the speed step by step until a key is pressed; return that speed."""
    speed = RAMP_START
    flush_input()
    with cbreak():
        while speed <= 1.0 + 1e-9:
            getattr(motor, direction)(speed)
            print(f"\r   speed {speed:.2f} ", end="", flush=True)
            if key_pressed(RAMP_HOLD_S):
                motor.stop()
                print("<- started")
                return speed
            speed = round(speed + RAMP_STEP, 2)
    motor.stop()
    print("<- never started")
    return None


def test_start(rep):
    heading("Start speed (dead band)")
    print("The speed rises slowly, the same way robot_drive.py drives (PWM).\n"
          "Press ANY key the moment the wheel starts turning on its own.")
    worst = 0.0
    for wheel in WHEELS:
        motor = make_motor(wheel)
        try:
            for direction in ("forward", "backward"):
                print(f"\n-> {wheel} wheel {direction}")
                getkey("Start ramp?", "y")
                s = ramp(motor, direction)
                sleep(0.5)
                name = f"Start speed {wheel} {direction}"
                if s is None:
                    rep.add(name, FAIL, "did not start even at 1.0")
                    rep.hint(f"{wheel} wheel never started: run the wiring test; check the "
                             f"battery voltage and that the motor/gearbox turns freely by hand.")
                    worst = 1.0
                else:
                    rep.add(name, INFO, f"starts at {s:.2f}")
                    worst = max(worst, s)
        finally:
            motor.stop()
            motor.close()

    rep.fact(f"Highest start speed (wheels in the air): {worst:.2f}. On the floor it is "
             f"higher (weight + caster friction), spin turns need the most.")
    ok = round(worst + 0.1, 2) <= DRIVE_SPEED
    rep.add("Start margin vs robot_drive 0.5", PASS if ok else FAIL,
            f"needs >= {worst + 0.1:.2f} incl. 0.10 margin")
    if not ok:
        rep.hint(f"Speed {DRIVE_SPEED} is too close to (or below) where the motors start. "
                 f"That alone explains 'sometimes nothing / only one wheel spins'. "
                 f"Raise robot_drive.py's start speed and minimum to about "
                 f"{min(1.0, worst + 0.15):.2f}.")
    if worst > DRIVE_MIN_SPEED:
        rep.fact(f"robot_drive.py lets '-' go down to {DRIVE_MIN_SPEED}, below the start "
                 f"speed - at that setting the wheels cannot move.")


def run_sequence(rep, robot, speed):
    print(f"\n-> sequence at speed {speed:.1f}: commands change WITHOUT stopping, "
          f"like keys in robot_drive.py")
    getkey("Start?", "y")
    wrong = []
    for i, cmd in enumerate(SEQUENCE, 1):
        getattr(robot, cmd)(speed)
        print(f"   {i}/{len(SEQUENCE)} {cmd:8} expect: {COMMANDS[cmd]}")
        if getkey("   correct?") != "y":
            wrong.append(f"#{i} {cmd}")
    robot.stop()
    name = f"Sequence speed {speed:.1f}"
    if wrong:
        rep.add(name, FAIL, f"{len(wrong)}/{len(SEQUENCE)} wrong: {', '.join(wrong)}")
    else:
        rep.add(name, PASS)
    return not wrong


def test_sequence(rep):
    heading("Command sequence (what robot_drive.py does)")
    robot = Robot(left=make_motor("LEFT"), right=make_motor("RIGHT"))
    try:
        full_ok = run_sequence(rep, robot, 1.0)
        drive_ok = run_sequence(rep, robot, DRIVE_SPEED)
    finally:
        robot.stop()
        robot.close()
    if full_ok and not drive_ok:
        rep.hint("Fine at full speed but not at 0.5 -> not wiring, the speed is too low "
                 "(see the start speed result).")
    elif not full_ok:
        rep.hint("Fails even at full speed -> wiring/contact problem; run the wiring, "
                 "enable and wiggle tests and fix those first.")


TESTS = {
    "wiring": test_wiring,
    "enable": test_enable,
    "wiggle": test_wiggle,
    "start": test_start,
    "sequence": test_sequence,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma separated: " + ",".join(TESTS))
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        ap.error(f"unknown test(s): {', '.join(unknown)}")

    require_battery("the motor test")          # no battery: never touch the motor pins
    rep = Report("motors")
    try:
        if test_preflight(rep):
            for n in names:
                TESTS[n](rep)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        rep.add("Run", SKIP, "aborted with Ctrl+C")
    finally:
        release_pins(PINS.values())

    print("\n" + rep.text())
    print(f"\nSaved: {rep.save()}")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
