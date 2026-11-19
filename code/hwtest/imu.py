#!/usr/bin/env python3
"""
Hardware test: MPU6050 IMU (GY-521) on the I2C bus, through robot/imu.py.

The robot (or just the mounted board) on a flat table. Motors are not used,
the battery switch may stay OFF if the Pi has USB power.

  python3 code/hwtest/imu.py              # all tests (~3 min)
  python3 code/hwtest/imu.py --only axes  # one or more tests, comma separated

Tests (in order):
  selftest  factory self-test: the chip moves its own sensors and compares
  still     5 s still: gravity = 1 g, gyro noise, sample rate, no I2C errors
  axes      flat + nose up -> which sensor axis is which robot axis (board is upside down)
  turn      turn the robot 90 deg left by hand and back -> yaw sign and scale
  drift     30 s still -> how fast yaw drifts
  tip       tilt the robot past 45 deg -> 'tipped' must switch on
  bump      tap the robot -> how strong a bump looks (to tune IMU_BUMP_G)

The report is printed and saved to ~/scratch/hwtest/.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (FAIL, INFO, PASS, SKIP, Report, cbreak, check_power, getkey,
                    heading, key_pressed)
from robot import config, imu
from robot.i2c import I2CDevice
from robot.state import State

STILL_S, DRIFT_S, ACTION_S = 5.0, 30.0, 15.0


class Ctx:
    """What the tests share: the device, the chip, the axis mapping in use."""
    dev = None
    chip = None
    axes = config.IMU_AXES
    who = None


def make_imu(ctx, calib_s=config.IMU_CALIB_S):
    m = imu.Imu(State(), dev=ctx.dev, axes=ctx.axes, filt=imu.ImuFilter(calib_s=calib_s))
    m.chip = ctx.chip
    return m


def run_for(m, seconds, until=None, show=None):
    """Step the IMU at its rate for `seconds` (or until until(reading) is true).
    Returns the readings. A key press ends it early when show is set."""
    readings, end = [], time.monotonic() + seconds
    next_t = time.monotonic()
    with cbreak():
        while time.monotonic() < end:
            r = m.step()
            if r is not None:
                readings.append(r)
                if show and len(readings) % 10 == 0:
                    print("\r   " + show(r) + "   ", end="", flush=True)
                if until and until(r):
                    break
            next_t += m.period
            if key_pressed(max(0.0, next_t - time.monotonic())) and show:
                break
    if show:
        print()
    return readings


def calibrated(ctx, rep):
    print("   keep the robot still - measuring the gyro bias ...")
    m = make_imu(ctx)
    run_for(m, config.IMU_CALIB_S * 4, until=lambda r: True)
    if m.filter.calibrating:
        rep.add("Gyro calibration", FAIL, f"robot not still ({m.filter.calib_failures} tries)")
        rep.hint("Calibration needs the robot completely still (no table bumps, "
                 "no motors, fan vibration is fine).")
        return None
    return m


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def sensor_gravity(ctx, n=50):
    """Average accel in the SENSOR frame (no mapping), g."""
    acc = [0.0, 0.0, 0.0]
    for _ in range(n):
        a, _, _ = imu.to_units(imu.decode(ctx.chip.read_raw()))
        acc = [s + v / n for s, v in zip(acc, a)]
        time.sleep(0.01)
    return acc


# ---------- tests ----------

def test_preflight(rep, ctx):
    heading("Preflight")
    ok, text = check_power()
    rep.add("Pi power (get_throttled)", SKIP if ok is None else PASS if ok else FAIL, text)
    if not os.path.exists(config.I2C_BUS):
        rep.add("I2C bus", FAIL, f"{config.I2C_BUS} missing")
        rep.hint("Enable I2C: sudo raspi-config -> Interface Options -> I2C, then reboot.")
        return False
    try:
        ctx.dev = I2CDevice(config.I2C_BUS, config.IMU_ADDR)
        ctx.chip = imu.Mpu6050(ctx.dev)
        ctx.who = ctx.chip.init()
    except OSError as e:
        rep.add(f"MPU6050 answers at 0x{config.IMU_ADDR:02x}", FAIL, str(e))
        rep.hint("No answer: check the 4 wires - VCC Pi pin 1 (3.3 V), GND Pi pin 25, "
                 "SDA breadboard col 2 (Pi pin 3), SCL col 3 (Pi pin 5). "
                 "SDA/SCL swapped is the most common mistake.")
        return False
    except imu.ImuError as e:
        rep.add("MPU6050 init", FAIL, str(e))
        return False
    kind = "genuine" if ctx.who == imu.WHO_AM_I_GENUINE else "clone"
    rep.add(f"MPU6050 answers at 0x{config.IMU_ADDR:02x}", PASS, f"WHO_AM_I 0x{ctx.who:02x} ({kind})")
    rep.add("Configuration written and read back", PASS,
            "100 Hz, low-pass 44 Hz, +-1000 deg/s, +-4 g")
    print("\nRobot on a flat table, nothing touching it?")
    return getkey("Ready?") == "y"


def test_selftest(rep, ctx):
    heading("Factory self-test")
    if ctx.who != imu.WHO_AM_I_GENUINE:
        rep.add("Self-test", SKIP, "clone chip: self-test is only defined for WHO_AM_I 0x68")
        return
    print("   keep the robot still ...")
    results = ctx.chip.self_test()
    bad = [r for r in results if not r.ok]
    for r in results:
        ch = "no trim" if r.change is None else f"{r.change * 100:+.1f} %"
        rep.fact(f"self-test {r.axis}: factory {r.factory:.0f}, measured {r.measured:.0f} LSB ({ch})")
    if bad:
        rep.add("Self-test (limit +-14 %)", FAIL, ", ".join(r.axis for r in bad))
        rep.hint("Self-test outside +-14 %: often a clone chip with an odd trim, or the "
                 "robot moved during the test. Repeat it still; if 'still', 'axes' and "
                 "'turn' pass, the chip is usable anyway.")
    else:
        rep.add("Self-test (limit +-14 %)", PASS, "all 6 axes")
    ctx.chip.init()


def test_still(rep, ctx):
    heading(f"Still for {STILL_S:.0f} s")
    m = calibrated(ctx, rep)
    if m is None:
        return
    t0 = time.monotonic()
    rs = run_for(m, STILL_S)
    elapsed = time.monotonic() - t0
    rate = len(rs) / elapsed
    g = mean([math.sqrt(sum(v * v for v in r.accel_g)) for r in rs])
    noise = max(std([r.gyro_dps[i] for r in rs]) for i in range(3))
    bias = max(abs(b) for b in m.filter.bias)
    temp = mean([r.temp_c for r in rs])

    rep.add("Sample rate", PASS if rate >= 90 else FAIL, f"{rate:.1f} Hz (want 100)")
    if rate < 90:
        rep.hint("Slow samples: the Pi is busy, or the I2C bus is slow / glitchy.")
    rep.add("I2C errors", PASS if m.total_errors == 0 else FAIL, f"{m.total_errors}")
    if m.total_errors:
        rep.hint("I2C errors while still: loose jumper on SDA/SCL, or wires too long.")
    rep.add("Gravity magnitude", PASS if 0.9 <= g <= 1.1 else FAIL, f"{g:.3f} g (want ~1.00)")
    rep.add("Gyro noise at rest", PASS if noise < 0.5 else FAIL, f"{noise:.2f} deg/s std (want < 0.5)")
    if noise >= 0.5:
        rep.hint("High gyro noise: something vibrates (fan?) or the board is loose on its mount.")
    rep.add("Gyro bias", PASS if bias < 20 else FAIL, f"largest {bias:.1f} deg/s (removed in software)")
    rep.add("Temperature", PASS if 0 < temp < 60 else FAIL, f"{temp:.1f} C")
    rep.fact(f"gyro bias {tuple(round(b, 2) for b in m.filter.bias)} deg/s, robot frame")


def biggest_axis(v):
    i = max(range(3), key=lambda k: abs(v[k]))
    return i, (1 if v[i] > 0 else -1)


def test_axes(rep, ctx):
    heading("Axes (which sensor axis is which robot axis)")
    print("1) Robot standing FLAT on its wheels.")
    getkey("Flat and still?", "y")
    flat = sensor_gravity(ctx)
    z_i, z_s = biggest_axis(flat)
    if abs(flat[z_i]) < 0.8:
        rep.add("Axes: flat", FAIL, f"no clear gravity axis {flat}")
        return
    print("2) Lift the FRONT (caster side) about 30-45 deg, wheels stay on the table.")
    getkey("Holding it nose up and still?", "y")
    up = sensor_gravity(ctx)
    rest = [up[k] if k != z_i else 0.0 for k in range(3)]
    y_i, y_s = biggest_axis(rest)
    if abs(rest[y_i]) < 0.3:
        rep.add("Axes: nose up", FAIL, f"not tilted enough {[round(v, 2) for v in up]}")
        rep.hint("Lift the caster side higher (at least 20 deg) and hold it still.")
        return
    # nose up: gravity gains a +Y (forward) component in the robot frame
    names = "xyz"
    z = f"{'+' if z_s > 0 else '-'}{names[z_i]}"
    y = f"{'+' if y_s > 0 else '-'}{names[y_i]}"
    x_i = 3 - z_i - y_i
    for sign in "+-":
        try:
            found = (f"{sign}{names[x_i]}", y, z)
            imu.parse_axes(found)
            break
        except ValueError:
            continue
    rep.fact(f"sensor gravity flat {[round(v, 2) for v in flat]}, nose up {[round(v, 2) for v in up]}")
    if tuple(found) == tuple(config.IMU_AXES):
        rep.add("Axes match config.IMU_AXES", PASS, str(found))
    else:
        rep.add("Axes match config.IMU_AXES", FAIL, f"config {config.IMU_AXES}, measured {found}")
        rep.hint(f"Paste into code/robot/config.py:  IMU_AXES = {found!r}")
    if z_s > 0:
        rep.fact("Gravity is on +z of the sensor: the board is chip-UP, not upside down as documented.")
    ctx.axes = found


def test_turn(rep, ctx):
    heading("Turn by hand")
    print("Put a mark (tape, table edge) along the robot's side so you can see 90 deg.")
    m = calibrated(ctx, rep)
    if m is None:
        return
    print("Turn the robot 90 deg LEFT (counter-clockwise from above), then press a key.")
    rs = run_for(m, 30, show=lambda r: f"yaw {r.yaw_deg:+7.1f} deg")
    yaw = rs[-1].yaw_deg if rs else 0.0
    ok = 85 <= yaw <= 95
    rep.add("Turn 90 left", PASS if ok else FAIL, f"yaw {yaw:+.1f} deg (want +85..+95)")
    if -95 <= yaw <= -85:
        rep.hint("Yaw has the wrong sign: run the 'axes' test and paste its IMU_AXES line.")
    print("Now turn it back to the mark, then press a key.")
    rs = run_for(m, 30, show=lambda r: f"yaw {r.yaw_deg:+7.1f} deg")
    back = rs[-1].yaw_deg if rs else yaw
    rep.add("Turn back to start", PASS if abs(back) <= 3 else FAIL, f"yaw {back:+.1f} deg (want 0 +-3)")
    sat = rs[-1].saturated if rs else 0
    if sat:
        rep.fact(f"{sat} clipped samples while turning (turned faster than 1000 deg/s?)")


def test_drift(rep, ctx):
    heading(f"Drift ({DRIFT_S:.0f} s still)")
    m = calibrated(ctx, rep)
    if m is None:
        return
    rs = run_for(m, DRIFT_S, show=lambda r: f"yaw {r.yaw_deg:+6.2f} deg")
    per_min = (rs[-1].yaw_deg - rs[0].yaw_deg) / (rs[-1].t - rs[0].t) * 60
    rep.add("Yaw drift", PASS if abs(per_min) < 5 else FAIL, f"{per_min:+.2f} deg/min (want < 5)")


def test_tip(rep, ctx):
    heading("Tipped detection")
    m = calibrated(ctx, rep)
    if m is None:
        return
    print(f"Tilt the robot slowly onto its side (past {config.IMU_TIPPED_DEG:.0f} deg) "
          f"within {ACTION_S:.0f} s.")
    rs = run_for(m, ACTION_S, until=lambda r: r.tipped,
                 show=lambda r: f"tilt {r.tilt_deg:5.1f} deg")
    tipped = bool(rs) and rs[-1].tipped
    rep.add("Tipped switches on", PASS if tipped else FAIL,
            f"at {rs[-1].tilt_deg:.0f} deg" if tipped else "never")
    print("Put it back on its wheels.")
    getkey("Back flat?", "y")


def test_bump(rep, ctx):
    heading("Bump strength")
    m = calibrated(ctx, rep)
    if m is None:
        return
    print(f"Within {ACTION_S:.0f} s: tap the FRONT of the robot firmly with a finger, "
          "like a gentle crash. Press a key when done.")
    rs = run_for(m, ACTION_S, show=lambda r: f"bump {r.bump_g:4.2f} g")
    peak = max((r.bump_g for r in rs), default=0.0)
    seen = any(r.last_bump_t is not None for r in rs)
    rep.add("Bump measured", INFO, f"peak {peak:.2f} g, threshold {config.IMU_BUMP_G} g, "
            f"{'detected' if seen else 'not detected'}")
    rep.fact(f"tap peak {peak:.2f} g. Later: drive into a soft wall and compare; "
             f"IMU_BUMP_G must sit above normal driving and below a crash.")


TESTS = {"selftest": test_selftest, "still": test_still, "axes": test_axes,
         "turn": test_turn, "drift": test_drift, "tip": test_tip, "bump": test_bump}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma separated: " + ",".join(TESTS))
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        ap.error(f"unknown test(s): {', '.join(unknown)}")

    rep, ctx = Report("imu"), Ctx()
    try:
        if test_preflight(rep, ctx):
            for n in names:
                TESTS[n](rep, ctx)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        rep.add("Run", SKIP, "aborted with Ctrl+C")
    finally:
        if ctx.dev:
            ctx.dev.close()

    print("\n" + rep.text())
    print(f"\nSaved: {rep.save()}")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
