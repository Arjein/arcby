#!/usr/bin/env python3
"""
IMU level: measure gravity with the robot standing still on a flat floor and print config.IMU_LEVEL_UP, so the
robot's normal stance reads 0 deg tilt (robot/imu.py rotates every sample by it). Run it after re-gluing the IMU
or changing the wheels / caster. Runs on the Pi's own power (no motors, no head).

    python3 code/hwtest/imu_level.py            # 5 s, robot still on a flat floor
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot import config                                  # noqa: E402
from robot.imu import Imu                                 # noqa: E402
from robot.state import State                             # noqa: E402


def measure(level_up, seconds):
    imu = Imu(State(), level_up=level_up)
    imu.start()
    out, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds + config.IMU_CALIB_S:
        r = imu.step()
        if r is not None:
            out.append(r)
        time.sleep(1 / config.IMU_RATE_HZ)
    imu.dev.close()
    if len(out) < 50:
        sys.exit("too few IMU samples - is the robot still?")
    return out


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    print(f"keep the robot still on a flat floor for {seconds + config.IMU_CALIB_S:.0f} s...")
    raw = measure(None, seconds)
    up = tuple(round(statistics.median(r.accel_g[i] for r in raw), 3) for i in range(3))
    tilt = statistics.median(r.tilt_deg for r in raw)
    print(f"untrimmed: tilt {tilt:.2f} deg (roll {statistics.median(r.roll_deg for r in raw):+.2f}, "
          f"pitch {statistics.median(r.pitch_deg for r in raw):+.2f})")
    now = config.IMU_LEVEL_UP
    if now is not None:
        trimmed = measure(now, 2.0)
        print(f"with the current IMU_LEVEL_UP {now}: tilt {statistics.median(r.tilt_deg for r in trimmed):.2f} deg")
    print(f"\nput this in code/robot/config.py:\nIMU_LEVEL_UP = {up}")


if __name__ == "__main__":
    main()
