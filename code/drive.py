#!/usr/bin/env python3
"""
Keyboard drive on top of the robot software core (robot/main.py).

HOLD a key to drive, let go and it stops (within 0.6 s):
  w forward        s backward       a spin left      d spin right
  q curve left     e curve right    space stop now
  + / -  speed level 0.1-1.0        r  reset a safety stop (bump / tipped)
  x or Ctrl+C  quit

Safe by design: the motors stop when no key is held, when the SSH connection
drops, when the IMU stops answering, when the robot tips over or bumps into
something, and when this program exits or crashes. Straight driving (w / s)
holds the heading with the IMU; every start is a soft start.

Keep the robot still for the first 2 s (IMU calibration).

  python3 code/drive.py          # the real robot
  python3 code/drive.py --sim    # simulated robot, nothing moves

Log: ~/scratch/drive.log
"""
import argparse
import logging
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from robot import config, imu                 # noqa: E402
from robot.main import Robot                  # noqa: E402
from robot.teleop import Teleop               # noqa: E402

LOG = Path.home() / "scratch" / "drive.log"
UI_HZ = 20


def sim_hardware():
    sys.path.insert(0, str(CODE / "hwtest"))
    from base import SimDrive, SimRobot
    sim = SimRobot()
    return SimDrive(sim, 1.0), sim.imu_device(imu.parse_axes(config.IMU_AXES))


def status_line(robot, cmd_name, tele, note):
    st = robot.state
    r = st.get("imu", max_age=0.5)
    m = st.get("motors") or {}
    s = st.get("safety") or {}
    out = m.get("applied", (0.0, 0.0))
    if s.get("stop"):
        safe = "STOP: " + ", ".join(s.get("reasons", []))
        if s.get("latched"):
            safe += " (r = reset)"
    else:
        safe = "ok"
    imu_txt = f"yaw {r.yaw_deg:+7.1f} tilt {r.tilt_deg:4.1f}" if r else "imu --"
    return (f"\r speed {tele.speed:.1f} | {cmd_name:11} | L {out[0]:+.2f} R {out[1]:+.2f} | "
            f"{imu_txt} | {safe}{' | ' + note if note else ''}\x1b[K")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="simulated robot, no motors / IMU needed")
    args = ap.parse_args()
    if not sys.stdin.isatty():
        sys.exit("drive.py needs a real terminal (run it in your SSH window)")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    log = logging.getLogger("arcby")
    driver = dev = power = None
    if args.sim:
        driver, dev = sim_hardware()
        power = lambda: (False, False, "sim")          # noqa: E731 - the simulation has no battery
    print(__doc__.split("Safe by design")[0].rstrip())
    print("\nStarting... keep the robot still (IMU calibration).\n")

    tele = Teleop()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    note, note_until = "", 0.0
    last_name = None
    try:
        with Robot(driver=driver, imu_dev=dev, power_reader=power) as robot:
            log.info("drive.py started (sim=%s)", args.sim)
            tty.setcbreak(fd)
            nxt = time.monotonic()
            while robot.alive and not tele.want_quit:
                now = time.monotonic()
                while select.select([sys.stdin], [], [], 0)[0]:
                    ch = os.read(fd, 1).decode(errors="ignore")
                    txt = tele.press(ch, now)
                    if txt:
                        note, note_until = txt, now + 1.5
                if tele.want_reset:
                    tele.want_reset = False
                    robot.safety.reset()
                    log.info("safety reset by operator")
                left, right, name = tele.command(now)
                robot.motors.request(left, right, "keyboard")
                if name != last_name:
                    log.info("command %s (%.2f, %.2f) speed %.1f", name, left, right, tele.speed)
                    last_name = name
                print(status_line(robot, name, tele, note if now < note_until else ""),
                      end="", flush=True)
                nxt += 1.0 / UI_HZ
                time.sleep(max(0.0, nxt - time.monotonic()))
            if robot.fault:
                print(f"\n\nSTOPPED: {robot.fault} (details in {LOG})")
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f"\nCould not start: {e}\nIs the MPU6050 connected (pins 1/3/5/25)? Details in {LOG}")
        log.error("start failed: %s", e)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\nmotors stopped, bye")
        log.info("drive.py exit")


if __name__ == "__main__":
    main()
