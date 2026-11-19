"""The whole system with the simulated robot (hwtest/base.py SimRobot) instead of hardware:
real Robot core (imu + safety + motors threads), real DriveServer, real HTTP - like a browser."""
import json
import logging
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "hwtest"))

import drive_web                                    # noqa: E402
from base import SimDrive, SimRobot                 # noqa: E402
from robot import config, imu                       # noqa: E402
from robot.main import Robot                        # noqa: E402


class SystemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.sim = SimRobot()
        cls.robot = Robot(driver=SimDrive(cls.sim, 1.0),
                          imu_dev=cls.sim.imu_device(imu.parse_axes(config.IMU_AXES)),
                          imu_filter=imu.ImuFilter(calib_s=0.3),
                          power_reader=lambda: (False, False, "sim"), blackbox_dir=None)
        cls.robot.__enter__()
        cls.web = drive_web.DriveServer(cls.robot, "key", host="127.0.0.1", port=0)
        cls.web.serve_in_thread()
        end = time.monotonic() + 3
        while time.monotonic() < end and (cls.robot.state.get("safety") or {"stop": True})["stop"]:
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.web.close()
        cls.robot.__exit__(None, None, None)
        logging.disable(logging.NOTSET)

    def post(self, body, key="key", path="/api/drive"):
        req = urllib.request.Request(f"http://127.0.0.1:{self.web.port}{path}", method="POST",
                                     data=json.dumps(body).encode(),
                                     headers={"X-Arcby-Key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None

    def hold(self, throttle, steer, seconds, speed=0.5, turn=90):
        """Like the page: send the held keys every 50 ms."""
        tele = None
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tele = self.post({"throttle": throttle, "steer": steer, "speed": speed, "turn": turn})[1]
            time.sleep(0.05)
        return tele

    def settle(self):
        self.hold(0, 0, 0.8)

    def test_1_ready_after_calibration(self):
        tele = self.post({})[1]
        self.assertFalse(tele["stop"], tele["reasons"])

    def test_2_gas_drives_straight_forward(self):
        self.settle()
        x0, h0 = self.sim.x, self.sim.heading
        tele = self.hold(1, 0, 1.5)
        h = math.radians(h0)
        moved = (self.sim.x - x0) * math.cos(h) + (self.sim.y - 0) * 0     # mostly along +x at start
        self.assertGreater(math.hypot(self.sim.x - x0, self.sim.y), 20)     # it really went somewhere
        self.assertLess(abs(self.sim.heading - h0), 5)                       # and kept its heading
        self.assertTrue(tele["hold"])
        self.settle()

    def test_3_letting_go_stops(self):
        self.hold(1, 0, 1.0)
        self.hold(0, 0, 0.5)
        tele = self.post({})[1]
        self.assertEqual(tele["applied"], [0.0, 0.0])

    def test_4_curve_reaches_the_turn_rate(self):
        self.settle()
        rates = []
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            t = self.post({"throttle": 1, "steer": 1, "speed": 0.5, "turn": 90})[1]
            rates.append(t["yaw_rate"])
            time.sleep(0.05)
        tail = rates[len(rates) // 2:]
        self.assertAlmostEqual(sum(tail) / len(tail), -90, delta=20)        # D = right = negative
        self.settle()

    def test_5_turn_on_the_spot_uses_both_wheels_and_stays_put(self):
        self.settle()
        x0, y0 = self.sim.x, self.sim.y
        tele = self.hold(0, -1, 1.5)                                          # A standing
        l, r = tele["applied"]
        self.assertLess(l, 0)
        self.assertGreater(r, 0)
        self.assertLess(math.hypot(self.sim.x - x0, self.sim.y - y0), 3.0)
        self.settle()

    def test_6_browser_gone_means_stop(self):
        self.hold(1, 0, 1.0)
        time.sleep(0.6)                                                       # no messages at all
        m = self.robot.state.get("motors")
        self.assertEqual(m["applied"], (0.0, 0.0))
        self.assertEqual(m["reason"], "command timeout")

    def test_7_wrong_key_cannot_drive(self):
        self.settle()
        code, _ = self.post({"throttle": 1}, key="guess")
        self.assertEqual(code, 403)
        time.sleep(0.1)
        self.assertEqual(self.robot.state.get("motors")["target"], (0.0, 0.0))

    def test_8_reset_endpoint(self):
        code, tele = self.post({}, path="/api/reset")
        self.assertEqual(code, 200)
        self.assertEqual(tele["latched"], [])


class ProgramTest(unittest.TestCase):
    """`python3 drive_web.py --sim` as a separate program, like Selim starts it."""

    def test_start_drive_and_ctrl_c(self):
        with tempfile.TemporaryDirectory() as home:
            env = dict(os.environ, HOME=home)                                 # logs into the temp home
            p = subprocess.Popen([sys.executable, "-u", str(CODE / "drive_web.py"), "--sim", "--port", "0"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            lines = queue.Queue()
            threading.Thread(target=lambda: [lines.put(l) for l in p.stdout], daemon=True).start()
            seen = []
            try:
                url = None
                end = time.monotonic() + 10
                while time.monotonic() < end and url is None:
                    try:
                        line = lines.get(timeout=0.5)               # never blocks forever
                    except queue.Empty:
                        continue
                    seen.append(line)
                    if "http://" in line:
                        url = line.strip()
                self.assertIsNotNone(url, "no link printed: " + "".join(seen))
                port, key = url.split(":")[2].split("/")[0], url.split("k=")[1]
                self.assertEqual(len(key), 22)                  # token_urlsafe(16): 128 bits
                page = urllib.request.urlopen(f"http://127.0.0.1:{port}/?k={key}", timeout=2).read()
                self.assertIn(b"Arcby drive", page)
                req = urllib.request.Request(f"http://127.0.0.1:{port}/api/drive", method="POST",
                                             data=b'{"throttle": 1}', headers={"X-Arcby-Key": key})
                self.assertIn("applied", json.loads(urllib.request.urlopen(req, timeout=2).read()))
            finally:
                p.send_signal(signal.SIGINT)                                  # Ctrl+C
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    raise
                time.sleep(0.2)
                rest = ""
                while not lines.empty():
                    rest += lines.get()
            self.assertEqual(p.returncode, 0, rest)
            self.assertIn("motors stopped, bye", rest)
            self.assertTrue((Path(home) / "scratch" / "drive.log").exists())


if __name__ == "__main__":
    unittest.main()
