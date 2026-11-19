"""drive_web.py: the HTTP drive server, against a fake robot (no hardware, real HTTP)."""
import json
import unittest
import urllib.error
import urllib.request

import drive_web
from robot.state import State
from tests.fakes import imu_reading


class FakeMotors:
    def __init__(self):
        self.requests = []

    def request(self, left, right, source="?"):
        self.requests.append(("wheels", round(left, 3), round(right, 3), source))

    def request_velocity(self, v, turn, source="?"):
        self.requests.append(("velocity", round(v, 3), round(turn, 3), source))


class FakeSafety:
    def __init__(self):
        self.resets = 0
        self.snapshots = []

    def reset(self):
        self.resets += 1

    def snapshot(self, label="manual"):
        self.snapshots.append(label)
        return "/home/x/scratch/blackbox/20261029-200000-manual.csv"


class FakeRobot:
    def __init__(self):
        self.state = State()
        self.motors = FakeMotors()
        self.safety = FakeSafety()
        self.fault = None


class DriveServerTest(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.web = drive_web.DriveServer(self.robot, "secret", host="127.0.0.1", port=0)
        self.web.serve_in_thread()
        self.addCleanup(self.web.close)
        self.base = f"http://127.0.0.1:{self.web.port}"

    def call(self, path, body=None, key="secret"):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method="GET" if data is None else "POST",
                                     headers={"X-Arcby-Key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_page_needs_the_key(self):
        self.assertEqual(self.call("/?k=wrong", key="")[0], 403)
        code, body = self.call("/?k=secret", key="")
        self.assertEqual(code, 200)
        self.assertIn(b"Arcby drive", body)

    def test_drive_needs_the_key(self):
        code, _ = self.call("/api/drive", {"throttle": 1}, key="nope")
        self.assertEqual(code, 403)
        self.assertEqual(self.robot.motors.requests, [])

    def test_gas_drives_forward(self):
        code, body = self.call("/api/drive", {"throttle": 1, "steer": 0, "speed": 0.5, "turn": 90})
        self.assertEqual(code, 200)
        self.assertEqual(self.robot.motors.requests[-1], ("velocity", 0.5, 0.0, "web"))
        tele = json.loads(body)
        self.assertIn("applied", tele)
        self.assertTrue(tele["stop"])                    # no safety verdict yet -> shown as stopped

    def test_no_keys_means_stop(self):
        self.call("/api/drive", {"throttle": 0, "steer": 0})
        self.assertEqual(self.robot.motors.requests[-1], ("velocity", 0.0, 0.0, "web"))

    def test_steering_asks_for_a_turn_rate(self):
        self.call("/api/drive", {"throttle": 0, "steer": 1, "speed": 0.5, "turn": 90})
        self.assertEqual(self.robot.motors.requests[-1], ("velocity", 0.0, -90.0, "web"))   # turn right
        self.call("/api/drive", {"throttle": 1, "steer": -1, "speed": 0.5, "turn": 60})
        self.assertEqual(self.robot.motors.requests[-1], ("velocity", 0.5, 60.0, "web"))    # curve left

    def test_limits_are_enforced_on_the_robot_side(self):
        self.call("/api/drive", {"throttle": 9, "steer": 0, "speed": 9, "turn": 90})
        self.assertEqual(self.robot.motors.requests[-1], ("velocity", 1.0, 0.0, "web"))
        self.call("/api/drive", {"throttle": 0, "steer": 1, "speed": 0.5, "turn": 9999})
        self.assertEqual(self.robot.motors.requests[-1], ("velocity", 0.0, -300.0, "web"))  # TURN_MAX_DPS

    def test_bad_messages_stop_the_robot(self):
        self.call("/api/drive", {"throttle": 1, "speed": 0.5})
        for bad in (b"not json", b"[1,2]", json.dumps({"throttle": "fast"}).encode(),
                    json.dumps({"throttle": None}).encode()):
            code, _ = self.call("/api/drive", bad)
            self.assertEqual(code, 400, bad)
            self.assertEqual(self.robot.motors.requests[-1], ("wheels", 0.0, 0.0, "web"), bad)

    def test_reset(self):
        code, _ = self.call("/api/reset", {})
        self.assertEqual(code, 200)
        self.assertEqual(self.robot.safety.resets, 1)

    def test_snapshot(self):
        code, body = self.call("/api/snapshot", {})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["saved"], "/home/x/scratch/blackbox/20261029-200000-manual.csv")
        self.assertEqual(self.robot.safety.snapshots, ["manual"])
        self.assertEqual(self.call("/api/snapshot", {}, key="bad")[0], 403)

    def test_telemetry_reports_safety_and_imu(self):
        self.robot.state.publish("safety", {"stop": True, "reasons": ["bump"], "latched": ["bump"]})
        self.robot.state.publish("imu", imu_reading(self.robot.state._clock(), yaw=42.0))
        self.robot.state.publish("motors", {"applied": (0.1, 0.2), "hold_yaw": 3.0, "reason": None})
        tele = json.loads(self.call("/api/drive", {})[1])
        self.assertEqual(tele["reasons"], ["bump"])
        self.assertEqual(tele["latched"], ["bump"])
        self.assertEqual(tele["yaw"], 42.0)
        self.assertEqual(tele["applied"], [0.1, 0.2])
        self.assertTrue(tele["hold"])
        self.assertEqual(tele["power"], {"ok": False, "error": "not read yet"})
        self.robot.state.publish("power", {"ok": True, "under_voltage": True, "since_boot": True, "source": "t"})
        self.assertTrue(json.loads(self.call("/api/drive", {})[1])["power"]["under_voltage"])

    def test_unknown_path(self):
        self.assertEqual(self.call("/api/nope", {})[0], 404)
        self.assertEqual(self.call("/nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
