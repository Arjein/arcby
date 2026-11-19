"""body_mcp.py: MCP protocol, Body actions on the simulated robot."""
import base64
import io
import json
import logging
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "hwtest"))

import body_mcp                                     # noqa: E402
from base import SimDrive, SimRobot                 # noqa: E402
from robot import config, imu                       # noqa: E402
from robot.main import Robot                        # noqa: E402
from robot.pan_tilt import PanTilt                  # noqa: E402
from tests.fakes import FakePca9685                 # noqa: E402


class WallSonar:
    """A wall WALL cm in front of the simulated start position, ~5 cm sensor offset ignored."""

    def __init__(self, sim, wall_cm):
        self.sim, self.wall = sim, wall_cm

    def ping(self):
        return round(self.wall - self.sim.x, 1)

    def measure(self, n=3):
        return self.ping()


class FakeCamera:
    def capture_image(self, stream):
        from PIL import Image
        return Image.new("RGB", (1152, 648), (40, 80, 120))


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.started = 0

        class Dummy:
            def status(_, front=True):
                return "ok"

            def move(_, cm):
                return "moved"

        def start():
            self.started += 1
            return Dummy()
        self.srv = body_mcp.Server(start)

    def rpc(self, method, params=None, mid=1):
        return self.srv.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})

    def test_initialize_and_list_do_not_start_hardware(self):
        r = self.rpc("initialize", {"protocolVersion": "2025-03-26"})["result"]
        self.assertEqual(r["protocolVersion"], "2025-03-26")
        self.assertIn("tools", r["capabilities"])
        names = [t["name"] for t in self.rpc("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["status", "move", "turn", "look", "see", "stop", "reset"])
        self.assertEqual(self.started, 0)

    def test_call_starts_hardware_once(self):
        for _ in range(2):
            r = self.rpc("tools/call", {"name": "status", "arguments": {}})["result"]
            self.assertEqual(r, {"content": [{"type": "text", "text": "ok"}], "isError": False})
        self.assertEqual(self.started, 1)

    def test_errors(self):
        self.assertIsNone(self.srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(self.rpc("nope")["error"]["code"], -32601)
        r = self.rpc("tools/call", {"name": "fly", "arguments": {}})["result"]
        self.assertTrue(r["isError"])
        r = self.rpc("tools/call", {"name": "move", "arguments": {}})["result"]
        self.assertTrue(r["isError"])
        self.assertIn("bad arguments", r["content"][0]["text"])

    def test_serve_loop_skips_garbage(self):
        inp = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\n\nnot json\n[1]\n')
        out = io.StringIO()
        self.srv.serve(inp, out)
        replies = [json.loads(x) for x in out.getvalue().splitlines()]
        self.assertEqual(replies[0], {"jsonrpc": "2.0", "id": 1, "result": {}})
        self.assertEqual(replies[1]["error"]["code"], -32700)
        self.assertEqual(replies[2]["error"]["code"], -32600)


class BodySimTest(unittest.TestCase):
    """Body on the real Robot core with the simulated robot (threads, safety, motors)."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.sim = SimRobot()
        self.robot = Robot(driver=SimDrive(self.sim, 1.0),
                           imu_dev=self.sim.imu_device(imu.parse_axes(config.IMU_AXES)),
                           imu_filter=imu.ImuFilter(calib_s=0.3),
                           power_reader=lambda: (False, False, "sim"), blackbox_dir=None)
        self.robot.__enter__()
        self.sonar = WallSonar(self.sim, 1000.0)
        head = PanTilt(FakePca9685(), sleep=lambda s: None)
        head.init()
        self.body = body_mcp.Body(self.robot, sonar=self.sonar, head=head, camera=FakeCamera())

    def tearDown(self):
        self.robot.__exit__(None, None, None)
        logging.disable(logging.NOTSET)

    def test_move_forward_holds_heading(self):
        text = self.body.move(40)
        self.assertIn("moved ~+40cm", text)
        self.assertIn("safe", text)
        self.assertGreater(self.sim.x, 15)                  # sim is slower than the carpet calibration
        self.assertLess(abs(self.sim.y), 5)
        self.assertAlmostEqual(self.body.x, 40, delta=4)
        self.assertEqual(self.robot.state.get("motors")["applied"], (0.0, 0.0))

    def test_move_back_is_clamped(self):
        self.assertIn("moved ~-30cm", self.body.move(-80))
        self.assertLess(self.sim.x, -5)

    def test_obstacle_stops_forward_move(self):
        self.sonar.wall = 35.0
        text = self.body.move(100)
        self.assertIn("obstacle", text)
        self.assertLess(self.sim.x, 30)
        self.sonar.wall = self.sim.x + 10
        with self.assertRaises(body_mcp.Blocked):
            self.body.move(20)

    def test_turn_left_and_right(self):
        text = self.body.turn(90)
        self.assertIn("asked +90", text)
        self.assertAlmostEqual(self.sim.heading, 90, delta=30)
        self.body.turn(-90)
        self.assertAlmostEqual(self.sim.heading, 0, delta=40)

    def test_latched_stop_blocks_until_reset(self):
        self.body._ready()
        self.robot.safety.latched.add("bump")
        time.sleep(0.1)
        with self.assertRaises(body_mcp.Blocked) as cm:
            self.body.move(20)
        self.assertIn("reset", str(cm.exception))
        self.assertIn("STOP bump", self.body.status())
        self.body.reset()
        self.assertIn("moved", self.body.move(10))

    def test_look_see_status(self):
        self.assertEqual(self.body.look(20, 5), "head pan +20° tilt +5°")
        self.assertIn("head +20/+5", self.body.status())
        jpg = self.body.see(320)
        self.assertEqual(jpg[:2], b"\xff\xd8")


class SubprocessTest(unittest.TestCase):
    def test_sim_server_over_stdio(self):
        p = subprocess.Popen([sys.executable, str(CODE / "body_mcp.py"), "--sim"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "status", "arguments": {}}},
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "see", "arguments": {}}}]
            out = []
            reader = threading.Thread(target=lambda: out.extend(p.stdout.readline() for _ in range(3)))
            reader.start()
            p.stdin.write("".join(json.dumps(m) + "\n" for m in msgs))
            p.stdin.flush()
            reader.join(timeout=20)
            replies = [json.loads(x) for x in out]
            self.assertEqual([r["id"] for r in replies], [1, 2, 3])
            self.assertIn("pos x+0 y+0cm", replies[1]["result"]["content"][0]["text"])
            self.assertTrue(replies[2]["result"]["isError"])          # no camera in --sim
            p.stdin.close()
            self.assertEqual(p.wait(timeout=10), 0)
        finally:
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    unittest.main()
