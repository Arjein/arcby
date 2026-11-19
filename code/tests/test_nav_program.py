"""code/nav.py: the dead man (no page = no driving), modes, the status the page shows, people geometry, the HTTP
key check - on the simulated room, no hardware."""
import json
import math
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

import nav
from robot import config
from robot.floor import ground_points
from robot.pan_tilt import PanTiltError
from robot.nav import Command
from sim.navworld import World

W, H = config.CAMERA_LORES


class WorldBody:
    """A body for Pilot: the simulated room, advanced by the test (no threads)."""

    def __init__(self):
        self.world = World(300.0, 200.0, start=(60.0, 100.0, 0.0))
        self.jpeg = None
        self.fault = None
        self.alive = True
        self.cmds = []

    def percept(self):
        return self.world.percept()

    def apply(self, cmd, drive):
        self.cmds.append((cmd, drive))
        self.world.apply(cmd if drive else Command(head=cmd.head))
        self.world.advance(1.0 / config.NAV_RATE_HZ)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Pilot(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self._mono = nav.time.monotonic
        nav.time.monotonic = self.clock
        self.body = WorldBody()
        self.pilot = nav.Pilot(self.body)

    def tearDown(self):
        nav.time.monotonic = self._mono

    def steps(self, n, beat=True):
        for _ in range(n):
            if beat:
                self.pilot.beat()
            self.pilot.step()
            self.clock.t += 1.0 / config.NAV_RATE_HZ

    def test_stopped_never_drives(self):
        self.steps(100)
        self.assertTrue(all(not d for _, d in self.body.cmds))
        self.assertLess(self.body.world.driven, 0.5)

    def test_explore_drives_while_the_page_talks(self):
        self.pilot.set_mode("explore")
        self.steps(400)
        self.assertGreater(self.body.world.driven, 20.0)
        self.assertEqual(self.body.world.collisions, 0)

    def test_no_page_no_driving(self):
        self.pilot.set_mode("explore")
        self.steps(200)
        n = len(self.body.cmds)
        cut = int(nav.DEAD_MAN_S * config.NAV_RATE_HZ) + 2         # the link counts as alive for DEAD_MAN_S
        self.steps(cut + 10, beat=False)                           # + 0.5 s to coast to a stop
        x = self.body.world.driven
        self.steps(100, beat=False)
        self.assertTrue(all(not d for _, d in self.body.cmds[n + cut:]))
        self.assertLess(self.body.world.driven - x, 0.2)
        self.assertTrue(self.pilot.status()["link_paused"])

    def test_stop_stops_and_the_status_is_json(self):
        self.pilot.set_mode("explore")
        self.steps(300)
        self.pilot.set_mode("stopped")
        self.steps(20)                          # the wheels' own slow-down (1 s)
        x = self.body.world.driven
        self.steps(40)
        self.assertLess(self.body.world.driven - x, 0.5)
        s = json.loads(json.dumps(self.pilot.status()))
        self.assertEqual(s["mode"], "stopped")
        self.assertIn("stats", s)

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.pilot.set_mode("race")


class Geometry(unittest.TestCase):
    class B:
        def __init__(self, box, score=0.8):
            self.box, self.score = np.array(box, float), score

    def test_person_feet_give_the_distance(self):
        y2 = H * 0.7
        gx, gy, _ = ground_points(W / 2, y2, W, H, 0.0, config.NAV_TILT_DEG)
        (p,) = nav.people_in([self.B([W / 2 - 30, 10, W / 2 + 30, y2])], W, H, 0.0, config.NAV_TILT_DEG)
        self.assertAlmostEqual(p.bearing, 0.0, places=3)
        self.assertAlmostEqual(p.dist, math.hypot(gx, gy), places=3)

    def test_cut_feet_no_distance_and_left_is_plus(self):
        (p,) = nav.people_in([self.B([20, 10, 120, H])], W, H, 10.0, config.NAV_TILT_DEG)
        self.assertIsNone(p.dist)
        self.assertGreater(p.bearing, 10.0)

    def test_unsure_bodies_are_dropped(self):
        self.assertEqual(nav.people_in([self.B([0, 0, 50, 50], 0.2)], W, H, 0.0, -28.0), [])

    def test_overlay_is_a_jpeg(self):
        w = World(300.0, 200.0, start=(60.0, 100.0, 0.0))
        w.head = [0.0, config.NAV_TILT_DEG]
        v = w._picture()
        jpg = nav.overlay_jpeg(np.zeros((H, W, 3), np.uint8), v, W, H)
        self.assertEqual(jpg[:2], b"\xff\xd8")


class SpeedFromTheSonar(unittest.TestCase):
    def test_slope_after_the_soft_start(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(nav.__file__).parent / "hwtest"))
        import navcal
        samples = [(t, 150.0 - (8.0 * t if t < 0.4 else 3.2 + 30.0 * (t - 0.4))) for t in np.arange(0, 1.5, 0.08)]
        self.assertAlmostEqual(navcal.slope(samples), 30.0, places=6)
        self.assertIsNone(navcal.slope(samples[:4]))


class Http(unittest.TestCase):
    def setUp(self):
        self.pilot = nav.Pilot(WorldBody())
        self.web = nav.Web(self.pilot, "secret", 0)
        self.base = f"http://127.0.0.1:{self.web.port}"

    def tearDown(self):
        self.web.close()

    def post(self, path, body, key="secret"):
        req = urllib.request.Request(f"{self.base}{path}?k={key}", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read())

    def test_wrong_key_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post("/api/mode", {"mode": "explore"}, key="nope")
        self.assertEqual(e.exception.code, 403)
        e.exception.close()
        self.assertEqual(self.pilot.mode, "stopped")

    def test_mode_and_heartbeat(self):
        before = time.monotonic()
        s = self.post("/api/mode", {"mode": "explore"})
        self.assertEqual(s["mode"], "explore")
        self.assertGreaterEqual(self.pilot.heartbeat, before)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post("/api/mode", {"mode": "race"})
        e.exception.close()
        self.assertEqual(self.pilot.mode, "stopped")            # a bad message stops it

    def test_page(self):
        with urllib.request.urlopen(f"{self.base}/?k=secret", timeout=2) as r:
            self.assertIn(b"Arcby drives", r.read())


if __name__ == "__main__":
    unittest.main()


class FlakyPanTilt:
    """A PTZ board whose writes fail when told to (EIO), like 0x40 on 2026-11-04."""

    def __init__(self):
        self.fail_writes = 0
        self.down = False
        self.inits = 0
        self.at = None

    def init(self):
        if self.down:                       # like robot/pan_tilt.py: init wraps the I2C error
            raise PanTiltError("PTZ board does not answer: [Errno 121] Remote I/O error")
        self.inits += 1

    def set(self, pan, tilt):
        if self.down or self.fail_writes:
            self.fail_writes = max(0, self.fail_writes - 1)
            raise OSError(5, "Input/output error")
        self.at = (pan, tilt)
        return (pan, tilt)


class HeadErrors(unittest.TestCase):
    def setUp(self):
        self.t = [0.0]
        self.pt = FlakyPanTilt()
        self.head = nav.Head(self.pt, clock=lambda: self.t[0], sleep=lambda s: None)

    def ticks(self, n):
        for _ in range(n):
            self.t[0] += 0.02
            self.head.tick()

    def test_one_failed_write_is_retried_after_a_reinit(self):
        self.head.target = (30.0, config.NAV_TILT_DEG)
        self.pt.fail_writes = 1
        self.ticks(30)
        self.assertEqual(self.head.errors, 1)
        self.assertEqual(self.pt.inits, 2)                  # the start + once after the error
        self.assertAlmostEqual(self.head.now[0], 30.0)

    def test_a_board_silent_too_long_stops_the_head(self):
        self.pt.down = True
        self.head.target = (30.0, config.NAV_TILT_DEG)
        with self.assertRaises((OSError, PanTiltError)):
            self.ticks(int(config.NAV_HEAD_DOWN_S / 0.02) + 5)

    def test_a_board_silent_for_a_moment_comes_back_even_when_the_reinit_fails(self):
        # 2026-11-04 night: EIO on a write, then the re-init got Errno 121 (PanTiltError) and killed the drive
        self.head.target = (30.0, config.NAV_TILT_DEG)
        self.pt.down = True
        self.ticks(15)                                      # 0.3 s silent: writes and re-inits fail
        self.pt.down = False
        self.ticks(40)
        self.assertGreater(self.head.errors, 5)
        self.assertAlmostEqual(self.head.now[0], 30.0)

    def test_a_silent_start_is_tried_three_times(self):
        pt = FlakyPanTilt()
        pt.fail_writes = 2
        head = nav.Head(pt, clock=lambda: 0.0, sleep=lambda s: None)
        self.assertEqual(head.errors, 2)
        self.assertEqual(head.now, (0.0, config.NAV_TILT_DEG))
