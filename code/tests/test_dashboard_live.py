"""pi/dashboard /api/live: the robot's live state (alive.py's state.json) - missing, stale, broken, good."""
import http.client
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

DASH = Path(__file__).resolve().parents[2] / "pi" / "dashboard" / "arcby_dashboard.py"
spec = importlib.util.spec_from_file_location("arcby_dashboard_live", DASH)
dash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dash)

STATE = {
    "camera": {"fps": 13.4, "ev": 0.0, "detect_ms": 50, "embed_ms": 716},
    "head": {"on": False, "pan": 0.0, "tilt": 0.0, "dps": 0.0},
    "faces": [{"track": 12, "person": "009cf37c", "name": "Selim", "known": True, "unknown": False, "picture": False,
               "looking": True, "engagement": 0.86, "sim": 0.62, "pan": -10.2, "tilt": 14.8, "size_deg": 5.1,
               "target": True}],
    "attention": {"mode": "follow", "pan": -10.0, "tilt": 15.0, "target": 12, "target_name": "Selim", "partner": "Selim"},
    "people": [{"id": "009cf37c", "name": "Selim", "call": "Selim", "voice_prints": 5}],
}


class LiveTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.old = dash.ALIVE_DIR
        dash.ALIVE_DIR = Path(self.dir.name)
        self.file = dash.ALIVE_DIR / "state.json"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        dash.ALIVE_DIR = self.old
        self.dir.cleanup()

    def get(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        conn.request("GET", "/api/live")
        r = conn.getresponse()
        body = r.read()
        conn.close()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("Content-Type"), "application/json")
        return json.loads(body)

    def test_missing_file(self):
        d = self.get()
        self.assertIsNone(d["state"])
        self.assertTrue(d["stale"])
        self.assertIsNone(d["age_s"])
        self.assertIn("no state file", d["why"])
        self.assertIn(d["running"], (True, False))

    def test_good_file(self):
        self.file.write_text(json.dumps(dict(STATE, wall=time.time() - 0.4)))
        d = self.get()
        self.assertFalse(d["stale"])
        self.assertLess(d["age_s"], 3)
        self.assertIsNone(d["why"])
        self.assertEqual(d["state"]["faces"][0]["name"], "Selim")
        self.assertEqual(d["state"]["attention"]["target_name"], "Selim")

    def test_stale_file(self):
        self.file.write_text(json.dumps(dict(STATE, wall=time.time() - 42)))
        d = self.get()
        self.assertTrue(d["stale"])
        self.assertAlmostEqual(d["age_s"], 42, delta=2)
        self.assertIn("old", d["why"])
        self.assertEqual(d["state"]["people"][0]["name"], "Selim")     # kept for the page, which hides it

    def test_broken_or_partial_file(self):
        for raw in (b'{"wall": 17935', b"", b"\xff\xfe junk", b"[1, 2]"):
            self.file.write_bytes(raw)
            d = self.get()
            self.assertIsNone(d["state"])
            self.assertTrue(d["stale"])
            self.assertIn("broken", d["why"])

    def test_no_wall_uses_the_file_time(self):
        self.file.write_text(json.dumps(STATE))
        d = dash.live_state(self.file, now=self.file.stat().st_mtime + 10)
        self.assertAlmostEqual(d["age_s"], 10, delta=0.2)
        self.assertTrue(d["stale"])

    def test_page_has_live_section_at_the_top(self):
        page = dash.PAGE
        for needle in ('id="lfaces"', 'id="lattn"', 'id="lears"', 'id="lsens"', 'id="lpeople"', "api/live"):
            self.assertIn(needle, page)
        self.assertLess(page.index('id="lfaces"'), page.index('id="health"'))
        self.assertLess(page.index('id="talk"'), page.index('id="health"'))


if __name__ == "__main__":
    unittest.main()
