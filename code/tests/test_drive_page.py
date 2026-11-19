"""The browser page's JavaScript (drive_web.PAGE), run in node with a pretend browser.
These are the page-side safety rules: what is sent for which keys, stop on blur / hidden tab /
Space, never two requests at once, recovery after a hung request."""
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import drive_web

HARNESS = Path(__file__).with_name("page_harness.js")


@unittest.skipUnless(shutil.which("node"), "node.js not installed")
class DrivePageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = re.search(r"<script>(.*)</script>", drive_web.PAGE, re.S).group(1)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(script)
        out = subprocess.run(["node", str(HARNESS), f.name], capture_output=True, text=True, timeout=30)
        Path(f.name).unlink()
        cls.r = json.loads(out.stdout.strip().splitlines()[-1])
        if "error" in cls.r:
            raise AssertionError(cls.r["error"])

    def body(self, name):
        b = self.r[name]
        return (b["throttle"], b["steer"])

    def test_defaults(self):
        self.assertEqual(self.r["defaults"], {"throttle": 0, "steer": 0, "speed": 0.4, "turn": 90})

    def test_sends_the_secret_key(self):
        self.assertEqual(self.r["w_key"], "sekret")

    def test_keys_held_together_like_a_car(self):
        self.assertEqual(self.body("w"), (1, 0))
        self.assertEqual(self.body("w_a"), (1, -1))          # W + A at the same time
        self.assertEqual(self.body("a_only"), (0, -1))       # W released, A still held
        self.assertEqual(self.body("none"), (0, 0))          # all released -> stop

    def test_arrow_keys_work_too(self):
        self.assertEqual(self.body("arrows"), (1, 1))

    def test_switching_window_stops_at_once(self):
        self.assertEqual(self.body("after_blur_immediate"), (0, 0))
        self.assertEqual(self.body("after_blur"), (0, 0))

    def test_hidden_tab_stops(self):
        self.assertEqual(self.body("after_hidden"), (0, 0))

    def test_space_stops(self):
        self.assertEqual(self.body("after_space"), (0, 0))

    def test_touch_buttons(self):
        self.assertEqual(self.body("touch_hold"), (1, 0))
        self.assertEqual(self.body("touch_leave"), (0, 0))   # finger slides off -> stop

    def test_sliders_are_sent_and_remembered(self):
        self.assertEqual(self.r["sliders"]["speed"], 0.7)
        self.assertEqual(self.r["sliders"]["turn"], 200)
        self.assertEqual(self.r["stored"], {"arcby-speed": "0.7", "arcby-turn": "200"})

    def test_reset_key(self):
        self.assertEqual(self.r["reset_path"], "api/reset")

    def test_save_key(self):
        self.assertEqual(self.r["snap_path"], "api/snapshot")

    def test_banner_shows_the_robot_state(self):
        self.assertEqual(self.r["banner_ok"], "banner ok")
        self.assertEqual(self.r["yaw_text"], "12°")
        self.assertEqual(self.r["banner_stop"], "banner bad")
        self.assertIn("bump", self.r["banner_text"])
        self.assertFalse(self.r["reset_hidden"])             # latched -> Reset button shown
        self.assertEqual(self.r["banner_calib"], "banner wait")

    def test_low_battery_is_shown(self):
        self.assertIn("LOW BATTERY", self.r["banner_battery"])
        self.assertEqual(self.r["power_text"], "UNDER-VOLTAGE")
        self.assertIn("dipped earlier", self.r["power_dipped"])

    def test_lost_link_is_shown(self):
        self.assertIn("NO LINK", self.r["link_fail"])

    def test_never_two_requests_at_once_and_recovers_after_a_hang(self):
        self.assertEqual(self.r["hang_requests"], 1)          # second tick skipped while busy
        self.assertEqual(self.r["recovered"], 2)              # after the 500 ms abort it sends again


if __name__ == "__main__":
    unittest.main()
