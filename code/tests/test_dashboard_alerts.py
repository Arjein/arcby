"""pi/dashboard: the alarm bar on every tab - power (under-voltage), heat, the SD card, the USB stick, the robot."""
import importlib.util
import unittest
from pathlib import Path

DASH = Path(__file__).resolve().parents[2] / "pi" / "dashboard" / "arcby_dashboard.py"
spec = importlib.util.spec_from_file_location("arcby_dashboard_alerts", DASH)
dash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dash)
GB = dash.GB


def health(uv_now=False, uv_past=False, temp=50.0, free=10 * GB, throttled_now=False):
    return {"temp_c": temp, "disk": {"free": free},
            "throttled": {"flags": [{"flag": "under-voltage", "now": uv_now, "since_boot": uv_past},
                                    {"flag": "throttled", "now": throttled_now, "since_boot": False}]}}


def levels(out):
    return [(a["level"], a["text"].split(" ")[0]) for a in out]


class AlertsTest(unittest.TestCase):
    def test_all_well_on_battery_says_nothing(self):
        self.assertEqual(dash.alerts(health(), True, 100 * GB, True, True), [])

    def test_under_voltage_now_is_the_big_red_one_first(self):
        out = dash.alerts(health(uv_now=True, uv_past=True, temp=81), True, 100 * GB, True, True)
        self.assertEqual(out[0]["level"], "danger")
        self.assertIn("UNDER-VOLTAGE NOW", out[0]["text"])
        self.assertIn("HOT", out[1]["text"])
        self.assertEqual(sum(1 for a in out if "under-voltage" in a["text"].lower()), 1)   # not twice

    def test_under_voltage_earlier_is_a_warning(self):
        (a,) = dash.alerts(health(uv_past=True), True, 100 * GB, True, True)
        self.assertEqual(a["level"], "warn")

    def test_the_stick_missing_means_nothing_is_recorded(self):
        (a,) = dash.alerts(health(), False, None, True, True)
        self.assertEqual(a["level"], "danger")
        self.assertIn("USB stick", a["text"])
        self.assertEqual(dash.alerts(health(), True, 2 * GB, True, True)[0]["level"], "warn")

    def test_disk_heat_robot_and_wall_power(self):
        self.assertEqual(dash.alerts(health(free=0.5 * GB), True, 100 * GB, True, True)[0]["level"], "danger")
        self.assertEqual(dash.alerts(health(temp=76), True, 100 * GB, True, True)[0]["level"], "warn")
        self.assertIn("alive.py", dash.alerts(health(), True, 100 * GB, False, None)[0]["text"])
        (a,) = dash.alerts(health(), True, 100 * GB, True, False)
        self.assertEqual(a["level"], "info")                 # wall power: told, not alarmed

    def test_broken_health_never_raises(self):
        self.assertEqual(dash.alerts({"error": "vcgencmd missing"}, True, 100 * GB, True, True), [])

    def test_the_bar_is_on_every_tab(self):
        page = dash.PAGE
        self.assertLess(page.index('id="alerts"'), page.index('data-tab="live"'))   # outside the tabs
        self.assertIn("THE PI DOES NOT ANSWER", page)


if __name__ == "__main__":
    unittest.main()
