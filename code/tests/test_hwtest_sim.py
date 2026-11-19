"""The hardware-test scripts end to end in --sim mode (slow: ~1 min).
Run with:  ARCBY_SLOW_TESTS=1 python3 -m unittest tests.test_hwtest_sim   (from code/)"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HW = Path(__file__).resolve().parents[1] / "hwtest"


@unittest.skipUnless(os.environ.get("ARCBY_SLOW_TESTS"), "slow - set ARCBY_SLOW_TESTS=1")
class HwtestSimTest(unittest.TestCase):
    def run_script(self, *args):
        with tempfile.TemporaryDirectory() as home:
            env = dict(os.environ, HOME=home, GPIOZERO_PIN_FACTORY="mock")
            r = subprocess.run([sys.executable, *args, "--sim", "--yes"], cwd=HW, env=env,
                               capture_output=True, text=True, timeout=240)
            logs = [sorted(f.name for f in d.iterdir())          # read while the temp home exists
                    for d in (Path(home) / "scratch" / "hwtest").glob("*-sim")]
            return r, logs

    def test_base_all_floor_tests(self):
        r, logs = self.run_script("base.py")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("ALL PASSED", r.stdout)
        self.assertEqual(logs, [["events.jsonl", "imu.csv", "report.txt", "summary.json"]])

    def test_base_spinramp(self):
        r, _ = self.run_script("base.py", "--only", "spinramp")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("suggestion: SPIN_START", r.stdout)

    def test_route_rectangle(self):
        r, _ = self.run_script("route.py")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("Back at the start", r.stdout)


    def test_bringup_all_modules(self):
        r, logs = self.run_script("bringup.py")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("ALL PASSED", r.stdout)
        self.assertEqual(logs, [["beeps.wav", "photo.jpg", "quiet.wav", "voice-playback.wav", "voice.wav"]])

if __name__ == "__main__":
    unittest.main()
