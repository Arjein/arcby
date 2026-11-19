"""robot/power.py: reading the Pi's under-voltage alarm (sysfs + vcgencmd faked)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robot import power
from robot.power import PiPowerReader, PowerMonitor
from robot.state import State
from tests.fakes import FakeClock


class PiPowerReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.alarm = Path(self.tmp.name) / "in0_lcrit_alarm"
        self.alarm.write_text("0\n")
        self.clock = FakeClock()

    def reader(self, throttled=0x0):
        self.calls = 0

        def vc():
            self.calls += 1
            return throttled
        mock.patch.object(power, "_find_alarm", return_value=self.alarm).start()
        mock.patch.object(power, "_vcgencmd", side_effect=vc).start()
        self.addCleanup(mock.patch.stopall)
        return PiPowerReader(since_boot_every_s=5.0, clock=self.clock)

    def test_reads_the_kernel_alarm(self):
        r = self.reader()
        self.assertEqual(r(), (False, False, "rpi_volt"))
        self.alarm.write_text("1\n")
        self.assertEqual(r()[0], True)

    def test_since_boot_comes_from_vcgencmd_but_not_every_poll(self):
        r = self.reader(throttled=0x50000)               # bit 16: under-voltage happened
        self.assertEqual(r()[1], True)
        r(); r()
        self.assertEqual(self.calls, 1)                    # cached
        self.clock.sleep(6)
        r()
        self.assertEqual(self.calls, 2)

    def test_falls_back_to_vcgencmd_without_the_alarm_file(self):
        mock.patch.object(power, "_find_alarm", return_value=None).start()
        mock.patch.object(power, "_vcgencmd", return_value=0x50001).start()
        self.addCleanup(mock.patch.stopall)
        self.assertEqual(PiPowerReader()(), (True, True, "vcgencmd"))

    def test_vcgencmd_failure_only_loses_the_since_boot_bit(self):
        mock.patch.object(power, "_find_alarm", return_value=self.alarm).start()
        mock.patch.object(power, "_vcgencmd", side_effect=OSError("no vcgencmd")).start()
        self.addCleanup(mock.patch.stopall)
        self.assertEqual(PiPowerReader()(), (False, None, "rpi_volt"))

    def test_real_pi_has_a_source(self):
        # on the Pi itself one of the two sources must work (skipped elsewhere)
        if power._find_alarm() is None and not Path("/usr/bin/vcgencmd").exists():
            self.skipTest("not a Raspberry Pi")
        now, since, source = PiPowerReader()()
        self.assertIn(source, ("rpi_volt", "vcgencmd"))
        self.assertIsInstance(now, bool)


class PowerMonitorTest(unittest.TestCase):
    def test_publishes_what_the_reader_says(self):
        st = State()
        PowerMonitor(st, reader=lambda: (True, False, "test")).step()
        self.assertEqual(st.get("power"), {"ok": True, "under_voltage": True, "since_boot": False, "source": "test"})

    def test_reader_errors_are_reported_not_raised(self):
        st = State()
        def broken():
            raise OSError("gone")
        info = PowerMonitor(st, reader=broken).step()
        self.assertFalse(info["ok"])
        self.assertIn("gone", info["error"])

    def test_run_loop_stops(self):
        import threading
        st, stop = State(), threading.Event()
        m = PowerMonitor(st, reader=lambda: (False, False, "t"))
        th = threading.Thread(target=m.run, args=(stop,))
        th.start()
        stop.set()
        th.join(timeout=2)
        self.assertFalse(th.is_alive())
        self.assertTrue(st.get("power")["ok"])


class BatteryTest(unittest.TestCase):
    """No battery (the PTZ board on the servo buck does not answer) = no head, no motors."""

    def test_no_i2c_bus_means_no_battery(self):
        self.assertFalse(power.battery_on(bus="/nonexistent/i2c-9"))

    def test_require_battery_refuses_with_a_clear_reason(self):
        with self.assertRaises(power.BatteryOff) as e:
            power.require_battery("the motors", check=lambda: False)
        self.assertIn("battery off", str(e.exception))
        power.require_battery("the motors", check=lambda: True)        # on: nothing happens

    def test_the_real_motor_driver_refuses_before_touching_a_pin(self):
        from robot.motors import GpioDriver
        with self.assertRaises(power.BatteryOff):
            GpioDriver(battery_check=lambda: False)                    # raised before gpiozero is even imported


class AliveRunningTest(unittest.TestCase):                                   # review 2026-11-01
    def test_a_drive_program_refuses_while_alive_py_runs(self):
        import subprocess
        from types import SimpleNamespace
        from robot.power import AliveRunning, alive_running, require_no_alive
        running = lambda *a, **k: SimpleNamespace(stdout="1234\n")        # noqa: E731
        none = lambda *a, **k: SimpleNamespace(stdout="")                   # noqa: E731

        def missing(*a, **k):
            raise FileNotFoundError("pgrep")
        self.assertEqual(alive_running(running), [1234])
        with self.assertRaises(AliveRunning):
            require_no_alive(running)
        require_no_alive(none)
        self.assertEqual(alive_running(missing), [])
        self.assertTrue(issubclass(AliveRunning, RuntimeError) and subprocess)


if __name__ == "__main__":
    unittest.main()
