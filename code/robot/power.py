"""
Power module: watches the Pi's supply, which is the battery through the Pi buck.

There is no battery-voltage sensor (the Pi has no ADC). But the Pi buck needs
about 7.0 V in to give 5.1 V out, so when the 2S LiPo runs low the Pi itself
sees under-voltage - that is what showed first when the pack was over-
discharged to 5.70 V (2026-10-28). Polled every POWER_POLL_S from the kernel's
rpi_volt alarm (/sys/class/hwmon/*/in0_lcrit_alarm), falling back to
`vcgencmd get_throttled`.

battery_on() / require_battery(): is the battery on at all (read-only probe of the PTZ board, which the servo
buck powers) - no battery = no head, no motors.

alive_running() / require_no_alive(): alive.py (started at boot) owns the IMU, the head and the sonar - a drive
program next to it would reset the IMU under it and fight over the head (review 2026-11-01).

Publishes "power" in State: {"ok": True, "under_voltage": bool, "since_boot":
bool or None, "source": str} or {"ok": False, "error": str} (no source, e.g.
not on a Pi - then nothing is stopped). safety.py turns a sustained
under-voltage into a latched "low battery" stop.
"""
import subprocess
import time
from pathlib import Path

from . import config


class AliveRunning(RuntimeError):
    """alive.py runs: it owns the IMU, the head and the sonar."""


def alive_running(run=subprocess.run):
    """The pids of a running code/alive.py (empty: none, or pgrep not there)."""
    try:
        # only the python process itself: a shell / editor / ssh command line naming the file is not alive.py
        r = run(["pgrep", "-f", r"python[0-9.]* .*code/alive\.py"], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in r.stdout.split() if x.isdigit()]


def require_no_alive(run=subprocess.run):
    pids = alive_running(run)
    if pids:
        raise AliveRunning(f"alive.py is running (pid {', '.join(map(str, pids))}) and owns the IMU, head and sonar - "
                           "stop it first: pkill -f code/alive.py   (run_alive.sh then stays stopped)")


def _find_alarm():
    for d in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (d / "name").read_text().strip() == "rpi_volt" and (d / "in0_lcrit_alarm").exists():
                return d / "in0_lcrit_alarm"
        except OSError:
            continue
    return None


def _vcgencmd():
    out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
    return int(out.strip().split("=")[1], 16)


class PiPowerReader:
    """() -> (under_voltage_now, since_boot or None, source)."""

    def __init__(self, since_boot_every_s=5.0, clock=time.monotonic):
        self.alarm = _find_alarm()
        self.clock = clock
        self.every = since_boot_every_s
        self._since = (None, -1e9)          # (value, time read)

    def __call__(self):
        if self.alarm is not None:
            now = self.alarm.read_text().strip() == "1"
            since, t = self._since
            if self.clock() - t > self.every:            # the "since boot" bit only from vcgencmd
                try:
                    since = bool(_vcgencmd() & 0x10000)
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    since = None
                self._since = (since, self.clock())
            return now, since, "rpi_volt"
        v = _vcgencmd()
        return bool(v & 0x1), bool(v & 0x10000), "vcgencmd"


class PowerMonitor:
    def __init__(self, state, reader=None, clock=time.monotonic):
        self.state = state
        self.reader = reader
        self.clock = clock
        self.period = config.POWER_POLL_S

    def step(self):
        try:
            if self.reader is None:
                self.reader = PiPowerReader()
            now, since, source = self.reader()
            info = {"ok": True, "under_voltage": bool(now), "since_boot": since, "source": source}
        except Exception as e:  # noqa: BLE001 - no power info must not crash the robot
            info = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.state.publish("power", info)
        return info

    def run(self, stop):
        while not stop.is_set():
            self.step()
            stop.wait(self.period)


class BatteryOff(RuntimeError):
    """The battery is switched off / unplugged: no head, no motors (the Pi may be on a wall adapter)."""


def battery_on(bus=config.I2C_BUS, addr=config.PTZ_ADDR):
    """Is the battery on? There is no battery sensor, but the PTZ board (PCA9685) is powered from the servo buck,
    which runs from the battery: no answer at 0x40 = battery off (or the board unplugged). Read-only - it never
    writes to the board (driving the head on wall power once reset the Pi)."""
    from .i2c import I2CDevice
    from .pan_tilt import MODE1
    try:
        dev = I2CDevice(bus, addr)
    except OSError:
        return False                    # no I2C bus at all (not a Pi)
    try:
        dev.read(MODE1, 1)
        return True
    except OSError:
        return False
    finally:
        dev.close()


def require_battery(what="the motors", check=battery_on):
    """Raise BatteryOff unless the battery is on - for everything that needs its power."""
    if not check():
        raise BatteryOff(f"battery off (the PTZ board at 0x{config.PTZ_ADDR:02x} does not answer): not starting "
                         f"{what} - switch the battery on first (the Pi alone can run on a wall adapter)")
