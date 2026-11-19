"""
Sensor logging: the IMU, the distance sensor and the Pi's own health go into the recorder (robot/recorder.py),
each in its own thread, also when the robot does not drive (carried, bumped, someone walking up to it).

    logger = SensorLogger(rec)       # rec = Recorder
    logger.start(stop_event)         # threads end when stop_event is set
    logger.status                    # {"imu": "ok" / "off: ...", "sonar": ..., "system": ...}

Streams: imu (accel g, gyro deg/s in the robot frame, roll / pitch / yaw / tilt, bump, temperature: DATA_IMU_HZ
while the robot moves, DATA_IMU_STILL_HZ while it stands still), sonar (cm or why not: when it changed by
DATA_SONAR_CHANGE_CM, else every DATA_SONAR_IDLE_S), system (every DATA_SYSTEM_S: CPU temperature, load,
memory, disk, under-voltage, fan). A robot standing in a room for hours makes little data, a moving one a lot. A sensor that does not answer is logged once and left alone - never stops the robot.
Uses only what the Pi powers itself, so it also runs on a wall adapter (no head, no motors).
"""
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


def cpu_temp_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def mem_available_mb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def fan_on():
    """The gpio-fan overlay's cooling device state (None if there is none)."""
    for d in Path("/sys/class/thermal").glob("cooling_device*"):
        try:
            if "fan" in (d / "type").read_text():
                return int((d / "cur_state").read_text()) > 0
        except (OSError, ValueError):
            continue
    return None


def system_sample(root, power=None):
    s = dict(cpu_c=cpu_temp_c(), load=os.getloadavg()[0], mem_free_mb=mem_available_mb(), fan=fan_on())
    try:
        du = shutil.disk_usage(Path(root).expanduser())
        s["disk_free_gb"] = round(du.free / 1e9, 2)
    except OSError:
        pass
    if power is not None:
        try:
            now, since, _ = power()
            s.update(undervolt_now=now, undervolt_since_boot=since)
        except Exception:  # noqa: BLE001 - no vcgencmd (Mac)
            pass
    return s


class SensorLogger:
    def __init__(self, rec, imu=None, sonar=None, power=None):
        """imu / sonar / power: ready objects (tests) or None = open the real ones in start()."""
        self.rec = rec
        self._imu, self._sonar, self._power = imu, sonar, power
        self.status = {}
        self.latest = {}                # stream -> its newest record (the dashboard's live view, via alive.py)
        self.threads = []

    def _put(self, stream, record):
        self.latest[stream] = record
        self.rec.write(stream, record)

    def start(self, stop):
        for name, fn in (("imu", self._run_imu), ("sonar", self._run_sonar), ("system", self._run_system)):
            th = threading.Thread(target=self._guard, args=(name, fn, stop), name=f"log-{name}", daemon=True)
            th.start()
            self.threads.append(th)
        return self

    def join(self, timeout=2.0):
        for th in self.threads:
            th.join(timeout)

    def _guard(self, name, fn, stop):
        try:
            fn(stop)
        except Exception as e:  # noqa: BLE001 - a missing sensor must not stop the robot
            self.status[name] = f"off: {e}"
            self._put("system", {"sensor": name, "off": str(e)})
            log.warning("sensors: %s off (%s)", name, e)

    # ---------- the IMU ----------

    def _run_imu(self, stop):
        imu = self._imu
        if imu is None:
            from .imu import Imu
            from .state import State
            imu = Imu(State())
            imu.start()
        self.status["imu"] = "ok"
        fast_dt, still_dt = 1.0 / config.DATA_IMU_HZ, 1.0 / config.DATA_IMU_STILL_HZ
        period = 1.0 / config.IMU_RATE_HZ
        next_t = time.monotonic()
        last_write, last_move, last_accel = -1e9, -1e9, None
        while not stop.is_set():
            r = imu.step()                          # the filter needs every sample; only some are written
            if r is not None:
                moved = (any(abs(g) > config.DATA_IMU_MOVE_DPS for g in r.gyro_dps) or r.bump_g > config.IMU_BUMP_G / 2
                         or (last_accel is not None
                             and max(abs(a - b) for a, b in zip(r.accel_g, last_accel)) > config.DATA_IMU_MOVE_G))
                if moved:
                    last_move = r.t
                fast = r.t - last_move < config.DATA_IMU_MOVE_HOLD_S
            if r is not None and r.t - last_write >= (fast_dt if fast else still_dt) - 1e-3:
                last_write, last_accel = r.t, r.accel_g
                self._put("imu", dict(
                    t=round(r.t, 4), accel_g=[round(v, 4) for v in r.accel_g],
                    gyro_dps=[round(v, 3) for v in r.gyro_dps], roll=round(r.roll_deg, 2),
                    pitch=round(r.pitch_deg, 2), yaw=round(r.yaw_deg, 2), tilt=round(r.tilt_deg, 2),
                    tipped=r.tipped, bump_g=round(r.bump_g, 3), temp_c=round(r.temp_c, 1)))
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            else:
                next_t = time.monotonic()

    # ---------- the distance sensor ----------

    def _run_sonar(self, stop):
        sonar = self._sonar
        if sonar is None:
            from .sonar import Sonar
            sonar = Sonar()
        self.status["sonar"] = "ok"
        period = 1.0 / config.DATA_SONAR_HZ
        last, last_t = object(), -1e9
        try:
            while not stop.is_set():
                t0 = time.monotonic()
                r = sonar.ping()
                cm = r if isinstance(r, float) else None
                changed = (type(r) is not type(last) or (cm is None and r != last)
                           or (cm is not None and abs(cm - last) >= config.DATA_SONAR_CHANGE_CM))
                if changed or t0 - last_t >= config.DATA_SONAR_IDLE_S:
                    self._put("sonar", {"t": round(t0, 4), "cm": r} if cm is not None else
                                   {"t": round(t0, 4), "cm": None, "why": r})
                    last, last_t = r, t0
                stop.wait(max(0.0, period - (time.monotonic() - t0)))
        finally:
            if self._sonar is None:
                sonar.close()

    # ---------- the Pi itself ----------

    def _run_system(self, stop):
        power = self._power
        if power is None:
            try:
                from .power import PiPowerReader
                power = PiPowerReader()
            except Exception:  # noqa: BLE001 - not a Pi
                power = None
        self.status["system"] = "ok"
        while not stop.is_set():
            self._put("system", system_sample(self.rec.root, power))
            stop.wait(config.DATA_SYSTEM_S)
