"""
Starts the robot modules (imu, safety, motors, power) as threads and owns shutdown.

    with Robot() as robot:          # real hardware
        robot.motors.request(0.3, 0.3, "me")

Any thread that crashes stops everything (and so the motors); leaving the
`with` block - normally, by Ctrl+C or by an exception - stops the motors and
releases the pins. Hardware can be swapped for fakes (driver=, imu_dev=).
On the real hardware it also writes the power breadcrumb (robot/breadcrumb.py: applied PWM once a second, fsync'ed)
and logs the previous run's last line; `robot.breadcrumb.state = lambda: "..."` adds what the program is doing.
"""
import logging
import os
import sys
import threading
import traceback

from . import config, imu
from .motors import GpioDriver, Motors
from .power import PowerMonitor
from .safety import Safety
from .state import State

log = logging.getLogger("arcby")


class Robot:
    def __init__(self, driver=None, imu_dev=None, imu_filter=None, power_reader=None,
                 blackbox_dir=config.BLACKBOX_DIR, breadcrumb=None):
        """breadcrumb: None = only on the real hardware (no driver= given), True / False = always / never."""
        self.state = State()
        self._blackbox_dir = blackbox_dir
        self._imu_filter = imu_filter
        self._power_reader = power_reader       # None = the Pi's own under-voltage alarm
        self.stop_event = threading.Event()
        self._driver_arg, self._imu_dev_arg = driver, imu_dev
        self.driver = self.imu_dev = None
        self.threads = []
        self.fault = None
        self._want_crumb = breadcrumb
        self.breadcrumb = None

    def __enter__(self):
        try:
            if self._imu_dev_arg is not None:
                self.imu_dev = self._imu_dev_arg
            else:
                from .i2c import I2CDevice
                self.imu_dev = I2CDevice(config.I2C_BUS, config.IMU_ADDR)
            self.driver = self._driver_arg if self._driver_arg is not None else GpioDriver()
            self.imu = imu.Imu(self.state, dev=self.imu_dev, filt=self._imu_filter)
            self.safety = Safety(self.state, blackbox_dir=self._blackbox_dir)
            self.motors = Motors(self.state, self.driver)
            self.power = PowerMonitor(self.state, reader=self._power_reader)
            self.imu.start()            # raises at once if the IMU does not answer
        except BaseException:
            self._close()
            raise
        for name, target in (("imu", self.imu.run_started), ("safety", self.safety.run),
                             ("motors", self.motors.run), ("power", self.power.run)):
            th = threading.Thread(target=self._guard, args=(name, target), name=name, daemon=True)
            th.start()
            self.threads.append(th)
        want = self._driver_arg is None if self._want_crumb is None else self._want_crumb
        if want:
            from .breadcrumb import Breadcrumb, boot_id, describe
            self.breadcrumb = Breadcrumb(os.path.basename(sys.argv[0]) or "?", pwm=lambda: self.motors.applied)
            prev = self.breadcrumb.start(self.stop_event)
            log.log(logging.WARNING if prev and not prev.get("exit") else logging.INFO, "%s", describe(prev, boot_id()))
        log.info("robot started")
        return self

    def _guard(self, name, target):
        try:
            target(self.stop_event)
        except Exception as e:  # noqa: BLE001 - any crash must stop the robot
            self.fault = f"{name} crashed: {type(e).__name__}: {e}"
            log.error("%s\n%s", self.fault, traceback.format_exc())
            self.state.publish("fault", self.fault)
            self.stop_event.set()

    @property
    def alive(self):
        return not self.stop_event.is_set()

    def __exit__(self, *exc):
        self.stop_event.set()
        for th in self.threads:
            th.join(timeout=1.0)
        if self.breadcrumb is not None:
            self.breadcrumb.close()
        self._close()
        log.info("robot stopped")
        return False

    def _close(self):
        if self.driver is not None:
            try:
                self.driver.stop()
            finally:
                self.driver.close()
        if self.imu_dev is not None and self._imu_dev_arg is None:
            self.imu_dev.close()
