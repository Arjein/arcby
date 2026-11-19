"""
The robot's drive hardware glue, shared by code/nav.py (its own web page) and alive.py (robot/legs.py):

    Head(pan_tilt)          nav.py's head: moved toward a target at NAV_HEAD_DPS in its own 50 Hz thread, remembers
                            where it pointed when, retries a silent board (alive.py drives the head through HeadLife's
                            "drive" mode instead)
    SonarThread(sonar)      the sonar pinged at NAV_SONAR_HZ; .last = (t, cm or None = nothing in range)
    SonarReader(thread)     that thread's newest echo as a Sonar for robot/sensors.py (one program pings, not two)
    StateImu(state)         the IMU readings robot.main.Robot publishes, as an Imu for robot/sensors.py (one reader)
    people_in(bodies, ...)  pose-model bodies -> navigation's [Person] (bearing, distance from the feet)
"""
import logging
import math
import threading
import time
from collections import deque

from . import config
from .floor import ground_points
from .nav import Person
from .pan_tilt import PanTiltError

log = logging.getLogger("arcby")


class Head:
    """The pan-tilt head moved smoothly toward a target in its own 50 Hz thread; remembers where it pointed when.
    A failed I2C write is retried (the board is re-initialised: a dip on the servo rail resets it); only when the
    board stays silent NAV_HEAD_DOWN_S does the thread give up (and with it the drive). 2026-11-04: two single
    EIOs from 0x40 in a minute stopped nav.py; later that night the re-init itself failed (PanTiltError, not an
    OSError) and crashed the thread on the first error."""

    def __init__(self, pan_tilt, clock=time.monotonic, sleep=time.sleep):
        self.pt = pan_tilt
        self.clock = clock
        self.target = (0.0, config.NAV_TILT_DEG)
        self.now = (0.0, 0.0)
        self.speed = 0.0
        self.hist = deque(maxlen=200)       # (t, pan, tilt, deg/s)
        self._lock = threading.Lock()
        self.errors = 0
        self._down_since = None
        self._reinit = False
        for attempt in range(3):
            try:
                self.pt.init()
                self.now = self.pt.set(0.0, config.NAV_TILT_DEG)
                break
            except (OSError, PanTiltError):
                self.errors += 1
                if attempt == 2:
                    raise
                log.warning("nav: head board did not answer at start - trying again")
                sleep(0.3)

    def at(self, t):
        """(pan, tilt, deg/s) at time t (the newest sample not after t)."""
        with self._lock:
            best = None
            for row in self.hist:
                if row[0] <= t:
                    best = row
                else:
                    break
        return best[1:] if best else (*self.now, self.speed)

    def tick(self, dt=0.02):
        """One step toward the target. Raises (OSError / PanTiltError) when the board has been silent NAV_HEAD_DOWN_S."""
        tp, tt = self.target
        p, t = self.now
        step = config.NAV_HEAD_DPS * dt
        np_ = p + max(-step, min(step, tp - p))
        nt = t + max(-step, min(step, tt - t))
        try:
            if self._reinit:
                self.pt.init()
                self.now = self.pt.set(p, t)            # where the head was: a reset board let it go limp
                self._reinit = False
            if (np_, nt) != (p, t):
                self.now = self.pt.set(np_, nt)
            if self._down_since is not None:
                log.info("nav: head board answers again (%.2f s silent)", self.clock() - self._down_since)
            self._down_since = None
        except (OSError, PanTiltError) as e:
            self.errors += 1
            self._reinit = True
            now = self.clock()
            if self._down_since is None:
                self._down_since = now
                log.warning("nav: head board write failed (%s) - retrying (%d errors so far)", e, self.errors)
            elif now - self._down_since >= config.NAV_HEAD_DOWN_S:
                raise
            np_, nt = p, t
        self.speed = max(abs(np_ - p), abs(nt - t)) / dt
        with self._lock:
            self.hist.append((self.clock(), *self.now, self.speed))

    def run(self, stop):
        while not stop.is_set():
            self.tick()
            stop.wait(0.02)


class SonarThread:
    def __init__(self, sonar):
        self.sonar = sonar
        self.last = None                    # (t, cm or None)
        self.answered = 0

    def run(self, stop):
        while not stop.is_set():
            t = time.monotonic()
            r = self.sonar.ping()
            if isinstance(r, float):
                self.last = (t, r)
                self.answered += 1
            elif r == "nothing":
                self.last = (t, None)
                self.answered += 1
            stop.wait(max(0.0, 1.0 / config.NAV_SONAR_HZ - (time.monotonic() - t)))


def people_in(bodies, width, height, pan, tilt):
    """Body detections -> [Person]: bearing from the box middle, distance from the feet if the box's bottom is in
    the picture (its ground point; robot/floor.py geometry)."""
    from robot.camera import angles_of
    out = []
    for b in bodies:
        if b.score < config.NAV_PERSON_SCORE:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.box)
        ax, _ = angles_of((x1 + x2) / 2, (y1 + y2) / 2, width, height)
        bearing = pan - ax
        dist = None
        if y2 < height - 6:                 # the feet are in the picture
            gx, gy, ok = ground_points((x1 + x2) / 2, y2, width, height, pan, tilt)
            if ok:
                dist = float(math.hypot(gx, gy))
        out.append(Person(bearing, dist, float(b.score)))
    return out


class SonarReader:
    """A SonarThread's newest echo with the robot/sonar.py interface (ping() -> cm, "nothing" or why not) - so
    robot/sensors.py records the distance without a second program pinging the same sensor."""

    def __init__(self, thread, max_age=0.5, clock=time.monotonic):
        self.thread, self.max_age, self.clock = thread, max_age, clock

    def ping(self):
        last = self.thread.last
        if last is None or self.clock() - last[0] > self.max_age:
            return "no echo"
        return "nothing" if last[1] is None else float(last[1])

    def close(self):
        pass


class StateImu:
    """The newest IMU reading in a robot.state.State (robot.main.Robot's own IMU thread publishes it) with the
    Imu.step() interface robot/sensors.py uses: a reading once per new sample, else None."""

    def __init__(self, state):
        self.state = state
        self._last_t = None

    def start(self):
        return self

    def step(self):
        r = self.state.get("imu")
        if r is None or r.t == self._last_t:
            return None
        self._last_t = r.t
        return r
