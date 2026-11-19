"""
Legs: alive.py drives (docs/navigation.md). The navigation brain of code/nav.py (robot/nav.py Navigator, the
floor check robot/floor.py, the short memory robot/localmap.py) inside the living robot - no web page, no dead man:
every stop counts instead.

    legs = Legs(robot, sonar_thread, head_now=lambda: life.out, rec=recorder)   # robot = an entered robot.main.Robot
    legs.run(stop_event)                    # own thread: one navigation step every 1 / NAV_RATE_HZ
    legs.start("explore", why="dashboard")  # modes: explore / people (walk up to people) / follow ("gel gel")
    legs.frame(rgb, t, pan, tilt, head_dps) # every camera frame while active (the floor check)
    legs.see_people(people, t)              # [nav.Person] from each new pose-model result
    legs.follow_target(person, t)           # follow mode: the one followed (FollowTarget picks their body track)
    legs.head_target                        # (pan, tilt) the drive wants (HeadLife "drive" mode), None = not driving
    legs.stop("Arcby dur")                   # at once: motors to 0 (they ramp down within ~0.15 s), head back
    legs.events                             # [{"kind": "drive", ...}] for alive.py's event log, taken by drain()

    bored = Boredom()                       # wander by itself when nobody was around for long (BORED_EXPLORE_S)
    bored.update(t, someone_in_view, legs, drive_events)

A drive stops - and stays stopped until a new start - on: stop() (voice, dashboard, a stop sign), any latched safety
stop (bump, tipped, low battery: robot/safety.py), the navigator asking for HELP or being DONE (follow: lost them),
the robot's own threads failing, or max_s running out. The motors module's watchdog is under all of it: no fresh request for 0.3 s = stop.
"""
import logging
import math
import threading
import time

from . import config
from .floor import FloorSeer, ground_points
from .nav import Command, Navigator, Percept, Person

log = logging.getLogger("arcby")

MODES = ("explore", "people", "follow")


class Legs:
    def __init__(self, robot, sonar, head_now, rec=None, recorder_factory=None, clock=time.monotonic):
        """robot: robot.main.Robot (motors, safety, state); sonar: drive_body.SonarThread; head_now: () -> (pan, tilt)
        the head shows now; rec: alive.py's Recorder - each drive goes into it (robot/nav_rec.py)."""
        self.robot, self.sonar, self.head_now = robot, sonar, head_now
        self.rec = rec
        self._rec_factory = recorder_factory
        self.clock = clock
        self.seer = FloorSeer()
        self.nav = None
        self.mode = "stopped"
        self.why = "not started yet"        # why it is stopped / why it started
        self.started_t = self.stopped_t = None
        self.max_s = None
        self.view = None
        self.people, self.people_t = [], None
        self.target, self.target_t = None, None     # follow mode: the person followed
        self.force_learn = False
        self.last_cmd, self.last_p = Command(), None
        self.events = []
        self.drives = 0
        self._nav_rec = None
        self._lock = threading.RLock()

    # ---------- outside ----------

    @property
    def active(self):
        return self.mode != "stopped"

    @property
    def applied(self):
        return self.robot.motors.applied

    @property
    def head_target(self):
        """Where the drive wants the head (HeadLife "drive" mode), or None while stopped (attention looks)."""
        with self._lock:
            return self.last_cmd.head if self.active else None

    def start(self, mode, why="", max_s=None):
        """Start (or switch to) a drive. Returns (True, "") or (False, why not)."""
        if mode not in MODES:
            raise ValueError(f"unknown drive mode {mode!r}")
        with self._lock:
            if not self.robot.alive:
                return self._refuse(f"the motors are off ({self.robot.fault or 'stopped'})")
            self.robot.safety.reset()                       # a new start clears a bump; tipped / low battery stay
            s = self.robot.state.get("safety", max_age=config.SAFETY_MAX_AGE_S)
            latched = (s or {}).get("latched", [])
            if latched and set(latched) - {"bump"}:
                return self._refuse("safety: " + ", ".join(latched))
            if self.mode == mode and self.nav is not None:
                self.max_s = max_s
                return True, ""
            self.nav = Navigator(mode)
            if mode != "follow":
                self.target = self.target_t = None
            if not self.active:
                self.seer.forget()                          # maybe another floor now: learn it again in front
                self.view = None
                self._open_rec()
            self.mode, self.why, self.max_s = mode, why, max_s
            self.started_t, self.drives = self.clock(), self.drives + 1
            self.last_cmd = Command()
            self._event("started", why)
            log.info("drive: %s started (%s)", mode, why or "-")
            return True, ""

    def stop(self, why):
        """Stand still now; stays stopped until the next start(). False if it was not driving."""
        with self._lock:
            self.robot.motors.request_velocity(0.0, 0.0, "legs")
            if not self.active:
                return False
            mode = self.mode
            self.mode, self.why, self.stopped_t = "stopped", why, self.clock()
            if self.nav is not None:
                self.nav.pause()
            self.force_learn = False
            self._close_rec()
            self._event("stopped", why, mode=mode)
            log.info("drive: %s stopped (%s)", mode, why)
            return True

    def drain(self):
        with self._lock:
            out, self.events = self.events, []
        return out

    def frame(self, rgb, t, pan, tilt, head_dps):
        """One camera frame (alive.py's small stream) with the head pose it was taken at: the floor check. Only while
        driving; frames with the head or the body turning fast are skipped (blurred, the pose unsure)."""
        if not self.active or self.mode == "follow":     # following: the head looks at them, not at the floor
            return None
        imu = self.robot.state.get("imu", max_age=0.3)
        if head_dps >= 40.0 or (imu is not None and abs(imu.yaw_rate_dps) > 20.0):
            return None
        free = self.sonar_free()
        v = self.seer.see(rgb, pan, tilt, t, sonar_free_cm=free, force_learn=self.force_learn)
        self.view = v
        rec = self._nav_rec
        if rec is not None:
            try:
                rec.frame(rgb, v, t, free)
            except Exception:  # noqa: BLE001 - recording never stops a drive
                log.exception("drive: recording a frame failed")
        return v

    def see_people(self, people, t):
        self.people, self.people_t = list(people), t

    def follow_target(self, person, t):
        """Follow mode: the newest sighting of the one followed (a nav.Person: bearing, distance, tilt)."""
        self.target, self.target_t = person, t

    def sonar_free(self):
        last = self.sonar.last
        if last is None or self.clock() - last[0] > 0.5:
            return None
        return float("inf") if last[1] is None else last[1] - (config.NAV_FRONT_CM - config.NAV_SONAR_X_CM)

    # ---------- the loop ----------

    def percept(self):
        st = self.robot.state
        imu = st.get("imu", max_age=config.SAFETY_IMU_STALE_S)
        son = self.sonar.last
        return Percept(t=self.clock(), yaw=None if imu is None else imu.yaw_deg,
                       yaw_rate=0.0 if imu is None else imu.yaw_rate_dps,
                       sonar=None if son is None else son[1], sonar_t=None if son is None else son[0],
                       floor=self.view, people=list(self.people), people_t=self.people_t,
                       safety=st.get("safety", max_age=config.SAFETY_MAX_AGE_S), head=tuple(self.head_now()),
                       target=self.target, target_t=self.target_t)

    def step(self):
        p = self.percept()
        with self._lock:
            self.last_p = p
            if not self.active:
                return None
            why = self._must_stop(p)
            if why:
                self.stop(why)
                return None
            cmd = self.nav.step(p)
            if self.nav.state == "HELP":
                self.stop("stuck: " + (cmd.note or self.nav.note or "no way out"))
                return None
            if self.nav.state == "DONE":
                self.stop(self.nav.note or "done")
                return None
            self.robot.motors.request_velocity(cmd.v, cmd.turn, "legs")
            self.force_learn = cmd.learn_floor
            self.last_cmd = cmd
            rec = self._nav_rec
        if rec is not None:
            try:
                rec.step(p, cmd, True)
            except Exception:  # noqa: BLE001
                log.exception("drive: recording a step failed")
        return cmd

    def _must_stop(self, p):
        if not self.robot.alive:
            return f"the motors stopped ({self.robot.fault or 'robot stopped'})"
        latched = (p.safety or {}).get("latched", [])
        if latched:
            return "safety: " + ", ".join(latched)
        if self.max_s is not None and p.t - self.started_t >= self.max_s:
            return f"time up ({self.max_s:.0f} s)"
        return None

    def run(self, stop):
        period = 1.0 / config.NAV_RATE_HZ
        nxt = self.clock()
        while not stop.is_set():
            try:
                self.step()
            except Exception as e:  # noqa: BLE001 - a broken step stops the drive, never alive.py
                log.exception("drive: a step failed")
                self.stop(f"error: {type(e).__name__}: {e}")
            nxt += period
            stop.wait(max(0.0, nxt - self.clock()))
            if nxt < self.clock() - 1.0:
                nxt = self.clock()
        self.stop("alive.py stops")

    # ---------- status ----------

    def status(self):
        """For the dashboard (state.json "drive")."""
        with self._lock:
            c, p, nav = self.last_cmd, self.last_p, self.nav
            now = self.clock()
            out = dict(mode=self.mode, why=self.why, state=(c.state or "-") if self.active else "-",
                       note=c.note if self.active else "", v=round(c.v, 2), turn=round(c.turn),
                       for_s=round(now - (self.started_t if self.active else (self.stopped_t or now)), 1),
                       max_s=self.max_s, drives=self.drives, fault=self.robot.fault)
            if p is not None:
                out["sonar"] = None if p.sonar_t is None else ("nothing" if p.sonar is None else round(p.sonar))
                out["safety"] = (p.safety or {}).get("reasons", ["no verdict"])
            if nav is not None:
                out["stats"] = {k: round(v) for k, v in nav.stats.items()}
        return out

    # ---------- parts ----------

    def _refuse(self, why):
        log.info("drive: not starting - %s", why)
        self._event("refused", why)
        return False, why

    def _event(self, what, why, mode=None):
        self.events.append(dict(kind="drive", what=what, mode=mode or self.mode, why=why))
        del self.events[:-50]

    def _open_rec(self):
        if self.rec is None and self._rec_factory is None:
            return
        try:
            if self._rec_factory is not None:
                self._nav_rec = self._rec_factory()
            else:
                from .nav_rec import NavRecorder
                self._nav_rec = NavRecorder(rec=self.rec)
        except Exception as e:  # noqa: BLE001 - drive without recording
            log.warning("drive: not recording (%s)", e)
            self._nav_rec = None

    def _close_rec(self):
        rec, self._nav_rec = self._nav_rec, None
        if rec is not None:
            try:
                rec.close()
            except Exception:  # noqa: BLE001
                log.exception("drive: closing the recording failed")


class FollowTarget:
    """Which body track is the person being followed (alive.py, follow mode). The one that beckoned; when its track
    breaks (out of view a moment, the robot turned), the same known face's body, else a new body showing up near where
    they were last (FOLLOW_REFIND_DEG) - clothes are not compared yet."""

    def __init__(self, body_id, person=None, pan=None, t=None):
        self.body_id, self.person = body_id, person
        self.last = (pan, t)                # where / when they were last seen (head angles, time)

    def pick(self, tracks, t):
        """tracks = robot/body.py visible BodyTracks. Returns the target's track or None."""
        tr = next((b for b in tracks if b.id == self.body_id), None)
        if tr is None and self.person is not None:
            tr = next((b for b in tracks if b.person == self.person), None)
        if tr is None and self.last[0] is not None and self.last[1] is not None and \
                t - self.last[1] <= config.FOLLOW_LOST_S + config.FOLLOW_REFIND_S:
            near = [b for b in tracks if b.first_t >= self.last[1] - 0.5 and
                    abs(b.pan - self.last[0]) <= config.FOLLOW_REFIND_DEG]
            tr = min(near, key=lambda b: abs(b.pan - self.last[0]), default=None)
        if tr is not None:
            if tr.id != self.body_id:
                log.info("drive: following body %d now (was %d)", tr.id, self.body_id)
            self.body_id = tr.id
            self.person = self.person or tr.person
            self.last = (tr.pan, tr.last_t)
        return tr


def target_person(tr, size, pose):
    """A body track -> nav.Person to follow: bearing = its middle, distance from its feet when they are in the picture
    (else from its height when the whole person is in it), tilt = where the head should look (its chest)."""
    w, h = size
    x1, y1, x2, y2 = (float(v) for v in tr.body.box)
    dist = None
    if y2 < h - 6:                          # the feet are in the picture
        gx, gy, ok = ground_points((x1 + x2) / 2, y2, w, h, pose[0], pose[1])
        if ok:
            dist = float(math.hypot(gx, gy))
    if dist is None and y1 > 4 and y2 < h - 4 and tr.height_deg > 1.0:
        dist = config.FOLLOW_PERSON_CM / (2 * math.tan(math.radians(tr.height_deg) / 2))
    tilt = tr.tilt + 0.25 * tr.height_deg   # a little above the middle of the box: the chest
    return Person(float(tr.pan), dist, float(tr.body.score), tilt=float(tilt))


PERSON_STOPS = ("voice", "stop sign", "dashboard")     # a drive stopped by someone (not by safety / time / lost)


class Boredom:
    """Wander when bored (docs/navigation.md part 5): explore by itself for WANDER_MAX_S after nobody was seen
    for BORED_EXPLORE_S - only by day (not in TTS_NIGHT_HOURS, the household's quiet hours), not within
    BORED_AFTER_STOP_S of a person stopping a drive. Someone showing up while it wanders: it stops (and alive.py's
    attention / greeter do the rest). No position memory (no encoders): it does not drive back to where it was."""

    def __init__(self, t=None, wall=time.time, tz=None):
        self.seen_t = time.monotonic() if t is None else t
        self.person_stop_t = -math.inf
        self.wall = wall
        self.tz = tz
        self.wanders = 0

    def night(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        hour = datetime.fromtimestamp(self.wall(), self.tz or ZoneInfo(config.TTS_TIMEZONE)).hour
        a, b = config.TTS_NIGHT_HOURS
        return hour >= a or hour < b

    def update(self, t, someone, legs, drive_events=()):
        """One camera frame. Returns "started" / "stopped" when it started / ended a wander, else None."""
        for e in drive_events:
            if e.get("what") == "stopped" and str(e.get("why", "")).startswith(PERSON_STOPS):
                self.person_stop_t = t
            if e.get("what") == "stopped":
                self.seen_t = max(self.seen_t, t)       # the next wander only after another quiet stretch
        if someone:
            self.seen_t = t
            if legs.active and legs.mode == "explore" and legs.why == "bored":
                legs.stop("someone is here")
                return "stopped"
            return None
        if legs.active or not config.BORED_ENABLED:
            return None
        if t - self.seen_t < config.BORED_EXPLORE_S or t - self.person_stop_t < config.BORED_AFTER_STOP_S:
            return None
        if self.night():
            return None
        ok, _ = legs.start("explore", why="bored", max_s=config.WANDER_MAX_S)
        self.seen_t = t                                 # refused (tipped...) or started: not again at once
        if ok:
            self.wanders += 1
            return "started"
        return None
