"""
Navigation brain: explore carefully, go to people (docs/navigation.md).

Pure logic - no hardware, no threads, no clock of its own: code/nav.py (the robot) and code/sim/navworld.py (the
simulated room, the tests) feed it a Percept 20 times a second and carry out the Command it returns.

  START    wait for the IMU, the safety verdict and a learnt floor (robot/floor.py learns it from the patch in
           front while the sonar says that patch is free)
  SCAN     stand still, look around with the head (NAV_SCAN_PANS, tilted down at the floor)
  CHOOSE   the best direction: the longest way that is free of obstacles AND was seen free, a little less for
           bigger turns, more for ground not visited yet. Nothing good -> turn round (135 deg) and look again;
           NAV_STUCK_SCANS times in a row -> HELP
  TURN     on the spot with the gyro (fast spin, stopped early by a lead learnt per turn). Something inside the
           spin circle -> BACK first
  DRIVE    slow (NAV_V_CRUISE, crawling within NAV_SLOW_CM), steering gently toward the chosen heading (picked
           again every NAV_REPLAN_S within NAV_REPLAN_DEG), the head looking where it goes; stops NAV_STOP_CM before
           anything, or after NAV_LEG_S to look around again. Drives only through ground the camera saw free in the
           last NAV_SEEN_S
  CROSS    a floor edge (rug) ahead and the sonar free: crawl over it, learning the new floor
  BACK     reverse slowly at most NAV_BACK_CM, and only as far as it came forward (nothing sees behind)
  BUMPED   the IMU felt a hit: remember it, back up, clear the stop (at most NAV_BUMP_RESETS in NAV_BUMP_WINDOW_S)
  APPROACH (people mode) a person seen: turn to them, crawl to NAV_PERSON_STOP_CM from their feet
  GREET    look up at them for NAV_PERSON_GREET_S, then leave them alone for NAV_PERSON_COOLDOWN_S
  HELP     stopped for good, until resume() (the web page's Start)

Follow mode (alive.py: someone beckoned, "gel gel") - the Percept's `target` is that one person:
  FOLLOW   the head looks at them (so the floor check is off: obstacles = sonar + memory + bump, at crawl speed
           behind a person who walked there); turn on the spot when they are more than FOLLOW_TURN_DEG aside, crawl
           steering at them, stand FOLLOW_STOP_CM from them (feet / sonar), go again when they are FOLLOW_GO_CM farther
  LOST     not seen for FOLLOW_LOST_S: stand, look around once (FOLLOW_LOOK_PANS); seen again -> FOLLOW, else DONE
  DONE     the drive is over ("lost them") - robot/legs.py stops it

The safety layer (robot/safety.py: tipped, bump, IMU lost, low battery) is below all this and stops the motors
whatever the brain wants; the brain only reacts to it.
"""
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from . import config
from .localmap import LocalMap

log = logging.getLogger("arcby")

CANDIDATES = np.arange(-100.0, 100.1, 5.0)     # directions considered after a look around (deg, + = left)


@dataclass
class Person:
    bearing: float          # deg from the robot's heading, + = left
    dist: float = None      # cm from the axle to their feet (None = feet not seen)
    score: float = 1.0
    tilt: float = None      # head tilt that looks at them (follow mode), None = unknown


@dataclass
class Percept:
    t: float
    yaw: float = None               # IMU heading (deg, + = left); None = no IMU reading
    yaw_rate: float = 0.0
    sonar: float = None             # newest echo (cm from the sonar face); None = nothing in range
    sonar_t: float = None           # when that ping was taken; None = no sonar at all
    floor: object = None            # newest FloorView (head still enough when it was taken), or None
    people: list = field(default_factory=list)  # [Person] from the newest body detection
    people_t: float = None
    safety: dict = None             # robot/safety.py verdict
    head: tuple = (0.0, 0.0)        # head angles now
    target: Person = None           # follow mode: the person followed, newest sighting (None = not seen yet)
    target_t: float = None          # when that sighting was taken


@dataclass
class Command:
    v: float = 0.0
    turn: float = 0.0
    head: tuple = (0.0, config.NAV_TILT_DEG)
    reset_safety: bool = False
    learn_floor: bool = False       # learn the patch ahead whatever it looks like (crossing onto a new floor)
    state: str = ""
    note: str = ""


def speed_cm_s(v):
    """Estimated speed for a request (no encoders: from config, NAV_CRAWL_CM_S / NAV_CRUISE_CM_S)."""
    a, b = abs(config.NAV_V_CRAWL), abs(config.NAV_V_CRUISE)
    if abs(v) < 0.005:
        return 0.0
    s = config.NAV_CRAWL_CM_S + (abs(v) - a) * (config.NAV_CRUISE_CM_S - config.NAV_CRAWL_CM_S) / (b - a)
    return math.copysign(max(5.0, s), v)


class Navigator:
    def __init__(self, mode="explore"):
        if mode not in ("explore", "people", "follow"):
            raise ValueError(f"mode must be explore, people or follow, not {mode!r}")
        self.mode = mode
        self.map = LocalMap()
        self.state, self.state_t = "START", None
        self.note = ""
        self.t = None
        self._speed = 0.0               # estimated cm/s now (lags the request a little)
        self._last_v = 0.0
        self._floor_t = self._sonar_t = self._people_t = None
        self._sonar = None              # (t, cm) of the newest echo
        self._sonar_hist = []           # (t, cm) while driving: stuck check
        self._view = None               # newest floor view
        # per state
        self._scan_i = 0
        self._scan_order = config.NAV_SCAN_PANS
        self._scan_since = None
        self._scan_t0 = None
        self._target_yaw = None
        self._after_turn = "DRIVE"
        self._turn_sign = 0.0
        self._lead = {"small": config.NAV_TURN_LEAD_S, "big": config.NAV_TURN_LEAD_S}    # learnt per turn size
        self._turn_size = "big"
        self._turn_rate_stop = config.NAV_TURN_DPS
        self._turn_t0 = None
        self._settle_t = None
        self._dt = 0.0
        self._head = (0.0, 0.0)
        self._back_left = 0.0
        self._after_back = "SCAN"
        self._fwd_since_turn = 0.0
        self._leg_t0 = None
        self._replan_t = None
        self._cross_left = 0.0
        self._cross_edge = 0.0
        self._scan_edge = None          # (edge_cm, cm ahead of the front) of the newest floor edge seen in this look
        self._failed_scans = 0
        self._round_sign = 0.0
        self._bump_resets = []
        self._bump_seen = False
        self._reset_sent_t = None
        self._person = None             # (odom bearing deg, dist, t seen)
        self._dists = []
        self._cooldown = []             # (odom bearing deg, until t)
        self._greet_t0 = None
        self._target = None             # follow: (odom bearing deg, dist or None, t seen, tilt or None)
        self._target_t = None
        self._tdists = []
        self._waiting = False           # follow: standing near them
        self._lost_t0 = None
        self._look_i = 0
        self._look_since = None
        self.events = []                # (t, text): what happened, for the log / page
        self.stats = {"driven_cm": 0.0, "turns": 0, "scans": 0, "bumps": 0, "stuck": 0, "greeted": 0,
                      "crossed": 0, "followed_cm": 0.0}

    # ---------- outside ----------
    def pause(self):
        """The robot was held still from outside (Stop, the page closed): look around again when it goes on."""
        self.t = None                       # no dead reckoning across the pause
        self._last_v = self._speed = 0.0
        if self.state not in ("START", "HELP", "SCAN", "DONE", self._home()):
            self._go(self._home(), "paused")

    def _home(self):
        """Where the brain goes back to after a turn, a back-up, a pause: looking around, or following."""
        return "FOLLOW" if self.mode == "follow" else "SCAN"

    def resume(self):
        """Start again after HELP (the operator looked)."""
        self._failed_scans = 0
        self._go("SCAN", "resumed")

    def _event(self, text):
        self.events.append((self.t, text))
        del self.events[:-50]
        log.info("nav: %s", text)

    def _go(self, state, why=""):
        if state != self.state:
            self._event(f"{self.state} -> {state}" + (f" ({why})" if why else ""))
        self.state, self.state_t = state, self.t
        if state == "SCAN":
            self._scan_i, self._scan_since, self._scan_t0 = 0, None, self.t
            self._scan_edge = None
            pans = config.NAV_SCAN_PANS
            near_end = abs(self._head[0] - pans[-1]) < abs(self._head[0] - pans[0])
            self._scan_order = pans[::-1] if near_end else pans
            self.stats["scans"] += 1
        elif state == "DRIVE":
            self._leg_t0 = self._replan_t = self.t
            self._sonar_hist = []
        elif state == "APPROACH":
            self._leg_t0 = self.t
            self._sonar_hist = []
        elif state == "GREET":
            self._greet_t0 = self.t
        elif state == "LOST":
            self._lost_t0, self._look_i, self._look_since = self.t, 0, None
        elif state == "FOLLOW":
            self._leg_t0 = self.t
            self._sonar_hist = []

    # ---------- every step ----------
    def step(self, p):
        prev_t = self.t if self.t is not None else p.t
        self.t = p.t
        dt = min(0.2, max(0.0, p.t - prev_t))
        self._dt, self._head = dt, p.head
        # dead reckoning: the estimated speed of what was asked last step (first-order lag ~0.15 s)
        goal = speed_cm_s(self._last_v)
        self._speed += (goal - self._speed) * min(1.0, dt / 0.15)
        if p.yaw is not None:
            d = self._speed * dt
            self.map.move(p.yaw, d)
            if d > 0:
                self._fwd_since_turn += d
                self.stats["driven_cm"] += d
                if self.state == "FOLLOW":
                    self.stats["followed_cm"] += d
        self._ingest(p)
        self.map.forget(p.t)
        cmd = self._decide(p)
        self._last_v = cmd.v
        cmd.state, cmd.note = self.state, self.note
        return cmd

    def _ingest(self, p):
        if p.sonar_t is not None and p.sonar_t != self._sonar_t:
            self._sonar_t = p.sonar_t
            self.map.add_sonar(p.sonar, p.sonar_t)
            self._sonar = (p.sonar_t, p.sonar)
            if self.state in ("DRIVE", "APPROACH", "FOLLOW") and p.sonar is not None:
                self._sonar_hist.append((p.sonar_t, p.sonar, p.yaw))
                self._sonar_hist = [s for s in self._sonar_hist if p.t - s[0] <= config.NAV_STUCK_S + 0.5]
        if p.floor is not None and p.floor.t != self._floor_t and abs(p.yaw_rate) < 20.0:
            self._floor_t = p.floor.t
            self._view = p.floor
            edge_ok = (p.floor.edge and p.floor.edge_cm is not None
                       and self._edge_sonar_ok(p.t, p.floor.edge_cm - (config.NAV_FRONT_CM - p.floor.ox)))
            if p.floor.edge and not edge_ok:
                p.floor.edge = False        # an "edge" the sonar says is a wall: an obstacle
            if self.state == "SCAN" and p.floor.edge and p.floor.edge_cm is not None:
                self._scan_edge = (p.floor.edge_cm, p.floor.edge_cm - (config.NAV_FRONT_CM - p.floor.ox))
            self.map.add_view(p.floor, p.floor.t)
        if p.people_t is not None and p.people_t != self._people_t:
            self._people_t = p.people_t
            self._see_people(p)
        if p.target is not None and p.target_t is not None and p.target_t != self._target_t and p.yaw is not None:
            self._target_t = p.target_t
            tg = p.target
            self._tdists = self._tdists[-4:] + [tg.dist]
            known = [d for d in self._tdists if d is not None]
            self._target = (p.yaw + tg.bearing, float(np.median(known)) if known else None, p.target_t, tg.tilt)

    def sonar_gap(self, t, max_age=0.35):
        """Free cm ahead of the front by the newest echo (inf = nothing in range, 0 if no fresh echo)."""
        if self._sonar is None or t - self._sonar[0] > max_age:
            return 0.0
        cm = self._sonar[1]
        if cm is None:
            return math.inf
        return max(0.0, cm - (config.NAV_FRONT_CM - config.NAV_SONAR_X_CM))

    def sonar_echo_gap(self, t, max_age=0.35):
        """Like sonar_gap, but only a real echo counts: "nothing" can also be a wall hit at a slant (the sound
        bounces away) - not proof enough to cross a floor edge."""
        if self._sonar is None or t - self._sonar[0] > max_age or self._sonar[1] is None:
            return 0.0
        return self.sonar_gap(t, max_age)

    def _edge_sonar_ok(self, t, edge_ahead):
        """A floor edge edge_ahead cm in front is a floor change, not the foot of a wall or a box: the sonar's echo
        comes from well past it."""
        need = max(config.NAV_EDGE_SONAR_MIN_CM, edge_ahead + config.NAV_EDGE_SONAR_PAST_CM)
        return self.sonar_echo_gap(t) >= need

    # ---------- decisions ----------
    def _decide(self, p):
        s = p.safety or {"stop": True, "reasons": ["no safety verdict"], "latched": []}
        latched = set(s.get("latched", []))
        if "tipped" in latched or "low battery" in latched:
            if self.state != "HELP":
                self.note = "safety: " + ", ".join(sorted(latched))
                self._go("HELP", self.note)
            return self._help()
        if "bump" in latched and self.state not in ("BUMPED", "HELP"):
            self._bumped()
        if self.state in ("HELP", "DONE"):
            return self._help()
        if self.state == "BUMPED":
            return self._bump_step(p, s)
        if p.yaw is None or (s["stop"] and not latched):
            self.note = "waiting: " + (", ".join(s.get("reasons", [])) or "no IMU")
            if self.state in ("DRIVE", "CROSS", "BACK", "APPROACH", "TURN"):
                self._go(self._home(), "sensors paused")
            return Command(head=self._head_hold(p))
        self.note = ""
        fn = getattr(self, "_" + self.state.lower())
        return fn(p)

    def _head_hold(self, p):
        return (p.head[0], config.NAV_TILT_DEG)

    # START
    def _start(self, p):
        if self.state_t is None:
            self.state_t = p.t
        if self.mode == "follow":                   # the head looks at the person, not at the floor: no floor needed
            self._go("FOLLOW", "following")
            return self._follow(p)
        if self._view is not None and self._view.known:
            self._go("SCAN", "floor learnt")
            return Command()
        gap = self.sonar_gap(p.t)
        if p.t - self.state_t > 10.0:
            self.note = ("floor not learnt yet: put me on the floor with ~50 cm free in front"
                         if gap < 45 else "floor not learnt yet: no camera picture?")
        return Command(head=(0.0, config.NAV_TILT_DEG))

    # SCAN
    def _scan(self, p):
        pans = self._scan_order
        if self._scan_i >= len(pans):
            return self._choose(p)
        target = pans[self._scan_i]
        still = abs(p.head[0] - target) < 2.0 and abs(p.head[1] - config.NAV_TILT_DEG) < 2.0
        if still:
            if self._scan_since is None:
                self._scan_since = p.t
            if p.t - self._scan_since >= config.NAV_SCAN_SETTLE_S + config.NAV_SCAN_LOOK_S:
                self._scan_i += 1
                self._scan_since = None
        else:
            self._scan_since = None
        if self.mode == "people" and self._person_target(p):
            return Command(head=(p.head[0], config.NAV_TILT_DEG))
        return Command(head=(target, config.NAV_TILT_DEG))

    def _scored(self, rels, t, max_age, turn_cost=0.15):
        ways = self.map.ways(rels, t, max_age)
        cap = 120.0
        score = np.minimum(ways, cap) - turn_cost * np.abs(rels)
        for i, r in enumerate(rels):
            n = self.map.visited(r)
            score[i] += 25.0 if n == 0 else -8.0 * min(n, 4)
        return ways, score

    def _choose(self, p):
        scan_age = (p.t - self._scan_t0) + 1.0 if self._scan_t0 is not None else config.NAV_SEEN_S
        if self.mode == "people" and self._person_target(p):
            return Command(head=(p.head[0], config.NAV_TILT_DEG))
        ways, score = self._scored(CANDIDATES, p.t, scan_age)
        small = np.abs(CANDIDATES) <= 6.0           # driven without turning first: must be free straight ahead too
        straight = self.map.way(0.0, p.t, scan_age)
        ways[small] = np.minimum(ways[small], straight)
        ok = ways >= config.NAV_GO_CM
        if not ok.any():                            # a tight spot: a short way out is better than none
            ok = ways >= config.NAV_STOP_CM + 10.0
        if ok.any():
            i = int(np.argmax(np.where(ok, score, -1e9)))
            rel = float(CANDIDATES[i])
            self._failed_scans = 0
            self._event(f"chose {rel:+.0f} deg, {ways[i]:.0f} cm free")
            if abs(rel) > 6.0:
                return self._turn_to(p, p.yaw + rel, "DRIVE")
            self._target_yaw = p.yaw
            self._go("DRIVE", f"{ways[i]:.0f} cm free ahead")
            return Command(head=(0.0, config.NAV_TILT_DEG))
        crossing = self._edge_way_out(p)
        if crossing is not None:
            return crossing
        self._failed_scans += 1
        if self._failed_scans >= config.NAV_STUCK_SCANS:
            self.note = "no way out after looking around %d times" % self._failed_scans
            self._go("HELP", self.note)
            return self._help()
        # nothing seen good: turn round toward the more open side and look again - always the same way round
        # while it keeps failing, so the looks cover the whole circle
        if self._failed_scans == 1 or not self._round_sign:
            left = float(np.sum(ways[CANDIDATES > 0]))
            right = float(np.sum(ways[CANDIDATES < 0]))
            self._round_sign = 1.0 if left >= right else -1.0
        rel = 135.0 * self._round_sign
        self._event(f"no way ahead (best {ways.max():.0f} cm): turning {rel:+.0f}")
        return self._turn_to(p, p.yaw + rel, "SCAN")

    # TURN
    def _turn_to(self, p, target_yaw, after):
        gap = self.map.spin_gap()
        if gap < config.NAV_SPIN_MARGIN_CM:
            back = min(config.NAV_BACK_CM, self._fwd_since_turn)
            if back >= 3.0:
                self._back_left, self._after_back = back, self._home()
                self._go("BACK", f"too close to turn ({gap:.0f} cm)")
                return Command(v=config.NAV_V_BACK, head=(0.0, config.NAV_TILT_DEG))
            if gap < 0:
                self.note = "boxed in: something inside the spin circle and no room to back up"
                self._go("HELP", self.note)
                return self._help()
        self._target_yaw, self._after_turn = target_yaw, after
        self._turn_sign = 1.0 if target_yaw > p.yaw else -1.0
        self._turn_size = "small" if abs(target_yaw - p.yaw) < config.NAV_TURN_SMALL_DEG else "big"
        self._turn_t0 = p.t
        self._settle_t = None
        self._go("TURN", f"{target_yaw - p.yaw:+.0f} deg")
        self.stats["turns"] += 1
        return self._turn(p)

    def _turn(self, p):
        left = (self._target_yaw - p.yaw) * self._turn_sign
        rate = abs(p.yaw_rate)
        look = max(-60.0, min(60.0, self._target_yaw - p.yaw))
        if self._settle_t is not None:
            if rate < 5.0 or p.t - self._settle_t > 1.5:
                over = -left                                # + = turned too far
                self._learn_lead(self._turn_size, over, self._turn_rate_stop)
                self._settle_t = None
                self._fwd_since_turn = 0.0
                self._go(self._after_turn, f"turned, {over:+.1f} deg over")
                if self.state == "DRIVE":
                    self._target_yaw = p.yaw + (-over if abs(over) < 15 else 0.0)
            return Command(head=(0.0, self._tilt()))
        if left <= rate * self._lead[self._turn_size] or p.t - self._turn_t0 > 3.0:
            self._turn_rate_stop = max(rate, 50.0)
            self._settle_t = p.t
            return Command(head=(look, self._tilt()))
        return Command(turn=self._turn_sign * config.NAV_TURN_DPS, head=(look, self._tilt()))

    def _tilt(self):
        """The head's tilt while turning: down at the floor, or (following) up at the person."""
        return config.FOLLOW_TILT_DEG if self.mode == "follow" else config.NAV_TILT_DEG

    def _learn_lead(self, size, over, rate_stop):
        """How early to stop the spin, learnt from how far the last turn of that size went over (+) or short (-).
        2026-11-08: one shared lead learnt from small turns (+10..+32 deg over: they stop before full speed) grew to
        0.4 s, then every 135 deg turn stopped 52-76 deg short - and errors over 40 deg were not learnt from at all, so
        it never came back: turned toward the same wall 4 times, HELP. Now small and big turns learn apart, and a big
        miss counts as 40 deg."""
        over = max(-40.0, min(40.0, over))
        self._lead[size] = max(0.0, min(0.4, self._lead[size] + 0.5 * over / max(rate_stop, 50.0)))

    # DRIVE
    def _seen_s(self, p):
        """How old a picture may be to count as "seen free": NAV_SEEN_S, but just after a look around the head is
        still swinging back - the look around's pictures count for the first moment of the drive."""
        if self._scan_t0 is not None and self._leg_t0 is not None and p.t - self._leg_t0 < 1.0:
            return max(config.NAV_SEEN_S, p.t - self._scan_t0 + 0.5)
        return config.NAV_SEEN_S

    def _ahead(self, p, rel=0.0):
        """Free cm ahead now: the memory (obstacles + seen ground) along the heading (or rel deg off it), the
        sonar's echo."""
        return min(self.map.way(rel, p.t, self._seen_s(p)), self.sonar_gap(p.t))

    def _drive(self, p):
        if self.mode == "people" and self._person_target(p):
            return Command(head=(p.head[0], config.NAV_TILT_DEG))
        err = self._target_yaw - p.yaw
        if p.t - self._replan_t >= config.NAV_REPLAN_S:
            self._replan_t = p.t
            rels = np.arange(-config.NAV_REPLAN_DEG, config.NAV_REPLAN_DEG + 0.1, 5.0) + err
            ways, score = self._scored(rels, p.t, self._seen_s(p), turn_cost=0.3)
            ok = ways >= config.NAV_STOP_CM + 10
            if ok.any():
                i = int(np.argmax(np.where(ok, score, -1e9)))
                self._target_yaw = p.yaw + float(rels[i])
                err = float(rels[i])
        ahead = self._ahead(p)
        v = self._view
        if (v is not None and v.edge and p.t - v.t < 0.5 and v.edge_cm is not None
                and self._edge_sonar_ok(p.t, v.edge_cm - (config.NAV_FRONT_CM - v.ox)) and v.edge_cm - (config.NAV_FRONT_CM - v.ox) < config.NAV_SLOW_CM
                and self._clear_to_edge(v.edge_cm - (config.NAV_FRONT_CM - v.ox))):
            return self._start_cross(v.edge_cm, v.edge_cm - (config.NAV_FRONT_CM - v.ox))
        if ahead < config.NAV_STOP_CM:
            self._go("SCAN", f"{ahead:.0f} cm ahead")
            return Command(head=(p.head[0], config.NAV_TILT_DEG))
        if p.t - self._leg_t0 > config.NAV_LEG_S:
            self._go("SCAN", "look around")
            return Command(head=(p.head[0], config.NAV_TILT_DEG))
        if self._stuck(p):
            return self._unstick(p)
        speed = config.NAV_V_CRUISE if ahead > config.NAV_SLOW_CM else config.NAV_V_CRAWL
        turn = max(-config.NAV_STEER_DPS, min(config.NAV_STEER_DPS, config.NAV_STEER_K * err))
        sweep = 15.0 * math.sin(2 * math.pi * (p.t - self._leg_t0) / 4.0)
        pan = max(-40.0, min(40.0, 0.8 * err + sweep))
        return Command(v=speed, turn=turn, head=(pan, config.NAV_TILT_DEG))

    def _stuck(self, p):
        h = self._sonar_hist
        if not h or p.t - self.state_t < config.NAV_STUCK_S + 0.3:
            return False
        old = [s for s in h if p.t - s[0] >= config.NAV_STUCK_S]
        if not old or h[-1][1] >= 150.0 or old[-1][1] >= 150.0:
            return False
        win = h[h.index(old[-1]):]
        # one steady target (no jump from one echo to another) and a steady heading: else it proves nothing
        if any(abs(b[1] - a[1]) > 6.0 for a, b in zip(win, win[1:])):
            return False
        if None in (win[0][2], win[-1][2]) or abs(win[-1][2] - win[0][2]) > 8.0:
            return False
        expect = self._speed * (win[-1][0] - win[0][0])
        return expect >= 5 * config.NAV_STUCK_CM and win[0][1] - win[-1][1] < config.NAV_STUCK_CM

    def _unstick(self, p):
        self.stats["stuck"] += 1
        self.map.add_bump(p.t)
        back = min(config.NAV_BACK_CM, self._fwd_since_turn)
        self._event("stuck: the sonar distance does not shrink")
        self._back_left, self._after_back = max(back, 3.0), self._home()
        self._go("BACK", "stuck")
        return Command(v=config.NAV_V_BACK, head=(0.0, config.NAV_TILT_DEG))

    # CROSS
    def _start_cross(self, edge_cm, edge_ahead):
        self._cross_left = edge_cm + 25.0
        self._cross_edge = edge_ahead                   # cm ahead of the front
        self._go("CROSS", f"floor edge {edge_cm:.0f} cm ahead, sonar free")
        self.stats["crossed"] += 1
        return Command(v=config.NAV_V_CRAWL, learn_floor=True, head=(0.0, config.NAV_TILT_DEG))

    def _edge_way_out(self, p):
        """No way found, but this look saw a floor edge straight ahead and the sonar is still free: the floor beyond
        looks new, not blocked - cross it (2026-11-04: stopped at a rug's edge, the parquet beyond all "obstacles",
        turned round 3 times and asked for help)."""
        if self._scan_edge is None:
            return None
        edge_cm, edge_ahead = self._scan_edge
        if not self._edge_sonar_ok(p.t, edge_ahead):
            return None
        if not self._clear_to_edge(edge_ahead):
            return None
        self._failed_scans = 0
        return self._start_cross(edge_cm, edge_ahead)

    def _cross_band(self, edge_ahead):
        band, half_w = config.NAV_CROSS_BAND_CM, config.NAV_HALF_WIDTH_CM + config.NAV_MARGIN_CM + 5.0
        self.map.forget_band(edge_ahead - band, edge_ahead + band, half_w)

    def _clear_to_edge(self, edge_ahead):
        """Nothing remembered between the front and the edge (the edge itself does not count): else no crossing.
        2026-11-04: a crossing started with a camera point 1 cm ahead (the unlearnt strip at the picture's bottom)
        and stopped at once."""
        self._cross_band(edge_ahead)
        return self.map.obstacle_gap(0.0) >= edge_ahead - config.NAV_CROSS_BAND_CM

    def _cross(self, p):
        # the edge itself is not in the way: pictures of it that failed the edge test (closer, at a slant) left it
        # in the memory as obstacles; things more than NAV_CROSS_BAND_CM past it still stop the crossing
        self._cross_band(self._cross_edge)
        gap = min(self.sonar_gap(p.t), self.map.obstacle_gap(0.0))
        if gap < config.NAV_STOP_CM:
            self._go("SCAN", f"{gap:.0f} cm ahead while crossing")
            return Command(head=(0.0, config.NAV_TILT_DEG))
        self._cross_left -= abs(self._speed) * self._dt
        self._cross_edge -= abs(self._speed) * self._dt
        if self._cross_left <= 0 or p.t - self.state_t > 6.0:
            self._go("SCAN", "crossed")
            return Command(head=(0.0, config.NAV_TILT_DEG))
        return Command(v=config.NAV_V_CRAWL, learn_floor=True, head=(0.0, config.NAV_TILT_DEG))

    # BACK
    def _back(self, p):
        self._back_left -= abs(self._speed) * self._dt
        if self._back_left <= 0 or p.t - self.state_t > 3.0:
            self._fwd_since_turn = 0.0
            self._go(self._after_back, "backed up")
            return Command(head=(0.0, config.NAV_TILT_DEG))
        return Command(v=config.NAV_V_BACK, head=(0.0, config.NAV_TILT_DEG))

    # BUMPED
    def _bumped(self):
        self.stats["bumps"] += 1
        self.map.add_bump(self.t)
        self._bump_resets = [t for t in self._bump_resets if self.t - t < config.NAV_BUMP_WINDOW_S]
        self._reset_sent_t = None
        self._go("BUMPED", "the IMU felt a hit")

    def _bump_step(self, p, s):
        latched = set(s.get("latched", []))
        if "bump" in latched:
            if len(self._bump_resets) >= config.NAV_BUMP_RESETS:
                self.note = f"{len(self._bump_resets) + 1} bumps in {config.NAV_BUMP_WINDOW_S / 60:.0f} min"
                self._go("HELP", self.note)
                return self._help()
            if p.t - self.state_t >= 1.0 and (self._reset_sent_t is None or p.t - self._reset_sent_t > 1.0):
                self._reset_sent_t = p.t
                self._bump_resets.append(p.t)
                return Command(reset_safety=True)
            return Command()
        back = min(config.NAV_BACK_CM, max(self._fwd_since_turn, 5.0))
        self._back_left, self._after_back = back, self._home()
        self._go("BACK", "after the bump")
        return Command(v=config.NAV_V_BACK)

    # HELP
    def _help(self):
        return Command(head=(0.0, 10.0))

    # ---------- people ----------
    def _see_people(self, p):
        if p.yaw is None:
            return
        best = None
        for person in p.people:
            if person.score < config.NAV_PERSON_SCORE:
                continue
            ob = p.yaw + person.bearing
            if any(abs(_wrap(ob - b)) < 25.0 and p.t < until for b, until in self._cooldown):
                continue
            if best is None or (person.dist or 999) < (best[1] or 999):
                best = (ob, person.dist, p.t)
        if best is not None:
            # the feet distance jumps +-10 % picture to picture: the median of the last few of the same person
            same = self._person is not None and abs(_wrap(best[0] - self._person[0])) < 20.0
            self._dists = (self._dists if same else [])[-4:] + [best[1]]
            known = [d for d in self._dists if d is not None]
            self._person = (best[0], float(np.median(known)) if known else None, best[2])

    def _person_target(self, p):
        """People mode: a person seen recently and not just greeted -> APPROACH. True if it switched."""
        if self._person is None or p.t - self._person[2] > config.NAV_PERSON_LOST_S:
            return False
        self._go("APPROACH", "a person %+.0f deg" % _wrap(self._person[0] - p.yaw))
        return True

    def _approach(self, p):
        if self._person is None or p.t - self._person[2] > config.NAV_PERSON_LOST_S:
            self._person = None
            self._go("SCAN", "lost the person")
            return Command(head=(0.0, config.NAV_TILT_DEG))
        ob, dist, _ = self._person
        rel = _wrap(ob - p.yaw)
        look = (max(-60.0, min(60.0, rel)), config.NAV_TILT_DEG)
        gap_person = None if dist is None else dist - config.NAV_FRONT_CM
        ahead = self._ahead(p, rel if abs(rel) < 10 else 0.0)
        if gap_person is not None and gap_person <= config.NAV_PERSON_STOP_CM:
            self._go("GREET", "reached the person")
            self.stats["greeted"] += 1
            return Command(head=(look[0], 15.0))
        if abs(rel) > 25.0:
            return self._turn_to(p, p.yaw + rel, "APPROACH")
        if ahead < config.NAV_STOP_CM:
            self._go("SCAN", f"way to the person blocked ({ahead:.0f} cm)")
            return Command(head=look)
        if self._stuck(p):
            return self._unstick(p)
        turn = max(-config.NAV_STEER_DPS, min(config.NAV_STEER_DPS, config.NAV_STEER_K * rel))
        return Command(v=config.NAV_V_CRAWL, turn=turn, head=look)

    def _greet(self, p):
        rel = 0.0 if self._person is None else _wrap(self._person[0] - p.yaw)
        if p.t - self._greet_t0 >= config.NAV_PERSON_GREET_S:
            if self._person is not None:
                self._cooldown.append((self._person[0], p.t + config.NAV_PERSON_COOLDOWN_S))
            self._cooldown = [c for c in self._cooldown if c[1] > p.t]
            self._person = None
            self._go("SCAN", "greeted")
            return Command(head=(0.0, config.NAV_TILT_DEG))
        return Command(head=(max(-60.0, min(60.0, rel)), 15.0))


    # ---------- follow (alive.py: "gel gel") ----------
    def _follow(self, p):
        tg = self._target
        if tg is None and p.t - (self.state_t if self.state_t is not None else p.t) < config.FOLLOW_LOST_S:
            return Command(head=(0.0, config.FOLLOW_TILT_DEG))     # not seen yet: a moment to find them
        if tg is None or p.t - tg[2] > config.FOLLOW_LOST_S:
            self._waiting = False
            self._go("LOST", "not seen for %.0f s" % config.FOLLOW_LOST_S if tg is not None else "not seen yet")
            return self._lost(p)
        ob, dist, _, tilt = tg
        rel = _wrap(ob - p.yaw)
        look = (max(-60.0, min(60.0, rel)), max(-10.0, min(35.0, config.FOLLOW_TILT_DEG if tilt is None else tilt)))
        gap_person = None if dist is None else dist - config.NAV_FRONT_CM
        ahead_now = min(self.map.obstacle_gap(0.0), self.sonar_gap(p.t))     # sonar + memory (+ bumps): no floor check
        near = (gap_person is not None and gap_person <= config.FOLLOW_STOP_CM) or ahead_now <= config.FOLLOW_STOP_CM
        if self._waiting:
            go = config.FOLLOW_STOP_CM + config.FOLLOW_GO_CM
            if (gap_person is None or gap_person > go) and ahead_now > go:
                self._waiting = False
                self._event("they walked on - following")
            elif abs(rel) > config.FOLLOW_TURN_DEG + 15.0:     # standing: keep facing them (no turns for small steps)
                return self._turn_to(p, p.yaw + rel, "FOLLOW")
            else:
                return Command(head=look)
        elif near:
            self._waiting = True
            self._event("near them (%s) - standing" % ("%.0f cm" % gap_person if gap_person is not None else
                                                       "%.0f cm ahead" % ahead_now))
            return Command(head=look)
        if abs(rel) > config.FOLLOW_TURN_DEG:
            return self._turn_to(p, p.yaw + rel, "FOLLOW")
        if self._stuck(p):
            return self._unstick(p)
        turn = max(-config.NAV_STEER_DPS, min(config.NAV_STEER_DPS, config.NAV_STEER_K * rel))
        return Command(v=config.NAV_V_CRAWL, turn=turn, head=look)

    def _lost(self, p):
        tg = self._target
        if tg is not None and self._lost_t0 is not None and tg[2] > self._lost_t0:
            self._go("FOLLOW", "found them again")
            return self._follow(p)
        if p.t - self._lost_t0 >= config.FOLLOW_REFIND_S:
            self.note = "lost them"
            self._go("DONE", self.note)
            return self._help()
        pans = config.FOLLOW_LOOK_PANS
        target = pans[min(self._look_i, len(pans) - 1)]
        if abs(p.head[0] - target) < 2.0:
            if self._look_since is None:
                self._look_since = p.t
            elif p.t - self._look_since >= config.NAV_SCAN_SETTLE_S + config.NAV_SCAN_LOOK_S:
                self._look_i, self._look_since = self._look_i + 1, None
        return Command(head=(target, config.FOLLOW_TILT_DEG))

    def _done(self, p):
        return self._help()


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0
