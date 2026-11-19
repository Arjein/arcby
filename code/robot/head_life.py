"""
Head life: the pan-tilt head never freezes. A 50 Hz loop moves it toward where attention wants to look
(the focus) like a damped spring - soft and a little late, as an animal follows something - and adds small
motions on top so the robot looks alive:

  move            a big jump of the focus is a planned minimum-jerk move (smooth start and stop): quick to a face,
                  slow and lazy when only looking around
  micro-saccades  every 2-5 s a new tiny offset (<1 deg) - nobody stares perfectly still
  breathing       a very slow +-0.8 deg tilt wave
  blink           a quick ~2.5 deg dip (70 ms down, 40 ms hold, 140 ms up), sometimes twice
  listening       mode "listen" (attention: following the person who is speaking to it): small slow nods at the
                  pauses in their speech and now and then a slight sideways "head tilt" - from hearing()
  drive           mode "drive" (robot/legs.py drives: it needs the floor in the picture and the head exactly where it
                  asked): straight to the focus at NAV_HEAD_DPS, no life on top - back to normal with the next mode

    life = HeadLife(head)                   # head = robot/pan_tilt.PanTilt (or None to only compute)
    life.set_focus(20, 10, "follow")        # attention decides; modes: follow / idle / listen / drive
    life.hearing = lambda: ears.pause_s     # None = nobody speaking, else seconds of quiet in their sentence
    life.run(stop_event)                    # or life.step(t) from your own loop
    life.pose_at(t_frame)                   # where the head was told to point when a frame was taken
    life.speed_at(t0, t1)                   # how fast it moved then (fast = blurred frame, unsure pose)

Only robot/pan_tilt.py talks to the PTZ board; this module just calls head.set(). When the board stops answering
(battery switched off), it stops driving the head and probes it read-only every HEAD_PROBE_S; back -> init, go on.
"""
import bisect
import math
import random
import threading
import logging
import time
from collections import deque

from . import config

log = logging.getLogger(__name__)


def blink_offset(dt, depth=config.HEAD_BLINK_DEG):
    """Tilt offset (negative = down) dt seconds into a blink; 0 when it is over."""
    down, hold, up = 0.07, 0.04, 0.14
    if dt < 0:
        return 0.0
    if dt < down:
        return -depth * math.sin(dt / down * math.pi / 2)
    if dt < down + hold:
        return -depth
    if dt < down + hold + up:
        return -depth * (1 + math.cos((dt - down - hold) / up * math.pi)) / 2
    return 0.0


BLINK_S = 0.25


def min_jerk(x):
    """0 -> 1 with zero speed and acceleration at both ends (how people move their heads and arms)."""
    x = min(1.0, max(0.0, x))
    return x * x * x * (10 - 15 * x + 6 * x * x)


def nod_offset(dt, depth, length):
    """Tilt offset (negative = down) dt seconds into a listening nod: down and back up, soft at both ends."""
    if dt < 0 or dt >= length:
        return 0.0
    return -depth * math.sin(math.pi * dt / length) ** 2


class HeadLife:
    def __init__(self, head=None, rng=None, clock=time.monotonic, sleep=time.sleep,
                 pan_soft=config.HEAD_PAN_SOFT, tilt_soft=config.HEAD_TILT_SOFT, hearing=None):
        self.head = head
        self.hearing = hearing              # () -> None (nobody speaking) or seconds of quiet in their sentence
        self.errors, self._err_log_t = 0, -math.inf     # step() failures (logged, the loop goes on)
        self.lost_t = None                  # when the PTZ board stopped answering (battery off), or None
        self._probe_t = 0.0
        self.rng = rng or random.Random()
        self._clock, self._sleep = clock, sleep
        self.pan_soft, self.tilt_soft = pan_soft, tilt_soft
        self._lock = threading.Lock()
        self.focus = (0.0, 0.0)
        self.mode = "idle"
        self.pos = [0.0, 0.0]               # spring position (head angles, deg)
        self.vel = [0.0, 0.0]
        self.saccade = False                # a planned move is under way (no blinks, stiff spring)
        self._move = None                   # [start (pan, tilt) or None = from where it is now, t0, length s]
        self.micro = [0.0, 0.0]             # current micro-saccade offset (eased)
        self._micro_target = [0.0, 0.0]
        self._last_t = None
        self._next_micro = self._next_blink = None
        self._blinks = []                   # start times of blinks under way / queued
        self.sent = None                    # last angles sent to the head
        self.out = (0.0, 0.0)               # last computed angles (with life on top)
        self.history = deque(maxlen=int(3 * config.HEAD_RATE_HZ))   # (t, pan, tilt): 3 s
        # listening
        self._nod = None                    # (start t, depth, length) of the nod under way
        self._last_nod_t = -math.inf
        self._talk_s = 0.0                  # seconds of their speech since the last nod
        self._paused = False                # hearing() is in a pause now (a nod is due once per pause)
        self._heard_t = -math.inf           # when hearing() last said someone speaks
        self._cock = [0.0, 0.0]             # current sideways "head tilt" offset (eased)
        self._cock_target = (0.0, 0.0)
        self._cock_decided = False          # this listening turn: cock the head or not (decided once)

    # ---------- input ----------

    def set_focus(self, pan, tilt, mode="follow"):
        """Where to look (head angles; clamped to the soft range) and how: 'follow' (a face) or 'idle'."""
        pan = min(max(pan, self.pan_soft[0]), self.pan_soft[1])
        tilt = min(max(tilt, self.tilt_soft[0]), self.tilt_soft[1])
        with self._lock:
            d = math.hypot(pan - self.focus[0], tilt - self.focus[1])
            if mode == "drive":
                self._move, self.saccade = None, False      # the drive paces the head itself
            elif self.mode == "drive":
                self.vel = [0.0, 0.0]                       # out of a drive: the spring starts from rest
            if mode != "drive" and d > config.HEAD_SACCADE_DEG:
                # a jump: plan a minimum-jerk move from where the head is now - quick to a face, lazy otherwise
                dps, base = config.HEAD_MOVE_FACE if mode != "idle" else config.HEAD_MOVE_IDLE
                dist = math.hypot(pan - self.pos[0], tilt - self.pos[1])
                self._move = [None, None, base + dist / dps]
                self.saccade = True
            self.focus, self.mode = (pan, tilt), mode

    def start_at(self, pan, tilt):
        """The head is physically here now (e.g. after PanTilt.set): start the spring from it."""
        with self._lock:
            self.pos, self.vel = [pan, tilt], [0.0, 0.0]
            self.focus = (pan, tilt)
            self._move, self.saccade = None, False

    # ---------- output ----------

    def pose_at(self, t):
        """Head angles at time t (interpolated from the last 3 s; the oldest / newest outside that)."""
        with self._lock:
            h = list(self.history)
        if not h:
            return self.out
        times = [x[0] for x in h]
        i = bisect.bisect_left(times, t)
        if i <= 0:
            return h[0][1:]
        if i >= len(h):
            return h[-1][1:]
        (t0, p0, q0), (t1, p1, q1) = h[i - 1], h[i]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return p0 + (p1 - p0) * f, q0 + (q1 - q0) * f

    def speed_at(self, t0, t1):
        """Fastest head speed (deg/s) between t0 and t1, from the history (0 when it knows nothing)."""
        with self._lock:
            inside = [x for x in self.history if t0 < x[0] < t1]
        if not self.history:
            return 0.0
        pts = [(t0, *self.pose_at(t0))] + inside + [(t1, *self.pose_at(t1))]
        best = 0.0
        for (ta, pa, qa), (tb, pb, qb) in zip(pts, pts[1:]):
            if tb > ta:
                best = max(best, math.hypot(pb - pa, qb - qa) / (tb - ta))
        return best

    # ---------- the loop ----------

    def step(self, t=None):
        """Advance to time t; send the head there if it moved enough. Returns the angles."""
        t = self._clock() if t is None else t
        with self._lock:
            dt = 0.0 if self._last_t is None else min(0.1, max(0.0, t - self._last_t))
            self._last_t = t
            if self._next_micro is None:
                self._next_micro = t + self.rng.uniform(*config.HEAD_MICRO_EVERY_S)
                self._next_blink = t + self.rng.uniform(*config.HEAD_BLINK_EVERY_S)
            if self.mode == "drive":
                self._drive(dt)
                pan, tilt = self.pos
            else:
                self._spring(dt, t)
                self._life(t, dt)
                self._listen(t, dt)
                pan = self.pos[0] + self.micro[0] + self._cock[0]
                tilt = (self.pos[1] + self.micro[1] + self._cock[1] + self._breath(t) + self._blink(t)
                        + (nod_offset(t - self._nod[0], *self._nod[1:]) if self._nod else 0.0))
        if self.sent is None or math.hypot(pan - self.sent[0], tilt - self.sent[1]) >= config.HEAD_MIN_STEP_DEG:
            if self.head is None:
                self.sent = (pan, tilt)             # nothing to drive: the computed pose is the pose
            elif self._head_ok(t):
                try:
                    self.head.set(pan, tilt)
                    self.sent = (pan, tilt)
                except OSError as e:        # the battery went off: stop driving, look for it now and then
                    self.lost_t, self._probe_t = t, t + config.HEAD_PROBE_S
                    log.warning("head: the PTZ board stopped answering (%s - battery off?) - head stopped", e)
        # pose_at / speed_at tell vision where the camera really points: while the board is lost the head does not
        # move, so the last angles it really got (not the spring's) - else every face is placed with a made-up
        # pose and the moving 'head' blocks recognition (2026-11-01 review)
        real = (pan, tilt) if self.head is None or self.lost_t is None else (self.sent or (0.0, 0.0))
        with self._lock:
            self.out = real
            self.history.append((t, *real))
        return pan, tilt

    def _head_ok(self, t):
        """True unless the board was lost; then a read-only probe every HEAD_PROBE_S, and init() once it is back."""
        if self.lost_t is None:
            return True
        if t < self._probe_t:
            return False
        self._probe_t = t + config.HEAD_PROBE_S
        if not self.head.answers():
            return False
        try:
            self.head.init()
        except Exception as e:  # noqa: BLE001 - PanTiltError / OSError: still not right, try again later
            log.warning("head: the PTZ board answers but init failed (%s)", e)
            return False
        log.info("head: the PTZ board answers again (battery on) - head back after %.0f s", t - self.lost_t)
        self.lost_t, self.sent = None, None
        return True

    def run(self, stop_event):
        period = 1.0 / config.HEAD_RATE_HZ
        while not stop_event.is_set():
            t0 = self._clock()
            try:
                self.step(t0)
            except Exception:  # noqa: BLE001 - one bad step must not freeze the head for good (the thread died)
                self.errors += 1
                if t0 - self._err_log_t >= config.HEAD_ERROR_LOG_S:
                    self._err_log_t = t0
                    log.exception("head: a step failed (%d so far) - going on", self.errors)
            self._sleep(max(0.0, period - (self._clock() - t0)))

    # ---------- parts (lock held) ----------

    def _goal(self, t):
        """Where the spring pulls now: along the planned move while there is one, else the focus."""
        mv = self._move
        if mv is None:
            return self.focus
        if mv[0] is None:                       # planned in set_focus: starts here, now
            mv[0], mv[1] = tuple(self.pos), t
        x = (t - mv[1]) / mv[2]
        if x >= 1.0:
            self._move = None
            return self.focus
        s = min_jerk(x)                         # the focus may still move (a face): the end point follows it
        return (mv[0][0] + (self.focus[0] - mv[0][0]) * s, mv[0][1] + (self.focus[1] - mv[0][1]) * s)

    def _spring(self, dt, t):
        """Critically damped spring toward the goal (a planned move, or the focus), with a top speed."""
        goal = self._goal(t)
        err = [goal[0] - self.pos[0], goal[1] - self.pos[1]]
        if self.saccade and self._move is None and math.hypot(self.focus[0] - self.pos[0],
                                                              self.focus[1] - self.pos[1]) < 1.0:
            self.saccade = False
        hz = config.HEAD_SACCADE_HZ if self.saccade else (
            config.HEAD_IDLE_HZ if self.mode == "idle" else config.HEAD_FOLLOW_HZ)
        w = 2 * math.pi * hz
        vmax = config.HEAD_MAX_DPS[1 if self.saccade else 0]
        for k in range(2):
            self.vel[k] += (w * w * err[k] - 2 * w * self.vel[k]) * dt
        speed = math.hypot(*self.vel)
        if speed > vmax:
            self.vel = [v * vmax / speed for v in self.vel]
        for k in range(2):
            self.pos[k] += self.vel[k] * dt

    def _drive(self, dt):
        """Drive mode: from where the head shows now (life offsets folded in once) straight to the focus at
        NAV_HEAD_DPS, like nav.py's Head - navigation waits for the head to be within 2 deg of where it asked."""
        if any(self.micro) or any(self._cock) or self._nod or self._blinks:
            self.pos = [self.pos[0] + self.micro[0] + self._cock[0], self.pos[1] + self.micro[1] + self._cock[1]]
            self.micro, self._micro_target, self._cock = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
            self._cock_target, self._nod, self._blinks = (0.0, 0.0), None, []
        step = config.NAV_HEAD_DPS * dt
        for k in range(2):
            self.pos[k] += max(-step, min(step, self.focus[k] - self.pos[k]))
        self.vel = [0.0, 0.0]

    def _life(self, t, dt):
        if t >= self._next_micro:
            scale = 1.0 if self.mode == "idle" else 0.5
            sp, st = config.HEAD_MICRO_DEG
            self._micro_target = [self.rng.gauss(0, sp * scale), self.rng.gauss(0, st * scale)]
            self._next_micro = t + self.rng.uniform(*config.HEAD_MICRO_EVERY_S)
        ease = min(1.0, dt / config.HEAD_MICRO_EASE_S)
        for k in range(2):
            self.micro[k] += (self._micro_target[k] - self.micro[k]) * ease
        if t >= self._next_blink:
            if not self.saccade:                # never blink in the middle of a jump
                self._blinks.append(t)
                if self.rng.random() < 0.15:    # double blink
                    self._blinks.append(t + BLINK_S + 0.08)
            self._next_blink = t + self.rng.uniform(*config.HEAD_BLINK_EVERY_S)

    def _listen(self, t, dt):
        """Listening (mode "listen"): a nod at a pause in their speech - after some speech, at most every
        HEAD_NOD_EVERY_S, not always - and maybe a slight sideways head tilt held while they talk. Silent."""
        if self._nod is not None and t - self._nod[0] >= self._nod[2]:
            self._nod = None
        quiet = None
        if self.mode == "listen" and self.hearing is not None:
            try:
                quiet = self.hearing()
            except Exception:  # noqa: BLE001 - the ears are another thread's business: no nods, no crash
                quiet = None
        if quiet is not None:
            self._heard_t = t
            if quiet < config.HEAD_NOD_PAUSE_S:
                self._talk_s += dt
                self._paused = False
            elif not self._paused:                  # a pause just began: nod?
                self._paused = True
                if (self._talk_s >= config.HEAD_NOD_MIN_TALK_S and t - self._last_nod_t >= config.HEAD_NOD_EVERY_S
                        and self._nod is None and not self.saccade and self.rng.random() < config.HEAD_NOD_P):
                    self._nod = (t, self.rng.uniform(*config.HEAD_NOD_DEG), self.rng.uniform(*config.HEAD_NOD_S))
                    self._last_nod_t, self._talk_s = t, 0.0
        listening = self.mode == "listen" and t - self._heard_t < config.HEAD_LISTEN_HOLD_S
        if listening and not self._cock_decided:    # once per listening turn: cock the head a little, or not
            self._cock_decided = True
            if self.rng.random() < config.HEAD_COCK_P:
                side = self.rng.choice((-1.0, 1.0))
                self._cock_target = (side * config.HEAD_COCK_DEG[0], -config.HEAD_COCK_DEG[1])
        elif not listening:
            self._cock_decided, self._cock_target, self._talk_s, self._paused = False, (0.0, 0.0), 0.0, False
        ease = min(1.0, dt / config.HEAD_COCK_EASE_S)
        for k in range(2):
            self._cock[k] += (self._cock_target[k] - self._cock[k]) * ease

    def _breath(self, t):
        amp, period = config.HEAD_BREATH
        return amp * math.sin(2 * math.pi * t / period)

    def _blink(self, t):
        self._blinks = [b for b in self._blinks if t - b < BLINK_S]
        return min((blink_offset(t - b) for b in self._blinks), default=0.0)
