"""
A simulated room for the navigation brain (robot/nav.py) - the tests and `code/nav.py --sim`.

Top view, cm, x / y on the floor. The robot is a rectangle (robot/config.py NAV_FRONT_CM / NAV_REAR_CM /
NAV_HALF_WIDTH_CM) that moves as the motors would: the request's speed is NOT what nav.py believes (gain 0.8-1.2,
first-order lag), spins are at least 250 deg/s and coast. What it senses:

  sonar   a 24 deg cone 7 cm above the floor: only things taller than 5 cm, walls hit at more than 40 deg from
          square send no echo back (they bounce away), legs thinner than 2 cm answer only near the beam's middle,
          3 % of pings come back empty
  camera  the real robot/floor.py ray code on an ideal "floor / not floor" picture (every obstacle's foot, cables,
          rugs the robot has not learnt), seen through the real ground geometry of the head's pan and tilt; 1 % of
          cells wrongly "not floor" (shadows, specks)
  people  two legs (6 cm) - seen as a body by the camera (bearing, feet distance +-10 %, 10 % missed)
  bump    touching anything tall -> the robot stops there and safety latches "bump" (as the IMU would)

Contacts are counted: World.collisions must stay 0 - the point of navigation. Cables are counted apart
(World.cable_touches): the real robot drives over a cable, but it should not.
"""
import math
import random
from dataclasses import dataclass

import numpy as np

from robot import config
from robot.floor import FloorSeer
from robot.nav import Percept, Person, speed_cm_s

SONAR_H = 7.0


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    h: float = 50.0
    kind: str = "box"

    def inside(self, x, y, pad=0.0):
        return (x >= self.x0 - pad) & (x <= self.x1 + pad) & (y >= self.y0 - pad) & (y <= self.y1 + pad)


@dataclass
class Circle:
    x: float
    y: float
    r: float
    h: float = 45.0
    kind: str = "leg"

    def inside(self, x, y, pad=0.0):
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= (self.r + pad) ** 2


@dataclass
class Cable:
    pts: list               # [(x, y), ...] a polyline
    w: float = 1.0
    h: float = 1.0
    kind: str = "cable"

    def inside(self, x, y, pad=0.0):
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = np.zeros(np.broadcast(x, y).shape, bool)
        for (ax, ay), (bx, by) in zip(self.pts, self.pts[1:]):
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy or 1.0
            u = np.clip(((x - ax) * dx + (y - ay) * dy) / L2, 0, 1)
            out |= (x - ax - u * dx) ** 2 + (y - ay - u * dy) ** 2 <= (self.w / 2 + pad) ** 2
        return out


@dataclass
class Rug:
    x0: float
    y0: float
    x1: float
    y1: float
    h: float = 0.5
    kind: str = "rug"

    def inside(self, x, y, pad=0.0):
        return (x >= self.x0 - pad) & (x <= self.x1 + pad) & (y >= self.y0 - pad) & (y <= self.y1 + pad)


class Walker:
    """A person: two legs, walking slowly between random spots (or standing)."""

    def __init__(self, x, y, rnd, walks=False, W=400, H=300):
        self.x, self.y, self.rnd, self.walks, self.W, self.H = x, y, rnd, walks, W, H
        self.goal = (x, y)
        self.pause = 0.0

    def legs(self):
        return [Circle(self.x - 10, self.y, 6.0, 80, "person"), Circle(self.x + 10, self.y, 6.0, 80, "person")]

    def step(self, dt, robot_xy=None):
        if not self.walks:
            return
        if robot_xy is not None and math.hypot(robot_xy[0] - self.x, robot_xy[1] - self.y) < 45.0:
            gx, gy = self.goal                      # people step round a robot at their feet: wait / turn away
            if math.hypot(gx - robot_xy[0], gy - robot_xy[1]) < math.hypot(gx - self.x, gy - self.y):
                self.goal = (self.x + (self.x - robot_xy[0]) * 2, self.y + (self.y - robot_xy[1]) * 2)
                self.goal = (min(self.W - 40, max(40, self.goal[0])), min(self.H - 40, max(40, self.goal[1])))
        if self.pause > 0:
            self.pause -= dt
            return
        gx, gy = self.goal
        d = math.hypot(gx - self.x, gy - self.y)
        if d < 5:
            self.goal = (self.rnd.uniform(40, self.W - 40), self.rnd.uniform(40, self.H - 40))
            self.pause = self.rnd.uniform(3, 10)
            return
        s = min(d, 40.0 * dt)
        self.x += (gx - self.x) / d * s
        self.y += (gy - self.y) / d * s


def robot_outline(x, y, h, pad=0.0, step=1.5):
    """Points along the robot's rectangle (world cm) - for contact checks."""
    f, r, w = config.NAV_FRONT_CM + pad, config.NAV_REAR_CM + pad, config.NAV_HALF_WIDTH_CM + pad
    pts = []
    for a, b in (((f, w), (f, -w)), ((f, -w), (-r, -w)), ((-r, -w), (-r, w)), ((-r, w), (f, w))):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for i in range(n):
            u = i / n
            pts.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
    pts.append((0.0, 0.0))
    pts = np.array(pts)
    c, s = math.cos(math.radians(h)), math.sin(math.radians(h))
    return x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c


class World:
    DT = 0.01

    def __init__(self, W=400.0, H=300.0, things=(), start=(200.0, 150.0, 0.0), seed=0, gain=1.0, people=()):
        self.W, self.H = W, H
        self.things = list(things)
        self.people = list(people)
        self.rnd = random.Random(seed)
        self.x, self.y, self.h = start
        self.gain = gain
        self.v = 0.0            # cm/s now
        self.rate = 0.0         # deg/s now
        self.t = 0.0
        self.head = [0.0, 0.0]
        self.head_target = (0.0, 0.0)
        self.head_speed = 0.0
        self.latched = set()
        self.collisions = 0
        self.cable_touches = 0
        self.walked_into = 0
        self.learnt_rugs = set()
        self._touching = False
        self._on_cable = False
        self.cmd = (0.0, 0.0)
        self.yaw_drift = self.rnd.uniform(-0.5, 0.5) / 60.0     # deg/s
        self.yaw0 = self.h
        self.seer = FloorSeer()
        self.seer.chroma = np.ones((2, 2))              # "learnt" - the sim decides floor / not floor itself
        self.seer.bright = np.ones(2)
        self._next_sonar = self._next_cam = self._next_body = 0.0
        self.sonar = None
        self.sonar_t = None
        self.view = None
        self.people_seen = []
        self.people_t = None
        self.path = [(self.x, self.y)]
        self.driven = 0.0
        self._learn_floor = False

    # ---- world ----
    def obstacles(self):
        legs = [leg for p in self.people for leg in p.legs()]
        return [o for o in self.things if o.kind != "rug"] + legs

    def _walls_hit(self, xs, ys):
        return (xs <= 0) | (xs >= self.W) | (ys <= 0) | (ys >= self.H)

    def _contact(self, x, y, h):
        xs, ys = robot_outline(x, y, h)
        if self._walls_hit(xs, ys).any():
            return "wall"
        for o in self.obstacles():
            if o.kind in ("cable",):
                continue
            if o.inside(xs, ys).any():
                return o.kind
        return None

    def _on_cable_now(self, x, y, h):
        xs, ys = robot_outline(x, y, h)
        return any(o.inside(xs, ys).any() for o in self.things if o.kind == "cable")

    # ---- motion ----
    def apply(self, cmd):
        self.cmd = (cmd.v, cmd.turn)
        self.head_target = cmd.head
        self._learn_floor = cmd.learn_floor
        if cmd.reset_safety:
            self.latched.discard("bump")

    def advance(self, seconds):
        end = self.t + seconds
        while self.t < end - 1e-9:
            self._physics(self.DT)
            self.t += self.DT
            for p in self.people:
                p.step(self.DT, (self.x, self.y))
            self._sense()

    def _physics(self, dt):
        v_req, turn = self.cmd
        if self.latched:
            v_req, turn = 0.0, 0.0
        v_goal = speed_cm_s(v_req) * self.gain
        if abs(v_req) < 0.005 and abs(turn) > 1:
            rate_goal = math.copysign(max(abs(turn), config.SPIN_MIN_DPS) * (1.05 + 0.1 * self.rnd.random()), turn)
            tau = 0.06
        else:
            rate_goal, tau = turn, 0.15
        self.v += (v_goal - self.v) * min(1.0, dt / 0.15)
        self.rate += (rate_goal - self.rate) * min(1.0, dt / tau)
        nh = self.h + self.rate * dt
        mid = math.radians((self.h + nh) / 2)
        nx, ny = self.x + self.v * dt * math.cos(mid), self.y + self.v * dt * math.sin(mid)
        hit = self._contact(nx, ny, nh)
        if hit == "person" and self._contact(self.x, self.y, self.h) == "person":
            hit = None                              # they walked into the robot where it stood - let it be
            self.walked_into += 1
        if hit:
            if not self._touching:
                self.collisions += 1
                self._touching = True
                self.last_hit = (self.t, hit, round(self.x), round(self.y))
            self.latched.add("bump")
            self.v = self.rate = 0.0
            return
        self._touching = False
        cable = self._on_cable_now(nx, ny, nh)
        if cable and not self._on_cable:
            self.cable_touches += 1
        self._on_cable = cable
        self.driven += math.hypot(nx - self.x, ny - self.y)
        self.x, self.y, self.h = nx, ny, nh
        if not self.path or math.hypot(self.x - self.path[-1][0], self.y - self.path[-1][1]) > 5:
            self.path.append((self.x, self.y))
        # the head: to its target at NAV_HEAD_DPS
        moved = 0.0
        for i in (0, 1):
            d = self.head_target[i] - self.head[i]
            s = max(-config.NAV_HEAD_DPS * dt, min(config.NAV_HEAD_DPS * dt, d))
            self.head[i] += s
            moved = max(moved, abs(s) / dt)
        self.head_speed = moved
        if self._learn_floor:
            ax, ay = self._ahead(config.NAV_FRONT_CM + 15)
            for i, o in enumerate(self.things):
                if o.kind == "rug" and o.inside(ax, ay):
                    self.learnt_rugs.add(i)

    def _ahead(self, d):
        a = math.radians(self.h)
        return self.x + d * math.cos(a), self.y + d * math.sin(a)

    # ---- senses ----
    def _sense(self):
        if self.t >= self._next_sonar:
            self._next_sonar = self.t + 1.0 / config.NAV_SONAR_HZ
            self.sonar, self.sonar_t = self._ping(), self.t
        if self.t >= self._next_cam:
            self._next_cam = self.t + 1.0 / config.CAMERA_FPS
            if self.head_speed < 40.0 and abs(self.rate) < 20.0:
                self.view = self._picture()
        if self.t >= self._next_body:
            self._next_body = self.t + 0.2
            self.people_seen, self.people_t = self._bodies(), self.t

    def _ping(self):
        if self.rnd.random() < 0.03:
            return None
        a0 = math.radians(self.h)
        sx = self.x + config.NAV_SONAR_X_CM * math.cos(a0)
        sy = self.y + config.NAV_SONAR_X_CM * math.sin(a0)
        best = None
        half = config.NAV_SONAR_HALF_DEG
        for off in np.linspace(-half, half, 9):
            a = math.radians(self.h + off)
            d = self._ray(sx, sy, a, abs(off))
            if d is not None and (best is None or d < best):
                best = d
        if best is None or best > config.NAV_SONAR_MAX_CM + 50:
            return None
        return round(best + self.rnd.gauss(0, 0.7), 1)

    def _ray(self, sx, sy, a, off_deg):
        """Distance along a sonar ray to the first thing that echoes, or None."""
        ca, sa = math.cos(a), math.sin(a)
        d = np.arange(2.0, 400.0, 1.0)
        xs, ys = sx + d * ca, sy + d * sa
        hit = np.nonzero(self._walls_hit(xs, ys))[0]
        far = d[hit[0]] if len(hit) else None
        if far is not None:
            # wall: specular - the echo comes back only within 40 deg of square
            x, y = xs[hit[0]], ys[hit[0]]
            normal = (1, 0) if x <= 0 or x >= self.W else (0, 1)
            inc = math.degrees(math.acos(min(1.0, abs(ca * normal[0] + sa * normal[1]))))
            wall = far if inc <= 40 else None
        else:
            wall = None
        best = wall
        for o in self.obstacles():
            if o.h < SONAR_H - 2:
                continue
            if isinstance(o, Circle) and o.r < 2.0 and off_deg > 5:
                continue
            inside = np.nonzero(o.inside(xs, ys))[0]
            if len(inside) and (far is None or d[inside[0]] < far):
                dd = d[inside[0]]
                if best is None or dd < best:
                    best = dd
        return best

    def _picture(self):
        seer = self.seer
        pan, tilt = self.head
        gx, gy, ok = seer._geometry(pan, tilt)
        ok = ok & (gx > config.NAV_FRONT_CM + 1.0)
        c, s = math.cos(math.radians(self.h)), math.sin(math.radians(self.h))
        cols = seer.cols
        wx = self.x + gx[:, cols] * c - gy[:, cols] * s
        wy = self.y + gx[:, cols] * s + gy[:, cols] * c
        bad = self._walls_hit(wx, wy)
        for i, o in enumerate(self.things + [leg for p in self.people for leg in p.legs()]):
            if o.kind == "rug":
                if i in self.learnt_rugs:
                    continue
                inside = o.inside(wx, wy)
                here = o.inside(self.x, self.y)
                bad |= ~inside if here else inside          # standing on an unlearnt rug: the floor around is odd
            else:
                bad |= o.inside(wx, wy, pad=0.5)
        bad |= np.array([[self.rnd.random() < 0.01 for _ in range(bad.shape[1])] for _ in range(bad.shape[0])])
        floor = np.ones(gx.shape, bool)
        floor[:, cols] = ~bad
        return seer._rays(self.t, pan, tilt, floor, gx, gy, ok, False)

    def _bodies(self):
        out = []
        lens_x = self.x + config.NAV_CAM_X_CM * math.cos(math.radians(self.h))
        lens_y = self.y + config.NAV_CAM_X_CM * math.sin(math.radians(self.h))
        for p in self.people:
            d = math.hypot(p.x - lens_x, p.y - lens_y)
            b = math.degrees(math.atan2(p.y - self.y, p.x - self.x)) - self.h
            b = (b + 180) % 360 - 180
            if abs(b - self.head[0]) > 48 or d > 400 or self.rnd.random() < 0.1:
                continue
            dist = math.hypot(p.x - self.x, p.y - self.y) - 6.0
            out.append(Person(b + self.rnd.gauss(0, 1.5), dist * (1 + self.rnd.gauss(0, 0.05)) if d < 300 else None,
                              0.8))
        return out

    def percept(self):
        yaw = self.h - self.yaw0 + self.yaw_drift * self.t
        return Percept(t=self.t, yaw=yaw, yaw_rate=self.rate, sonar=self.sonar, sonar_t=self.sonar_t,
                       floor=self.view, people=list(self.people_seen), people_t=self.people_t,
                       safety={"stop": bool(self.latched), "reasons": sorted(self.latched),
                               "latched": sorted(self.latched)},
                       head=(self.head[0], self.head[1]))


def run(nav, world, seconds, rate=None, on_step=None):
    """Drive the world with the navigator for `seconds` of sim time."""
    rate = rate or config.NAV_RATE_HZ
    dt = 1.0 / rate
    steps = int(seconds * rate)
    for _ in range(steps):
        p = world.percept()
        if p.floor is not None:
            # each picture is fresh to the navigator once; a copy, because nav may change its edge flag
            p.floor = _copy_view(p.floor)
        cmd = nav.step(p)
        world.apply(cmd)
        if on_step:
            on_step(p, cmd)
        world.advance(dt)
    return world


def _copy_view(v):
    from dataclasses import replace
    return replace(v)


def random_room(seed, people=0, walking=False, rugs=True, cables=True):
    """A random living room: furniture boxes, chairs (4 thin legs), cables, a rug, maybe people."""
    rnd = random.Random(seed)
    W, H = rnd.uniform(300, 450), rnd.uniform(250, 350)
    start = (W / 2 + rnd.uniform(-30, 30), H / 2 + rnd.uniform(-30, 30), rnd.uniform(-180, 180))
    things = []

    def clear_of_start(x, y, r):
        return math.hypot(x - start[0], y - start[1]) > r + 55

    tries = 0
    while len([t for t in things if t.kind == "box"]) < rnd.randint(3, 6) and tries < 200:
        tries += 1
        w, h = rnd.uniform(25, 90), rnd.uniform(25, 70)
        x0, y0 = rnd.uniform(0, W - w), rnd.uniform(0, H - h)
        if clear_of_start(x0 + w / 2, y0 + h / 2, max(w, h) * 0.75):
            things.append(Box(x0, y0, x0 + w, y0 + h, rnd.choice([12.0, 40.0, 80.0])))
    for _ in range(rnd.randint(1, 2)):
        cx, cy = rnd.uniform(40, W - 40), rnd.uniform(40, H - 40)
        if clear_of_start(cx, cy, 35):
            for dx in (-18, 18):
                for dy in (-18, 18):
                    things.append(Circle(cx + dx, cy + dy, 1.5, 45.0, "leg"))
    if cables:
        for _ in range(rnd.randint(1, 2)):
            x, y = rnd.uniform(20, W - 20), rnd.uniform(20, H - 20)
            pts = [(x, y)]
            for _ in range(rnd.randint(1, 3)):
                x = min(W - 5, max(5, x + rnd.uniform(-70, 70)))
                y = min(H - 5, max(5, y + rnd.uniform(-70, 70)))
                pts.append((x, y))
            if all(clear_of_start(px, py, 5) for px, py in pts):
                things.append(Cable(pts))
    if rugs and rnd.random() < 0.8:
        w, h = rnd.uniform(80, 160), rnd.uniform(60, 120)
        x0, y0 = rnd.uniform(0, W - w), rnd.uniform(0, H - h)
        rug = Rug(x0, y0, x0 + w, y0 + h)
        if not rug.inside(start[0], start[1], pad=50):
            things.append(rug)
    folks = []
    for _ in range(people):
        for _ in range(50):
            x, y = rnd.uniform(40, W - 40), rnd.uniform(40, H - 40)
            if clear_of_start(x, y, 60) and not any(o.inside(x, y, pad=30) for o in things if o.kind != "rug"):
                folks.append(Walker(x, y, rnd, walking, W, H))
                break
    return World(W, H, things, start, seed=seed, gain=rnd.uniform(0.8, 1.2), people=folks)
