"""Navigation: floor geometry and the floor model (robot/floor.py), the memory (robot/localmap.py), the brain
(robot/nav.py) in the simulated room (sim/navworld.py) - it must never touch anything."""
import math
import unittest

import numpy as np

from robot import config
from robot.floor import FloorSeer, FloorView, ground_points, pixel_of_ground
from robot.localmap import CAM, SONAR, LocalMap
from robot.nav import Command, Navigator, Percept, Person, speed_cm_s
from sim.navworld import Box, Cable, Circle, Rug, Walker, World, random_room, robot_outline, run

W, H = config.CAMERA_LORES
TILT = config.NAV_TILT_DEG
FLOOR_RGB = (150, 120, 90)          # wooden floor
RUG_RGB = (60, 90, 140)             # blue rug


def picture(regions, pan=0.0, tilt=TILT, seed=0):
    """Small-stream picture of the floor in front of the robot: every pixel coloured by where its ray meets the
    floor. regions = [(test(gx, gy) -> bool array, rgb)] painted in order over the floor; above the horizon a
    wall colour."""
    rnd = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    gx, gy, ok = ground_points(xs, ys, W, H, pan, tilt)
    img = np.empty((H, W, 3))
    img[:] = (200, 200, 190)
    img[ok] = FLOOR_RGB
    for test, rgb in regions:
        m = ok & test(gx, gy)
        img[m] = rgb
    img += rnd.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class GroundGeometry(unittest.TestCase):
    def test_middle_of_the_picture_is_where_the_axis_meets_the_floor(self):
        gx, gy, ok = ground_points(W / 2, H / 2, W, H, 0.0, TILT)
        self.assertTrue(ok)
        expect = config.NAV_CAM_X_CM + config.NAV_CAM_H_CM / math.tan(math.radians(-TILT))
        self.assertAlmostEqual(float(gx), expect, places=4)
        self.assertAlmostEqual(float(gy), 0.0, places=4)

    def test_above_the_horizon_has_no_floor(self):
        _, _, ok = ground_points(W / 2, 0, W, H, 0.0, 0.0)
        self.assertFalse(ok)

    def test_pixel_and_ground_are_inverse(self):
        for pan in (-40.0, 0.0, 35.0):
            for gx, gy in ((40.0, 0.0), (80.0, -25.0), (120.0, 30.0)):
                px = pixel_of_ground(gx, gy, W, H, pan, TILT)
                self.assertIsNotNone(px)
                bx, by, ok = ground_points(px[0], px[1], W, H, pan, TILT)
                self.assertTrue(ok)
                self.assertAlmostEqual(float(bx), gx, places=3)
                self.assertAlmostEqual(float(by), gy, places=3)

    def test_left_is_left(self):
        # a pixel left of the middle sees floor on the robot's left (+y); panning left moves the view left
        _, gy, _ = ground_points(W / 4, H * 0.75, W, H, 0.0, TILT)
        self.assertGreater(float(gy), 0)
        _, gy, _ = ground_points(W / 2, H * 0.75, W, H, 30.0, TILT)
        self.assertGreater(float(gy), 0)


class FloorModel(unittest.TestCase):
    def learnt(self):
        seer = FloorSeer()
        v = seer.see(picture([]), 0.0, TILT, sonar_free_cm=100.0)
        self.assertTrue(v.learnt and v.known)
        return seer

    def test_nothing_learnt_without_the_sonar_saying_free(self):
        seer = FloorSeer()
        v = seer.see(picture([]), 0.0, TILT, sonar_free_cm=None)
        self.assertFalse(v.known)
        v = seer.see(picture([]), 0.0, TILT, sonar_free_cm=20.0)
        self.assertFalse(v.known)

    def test_empty_floor_is_free_far(self):
        seer = self.learnt()
        v = seer.see(picture([], seed=1), 0.0, TILT, sonar_free_cm=100.0)
        self.assertLess(v.hit.mean(), 0.1)
        mid = v.ranges[len(v.ranges) // 2]
        self.assertGreater(mid, 100.0)

    def test_a_box_ahead_is_found_where_it_stands(self):
        seer = self.learnt()
        box = (lambda gx, gy: (gx >= 70) & (gx <= 100) & (np.abs(gy) <= 15), (40, 40, 40))
        v = seer.see(picture([box], seed=2), 0.0, TILT, sonar_free_cm=55.0)
        mid = np.abs(v.bearings) < 5
        self.assertTrue(v.hit[mid].all())
        self.assertTrue(np.all(np.abs(v.ex[mid] - 70.0) < 4.0), v.ex[mid])
        self.assertFalse(v.edge)

    def test_a_thin_cable_is_an_obstacle(self):
        seer = self.learnt()
        cable = (lambda gx, gy: np.abs(gx - 50.0) <= 0.8, (20, 20, 20))     # 1.6 cm black cable across
        v = seer.see(picture([cable], seed=3), 0.0, TILT, sonar_free_cm=150.0)
        near = v.hit & (np.abs(v.ex - 50.0) < 4.0)
        self.assertGreater(near.mean(), 0.6)

    def test_an_object_in_the_reference_patch_is_not_learnt(self):
        seer = self.learnt()
        before = seer.chroma.copy()
        shoe = (lambda gx, gy: (gx >= 18) & (gx <= 40) & (np.abs(gy) <= 9), (200, 30, 30))
        v = seer.see(picture([shoe], seed=4), 0.0, TILT, sonar_free_cm=100.0)
        self.assertFalse(v.learnt)
        np.testing.assert_array_equal(seer.chroma, before)

    def test_a_rug_edge_is_an_edge_not_a_wall_of_obstacles(self):
        seer = self.learnt()
        rug = (lambda gx, gy: gx >= 60.0, RUG_RGB)
        v = seer.see(picture([rug], seed=5), 0.0, TILT, sonar_free_cm=150.0)
        self.assertTrue(v.edge)
        self.assertAlmostEqual(v.edge_cm, 60.0 - config.NAV_CAM_X_CM, delta=8.0)

    def test_crossing_learns_the_new_floor_and_keeps_the_old(self):
        seer = self.learnt()
        on_rug = picture([(lambda gx, gy: gx >= 15.0, RUG_RGB)], seed=6)
        seer.see(on_rug, 0.0, TILT, sonar_free_cm=150.0, force_learn=True)
        v = seer.see(picture([(lambda gx, gy: gx >= 60.0, RUG_RGB)], seed=7), 0.0, TILT, sonar_free_cm=150.0)
        self.assertLess(v.hit.mean(), 0.2)         # both floors are floor now
        self.assertFalse(v.edge)


def view(rays, t=0.0, hit_at=None):
    """A FloorView with rays fanned -45..45 deg from the lens, free to `rays` cm (hit at hit_at cm if given)."""
    b = np.radians(np.linspace(-45, 45, 19))
    r = np.full(len(b), float(rays))
    ox = config.NAV_CAM_X_CM
    hit = np.zeros(len(b), bool)
    if hit_at is not None:
        r[:] = hit_at
        hit[:] = True
    return FloorView(t, 0.0, TILT, ox, 0.0, ox + r * np.cos(b), r * np.sin(b), hit)


class Memory(unittest.TestCase):
    def test_nothing_known_is_not_free(self):
        m = LocalMap()
        self.assertEqual(m.seen_gap(0.0, 0.0, 1.0), 0.0)
        self.assertEqual(m.obstacle_gap(0.0), config.NAV_SEE_CM)

    def test_a_seen_free_fan_is_free_ahead_not_behind(self):
        m = LocalMap()
        m.add_view(view(120.0), 0.0)
        self.assertGreater(m.seen_gap(0.0, 0.1, 1.0), 90.0)
        self.assertEqual(m.seen_gap(180.0, 0.1, 1.0), 0.0)
        self.assertEqual(m.seen_gap(0.0, 5.0, 1.0), 0.0)        # too old

    def test_obstacle_feet_block_the_way_and_the_side_does_not(self):
        m = LocalMap()
        m.add_points([60.0], [0.0], 0.0)
        self.assertAlmostEqual(m.obstacle_gap(0.0), 60.0 - config.NAV_FRONT_CM, places=4)
        self.assertEqual(m.obstacle_gap(60.0), config.NAV_SEE_CM)
        m.add_points([0.0], [12.0], 0.0)                         # beside the robot: blocks a turn on the spot
        self.assertLess(m.spin_gap(), config.NAV_SPIN_MARGIN_CM)

    def test_points_move_with_the_robot(self):
        m = LocalMap()
        m.add_points([60.0], [0.0], 0.0)
        m.move(0.0, 20.0)
        self.assertAlmostEqual(m.obstacle_gap(0.0), 40.0 - config.NAV_FRONT_CM, places=4)
        m.move(90.0, 0.0)                                       # turned left: it is now on the right
        rx, ry = m.to_robot(m.px, m.py)
        self.assertAlmostEqual(float(rx[0]), 0.0, places=4)
        self.assertAlmostEqual(float(ry[0]), -40.0, places=4)

    def test_each_sensor_clears_only_its_own_points(self):
        m = LocalMap()
        m.add_points([50.0], [0.0], 0.0, CAM)                   # a cable the sonar cannot see
        m.add_sonar(40.0, 0.0)                                  # a table edge the camera cannot see
        m.add_sonar(None, 0.1)                                  # sonar: nothing now
        self.assertTrue((m.ps == CAM).any())
        self.assertFalse((m.ps == SONAR).any())
        m.add_sonar(40.0, 0.2)
        m.add_view(view(120.0, 0.3), 0.3)                       # camera: floor everywhere now
        self.assertFalse((m.ps == CAM).any())
        self.assertTrue((m.ps == SONAR).any())

    def test_old_points_are_forgotten(self):
        m = LocalMap()
        m.add_points([50.0], [0.0], 0.0)
        m.forget(config.NAV_MEMORY_S + 1)
        self.assertEqual(len(m.px), 0)


class Speeds(unittest.TestCase):
    def test_estimates(self):
        self.assertEqual(speed_cm_s(0.0), 0.0)
        self.assertAlmostEqual(speed_cm_s(config.NAV_V_CRAWL), config.NAV_CRAWL_CM_S)
        self.assertAlmostEqual(speed_cm_s(config.NAV_V_CRUISE), config.NAV_CRUISE_CM_S)
        self.assertLess(speed_cm_s(config.NAV_V_BACK), 0)


def corridor(things=(), length=300.0, width=120.0, heading=0.0, **kw):
    return World(length, width, list(things), start=(40.0, width / 2, heading), **kw)


class Brain(unittest.TestCase):
    def test_waits_for_a_learnt_floor(self):
        nav = Navigator()
        safe = {"stop": False, "reasons": [], "latched": []}
        cmd = nav.step(Percept(t=0.0, yaw=0.0, safety=safe))
        self.assertEqual((cmd.state, cmd.v, cmd.turn), ("START", 0.0, 0.0))

    def test_never_moves_without_a_safety_verdict(self):
        nav = Navigator()
        for i in range(40):
            cmd = nav.step(Percept(t=i * 0.05, yaw=0.0, safety=None, floor=view(120.0, i * 0.05)))
            self.assertEqual((cmd.v, cmd.turn), (0.0, 0.0))

    def test_tipped_is_help(self):
        nav = Navigator()
        cmd = nav.step(Percept(t=0.0, yaw=0.0, safety={"stop": True, "reasons": ["tipped"], "latched": ["tipped"]}))
        self.assertEqual(cmd.state, "HELP")
        self.assertEqual((cmd.v, cmd.turn), (0.0, 0.0))

    def test_stops_before_a_wall(self):
        w = corridor(gain=1.2)                  # faster than it believes
        nav = Navigator()
        gap = [1e9]

        def body_to_end_wall(p, c):             # any side of the body: it may turn and drive along the wall
            xs, _ = robot_outline(w.x, w.y, w.h)
            gap[0] = min(gap[0], w.W - float(xs.max()))

        run(nav, w, 25, on_step=body_to_end_wall)
        self.assertEqual(w.collisions, 0)
        self.assertGreater(w.driven, 100.0)
        self.assertGreater(gap[0], 5.0)         # never closer than 5 cm to the end wall

    def test_does_not_drive_over_a_cable_or_into_chair_legs(self):
        legs = [Circle(150 + dx, 60 + dy, 1.5) for dx in (-18, 18) for dy in (-18, 18)]
        w = corridor([Cable([(110.0, 0.0), (110.0, 70.0)])] + legs, gain=0.85)
        nav = Navigator()
        run(nav, w, 60)
        self.assertEqual((w.collisions, w.cable_touches), (0, 0))

    def test_crosses_a_rug(self):
        w = corridor([Rug(120.0, 0.0, 220.0, 120.0)], heading=0.0)
        nav = Navigator()
        run(nav, w, 40)
        self.assertEqual(w.collisions, 0)
        self.assertGreater(nav.stats["crossed"], 0)
        self.assertGreater(w.driven, 120.0)

    def test_crosses_a_rug_whose_edge_is_also_remembered_as_obstacles(self):
        # on the robot most pictures of a rug edge fail the edge test (a slant, a corner, too close) and leave the
        # edge in the memory as obstacle feet - crossing must not stop at them (2026-11-04: 3 crossings stopped
        # "17 cm ahead while crossing")
        w = corridor([Rug(120.0, 0.0, 220.0, 120.0)], heading=0.0)
        nav = Navigator()
        far = [0.0]

        def edge_as_obstacles(p, c):
            far[0] = max(far[0], w.x)
            if nav.state not in ("DRIVE", "CROSS") or w.x >= 115.0:
                return
            c_, s_ = math.cos(math.radians(w.h)), math.sin(math.radians(w.h))
            ys = np.arange(20.0, 101.0, 4.0)
            dx, dy = 120.0 - w.x, ys - w.y
            rx, ry = dx * c_ + dy * s_, -dx * s_ + dy * c_
            ahead = (rx > 0) & (rx < 150)
            nav.map.add_points(rx[ahead], ry[ahead], p.t, CAM)

        run(nav, w, 40, on_step=edge_as_obstacles)
        self.assertEqual(w.collisions, 0)
        self.assertGreater(far[0], 135.0)          # really on the rug (its edge at x = 120)

    def test_at_a_rug_edge_with_everything_beyond_looking_blocked_it_crosses_instead_of_giving_up(self):
        # 2026-11-04: stopped at the rug's edge, the parquet beyond all "obstacles" (the floor model knew only the
        # rug), turned round 3 times and asked for help - the edge straight ahead with the sonar free is a way out
        w = corridor([Rug(70.0, 0.0, 200.0, 120.0)], length=200.0)      # the end wall echoes (~140 cm)
        nav = Navigator()
        far, states = [0.0], []

        def beyond_blocked(p, c):
            far[0] = max(far[0], w.x)
            states.append(nav.state)
            if "CROSS" in states:
                return
            c_, s_ = math.cos(math.radians(w.h)), math.sin(math.radians(w.h))
            a = np.radians(np.arange(-180.0, 180.0, 5.0))
            gx, gy = w.x + 50.0 * np.cos(a), w.y + 50.0 * np.sin(a)      # a ring of "obstacles" 50 cm round it
            dx, dy = gx - w.x, gy - w.y
            nav.map.add_points(dx * c_ + dy * s_, -dx * s_ + dy * c_, p.t, CAM)

        run(nav, w, 40, on_step=beyond_blocked)
        self.assertEqual(w.collisions, 0)
        self.assertIn("CROSS", states)
        self.assertNotIn("HELP", states)
        self.assertGreater(far[0], 85.0)           # on the rug (its edge at x = 70)

    def test_small_turns_overshooting_do_not_cut_the_big_turns_short(self):
        # 2026-11-08: small turns went 10-32 deg over, the one shared lead grew to 0.4 s, every 135 deg turn then
        # stopped ~60 deg short and was never learnt back (misses over 40 deg were skipped)
        nav = Navigator()
        for over in (18.9, 1.3, 12.5, 31.6, 15.1, 27.4, 9.9):
            nav._learn_lead("small", over, 100.0)
        self.assertEqual(nav._lead["big"], config.NAV_TURN_LEAD_S)
        self.assertGreater(nav._lead["small"], config.NAV_TURN_LEAD_S)
        nav._lead["big"] = 0.4
        for _ in range(3):
            nav._learn_lead("big", -60.0, 250.0)
        self.assertLess(nav._lead["big"], 0.25)                    # a big miss is learnt from, not ignored

    def test_the_sonar_must_echo_well_past_the_edge(self):
        nav = Navigator()
        lens = config.NAV_FRONT_CM - config.NAV_SONAR_X_CM

        def ok(echo_from_front, edge_ahead):
            nav._sonar = (0.0, echo_from_front + lens)
            return nav._edge_sonar_ok(0.0, edge_ahead)

        self.assertTrue(ok(50.0, 7.0))          # 2026-11-04: rug edge 7 cm ahead, a chair 50 cm away - cross
        self.assertFalse(ok(17.0, 7.0))         # a box standing at the edge
        self.assertFalse(ok(35.0, 2.0))         # never on less than NAV_EDGE_SONAR_MIN_CM
        self.assertFalse(ok(70.0, 50.0))        # a far edge needs the echo 30 cm past it
        self.assertTrue(ok(85.0, 50.0))
        nav._sonar = (0.0, None)                # nothing heard is no proof (a wall at a slant)
        self.assertFalse(nav._edge_sonar_ok(0.0, 7.0))

    def test_no_crossing_with_something_remembered_before_the_edge(self):
        nav = Navigator()
        nav.map.add_points([config.NAV_FRONT_CM + 1.0], [3.0], 0.0, CAM)          # 1 cm ahead of the front
        self.assertFalse(nav._clear_to_edge(50.0))
        nav = Navigator()
        nav.map.add_points([config.NAV_FRONT_CM + 52.0], [3.0], 0.0, CAM)         # the edge itself
        self.assertTrue(nav._clear_to_edge(50.0))

    def test_a_box_just_past_the_edge_still_stops_a_crossing(self):
        m = LocalMap()
        m.add_points([config.NAV_FRONT_CM + 30.0, config.NAV_FRONT_CM + 60.0], [0.0, 0.0], 0.0, CAM)
        m.forget_band(30.0 - config.NAV_CROSS_BAND_CM, 30.0 + config.NAV_CROSS_BAND_CM, 15.0)
        self.assertAlmostEqual(m.obstacle_gap(0.0), 60.0, delta=0.5)

    def test_a_hit_it_did_not_see_backs_up_and_too_many_ask_for_help(self):
        w = corridor()
        nav = Navigator()
        cmds = []
        run(nav, w, 4, on_step=lambda p, c: cmds.append(c))
        w.latched.add("bump")                   # glass: no sensor saw it
        run(nav, w, 3, on_step=lambda p, c: cmds.append(c))
        states = [c.state for c in cmds]
        self.assertIn("BUMPED", states)
        self.assertTrue(any(c.reset_safety for c in cmds))
        self.assertIn("BACK", states[states.index("BUMPED"):])
        self.assertTrue(all(c.v < 0 for c in cmds if c.state == "BACK"))
        for _ in range(config.NAV_BUMP_RESETS):
            w.latched.add("bump")
            run(nav, w, 3)
        self.assertEqual(nav.state, "HELP")
        cmd = nav.step(w.percept())
        self.assertEqual((cmd.v, cmd.turn), (0.0, 0.0))

    def test_random_rooms_no_contact(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                w = random_room(seed)
                nav = Navigator()
                run(nav, w, 90)
                self.assertEqual(w.collisions, 0, getattr(w, "last_hit", None))
                self.assertEqual(w.cable_touches, 0)
                self.assertGreater(w.driven, 150.0)

    def test_goes_to_a_person_and_stops_short(self):
        rnd = __import__("random").Random(1)
        w = corridor(people=[Walker(220.0, 60.0, rnd)])
        nav = Navigator("people")
        gaps = []

        def on(p, c):
            if c.state == "GREET" and not gaps:
                gaps.append(math.hypot(220.0 - w.x, 60.0 - w.y) - 6.0 - config.NAV_FRONT_CM)
        run(nav, w, 30, on_step=on)
        self.assertEqual(w.collisions, 0)
        self.assertEqual(len(gaps), 1, nav.events[-10:])
        self.assertGreater(gaps[0], 25.0)
        self.assertLess(gaps[0], config.NAV_PERSON_STOP_CM + 15.0)

    def test_people_mode_rooms_no_contact(self):
        for seed in (100, 102):
            with self.subTest(seed=seed):
                w = random_room(seed, people=1, walking=True)
                nav = Navigator("people")
                run(nav, w, 90)
                self.assertEqual(w.collisions, 0, getattr(w, "last_hit", None))


def follow_run(nav, w, seconds, keep=lambda t: True, on_step=None):
    """Like sim.navworld.run, with the followed person as the Percept's target: the person seen nearest to where the
    target was last (alive.py does the same with body tracks). keep(t) False = the person is not reported (lost)."""
    rate = config.NAV_RATE_HZ
    last = [None]
    for _ in range(int(seconds * rate)):
        p = w.percept()
        if p.people and keep(w.t):
            best = min(p.people, key=lambda q: abs(q.bearing - last[0]) if last[0] is not None else q.dist or 999)
            last[0] = best.bearing
            p.target, p.target_t = best, p.people_t
        cmd = nav.step(p)
        w.apply(cmd)
        if on_step:
            on_step(p, cmd)
        w.advance(1.0 / rate)
    return w


class Follow(unittest.TestCase):
    """Follow mode (alive.py on "gel gel"): after a walking person, never touching anything (sim/navworld.py)."""

    def test_follows_a_walker_through_rooms_without_contact(self):
        followed = []
        for seed in range(200, 208):
            with self.subTest(seed=seed):
                w = random_room(seed, people=1, walking=True)
                p = w.people[0]
                w.h = w.yaw0 = math.degrees(math.atan2(p.y - w.y, p.x - w.x))     # they beckoned: in view
                nav = Navigator("follow")
                gaps = []

                def on(_, c):
                    if c.v > 0:                                 # (they may walk up to a robot standing still)
                        gaps.append(math.hypot(p.x - w.x, p.y - w.y) - 6.0 - config.NAV_FRONT_CM)
                follow_run(nav, w, 60, on_step=on)
                self.assertEqual(w.collisions, 0, getattr(w, "last_hit", None))
                self.assertGreater(min(gaps, default=99.0), 15.0)     # never drives into them
                followed.append(nav.stats["followed_cm"])
        # furniture between them stops it (no way round in v1) - but it does follow in most rooms
        self.assertGreaterEqual(sum(f > 50.0 for f in followed), 4, followed)

    def test_stands_about_60_cm_from_a_still_person_and_faces_them(self):
        rnd = __import__("random").Random(3)
        w = corridor(people=[Walker(260.0, 60.0, rnd)])
        nav = Navigator("follow")
        follow_run(nav, w, 15)
        gap = math.hypot(260.0 - w.x, 60.0 - w.y) - 6.0 - config.NAV_FRONT_CM
        self.assertEqual(w.collisions, 0)
        self.assertGreater(gap, config.FOLLOW_STOP_CM - 20.0)
        self.assertLess(gap, config.FOLLOW_STOP_CM + config.FOLLOW_GO_CM)
        self.assertEqual(nav.state, "FOLLOW")
        self.assertLess(abs(w.v), 1.0)                          # standing

    def test_lost_looks_around_then_done(self):
        rnd = __import__("random").Random(4)
        w = corridor(people=[Walker(300.0, 60.0, rnd)])
        nav = Navigator("follow")
        states = []
        follow_run(nav, w, 14, keep=lambda t: t < 2.0, on_step=lambda p, c: states.append(c.state))
        self.assertIn("LOST", states)
        self.assertEqual(nav.state, "DONE")
        self.assertEqual(nav.note, "lost them")
        lost_at = next(i for i, s in enumerate(states) if s == "LOST" and i / config.NAV_RATE_HZ > 2.0) \
            / config.NAV_RATE_HZ
        self.assertGreater(lost_at, 2.0 + config.FOLLOW_LOST_S - 0.5)
        done_at = states.index("DONE") / config.NAV_RATE_HZ
        self.assertAlmostEqual(done_at - lost_at, config.FOLLOW_REFIND_S, delta=0.5)

    def test_found_again_while_looking_goes_on(self):
        rnd = __import__("random").Random(5)
        w = corridor(people=[Walker(300.0, 60.0, rnd)])
        nav = Navigator("follow")
        states = []
        follow_run(nav, w, 14, keep=lambda t: not 2.0 < t < 6.5, on_step=lambda p, c: states.append(c.state))
        self.assertIn("LOST", states)
        self.assertNotIn("DONE", states)
        self.assertEqual(nav.state, "FOLLOW")

    def test_never_seen_is_lost_then_done(self):
        w = corridor()
        nav = Navigator("follow")
        follow_run(nav, w, config.FOLLOW_LOST_S + config.FOLLOW_REFIND_S + 1.0)
        self.assertEqual(nav.state, "DONE")
        self.assertEqual(w.driven, 0.0)


class Commands(unittest.TestCase):
    def test_command_defaults_are_standing_still_looking_down(self):
        c = Command()
        self.assertEqual((c.v, c.turn, c.head[1]), (0.0, 0.0, config.NAV_TILT_DEG))

    def test_person_bearing_is_kept_in_room_coordinates(self):
        nav = Navigator("people")
        nav.t = 0.0
        p = Percept(t=0.0, yaw=30.0, people=[Person(10.0, 200.0)], people_t=0.0)
        nav._see_people(p)
        self.assertAlmostEqual(nav._person[0], 40.0)
        self.assertAlmostEqual(nav._person[1], 200.0)


if __name__ == "__main__":
    unittest.main()
