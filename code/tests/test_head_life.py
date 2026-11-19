"""robot/head_life.py: smooth following, saccades, micro-motions, blinks - simulated at 50 Hz."""
import math
import random
import unittest

from robot import config
from robot.head_life import HeadLife, blink_offset

DT = 1.0 / config.HEAD_RATE_HZ


class FakeHead:
    def __init__(self):
        self.calls = []

    def set(self, pan, tilt):
        self.calls.append((pan, tilt))


def run(life, seconds, t0=0.0):
    out = []
    n = int(round(seconds / DT))
    for i in range(n):
        t = t0 + (i + 1) * DT
        out.append((t, *life.step(t)))
    return out


class BlinkShapeTest(unittest.TestCase):
    def test_dip_down_hold_and_back(self):
        self.assertEqual(blink_offset(0.0), 0.0)
        self.assertAlmostEqual(blink_offset(0.09), -config.HEAD_BLINK_DEG)
        self.assertGreater(blink_offset(0.2), -1.0)
        self.assertEqual(blink_offset(0.3), 0.0)


class HeadLifeTest(unittest.TestCase):
    def setUp(self):
        self.head = FakeHead()
        self.life = HeadLife(self.head, rng=random.Random(0))

    def test_follows_a_face_and_settles(self):
        self.life.set_focus(20, 10, "follow")
        trace = run(self.life, 4.0)
        _, pan, tilt = trace[-1]
        self.assertAlmostEqual(self.life.pos[0], 20, delta=0.5)        # spring arrived
        self.assertAlmostEqual(pan, 20, delta=3)                        # plus life on top
        self.assertAlmostEqual(tilt, 10, delta=4)

    def test_never_faster_than_the_top_speed(self):
        self.life.set_focus(60, 0, "follow")
        trace = run(self.life, 2.0)
        vmax = config.HEAD_MAX_DPS[1]
        for (t0, p0, _), (t1, p1, _) in zip(trace, trace[1:]):
            self.assertLessEqual(abs(p1 - p0) / (t1 - t0), vmax + 60)   # + micro-saccade jumps (<= ~1 deg/step)

    def test_big_jump_is_a_quick_saccade_small_one_is_soft(self):
        def time_to_90pct(target):
            life = HeadLife(None, rng=random.Random(1))
            life.set_focus(target, 0, "follow")
            for i in range(150):
                life.step((i + 1) * DT)
                if life.pos[0] >= 0.9 * target:
                    return (i + 1) * DT
            return math.inf
        jump, small = time_to_90pct(40), time_to_90pct(6)
        self.assertLess(jump, 0.5)                  # 40 deg in under half a second
        self.assertGreater(small, 0.25)             # 6 deg: soft, not snapped

    def test_focus_clamped_to_the_soft_range(self):
        self.life.set_focus(200, 99)
        self.assertEqual(self.life.focus, (config.HEAD_PAN_SOFT[1], config.HEAD_TILT_SOFT[1]))

    def test_alive_when_standing_still_but_not_flooding_the_bus(self):
        trace = run(self.life, 20.0)
        tilts = [x[2] for x in trace]
        self.assertGreater(max(tilts) - min(tilts), 1.0)                # breathing / blinks visible
        self.assertLess(min(tilts), -config.HEAD_BLINK_DEG + 1.5)       # at least one blink in 20 s
        self.assertLess(len(self.head.calls), 0.6 * len(trace))         # tiny changes not sent
        self.assertGreater(len(self.head.calls), 20)

    def test_pose_at_interpolates_the_past(self):
        self.life.set_focus(30, 0, "follow")
        trace = run(self.life, 1.0)
        t_a, p_a, _ = trace[10]
        t_b, p_b, _ = trace[11]
        p, _ = self.life.pose_at((t_a + t_b) / 2)
        self.assertAlmostEqual(p, (p_a + p_b) / 2, places=6)
        self.assertEqual(self.life.pose_at(-5)[0], trace[0][1])         # older than the history
        self.assertEqual(self.life.pose_at(99)[0], trace[-1][1])

    def test_speed_at_sees_a_jump_and_a_still_head(self):
        self.assertEqual(self.life.speed_at(0, 1), 0.0)                 # no history yet
        self.life.start_at(0, 0)
        run(self.life, 3.0)                                             # settled (life motions only)
        self.life.set_focus(40, 0, "follow")
        trace = run(self.life, 2.0, t0=3.0)
        t_jump = trace[0][0]
        self.assertGreater(self.life.speed_at(t_jump, t_jump + 0.4), 100)    # a quick move to a face
        self.assertLess(self.life.speed_at(trace[-1][0] - 0.08, trace[-1][0]), 30)

    def test_start_at_puts_the_spring_where_the_head_is(self):
        self.life.start_at(15, 5)
        run(self.life, 0.5)
        self.assertAlmostEqual(self.life.pos[0], 15, delta=0.01)


    def test_idle_moves_are_slow_smooth_and_arrive(self):
        """Looking around (nobody there): a lazy minimum-jerk move - soft start, soft stop, no whip."""
        life = HeadLife(None, rng=random.Random(2))
        life.start_at(0, 5)
        run(life, 1.0)
        life.set_focus(30, 5, "idle")
        trace = run(life, 3.0, t0=1.0)
        pos = [(t, life_pan) for t, life_pan, _ in trace]
        speeds = [abs(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(pos, pos[1:])]
        self.assertLess(max(speeds), 70)                                  # was a 240 deg/s jump
        self.assertLess(max(speeds[:5]), 15)                              # soft start
        self.assertAlmostEqual(life.pos[0], 30, delta=0.5)                # arrived
        t90 = next(t for t, p in pos if p >= 27)
        self.assertGreater(t90 - 1.0, 0.8)                                # 30 deg takes about a second

    def test_a_moving_face_during_a_move_is_still_reached(self):
        life = HeadLife(None, rng=random.Random(3))
        life.set_focus(30, 0, "follow")
        for i in range(100):
            life.set_focus(30 + i * 0.1, 0, "follow")                    # the face walks on
            life.step((i + 1) * DT)
        self.assertAlmostEqual(life.pos[0], 39.9, delta=2.0)


class DriveModeTest(unittest.TestCase):
    """robot/legs.py drives: the head goes exactly where navigation asks, at NAV_HEAD_DPS, with no life on top."""

    def setUp(self):
        self.life = HeadLife(FakeHead(), rng=random.Random(1))

    def test_goes_straight_there_at_nav_speed_and_holds_exactly(self):
        self.life.set_focus(10, 5, "follow")
        run(self.life, 3.0)
        self.life.set_focus(-30, config.NAV_TILT_DEG, "drive")
        trace = run(self.life, 2.0)
        for (t0, p0, q0), (t1, p1, q1) in zip(trace, trace[1:]):
            if t1 > t0:
                self.assertLessEqual(max(abs(p1 - p0), abs(q1 - q0)) / (t1 - t0), config.NAV_HEAD_DPS + 1e-6)
        tail = trace[-20:]
        self.assertTrue(all(abs(p + 30) < 1e-9 and abs(q - config.NAV_TILT_DEG) < 1e-9 for _, p, q in tail))

    def test_back_to_life_after_the_drive(self):
        self.life.set_focus(0, config.NAV_TILT_DEG, "drive")
        run(self.life, 1.0)
        self.life.set_focus(0, 0, "idle")
        run(self.life, 4.0)
        self.assertAlmostEqual(self.life.pos[1], 0, delta=1.0)


class HeadThreadTest(unittest.TestCase):
    def test_a_failing_step_is_logged_and_the_loop_goes_on(self):
        class Bad(FakeHead):
            def set(self, pan, tilt):
                raise ValueError("a bug somewhere")

        import threading
        stop = threading.Event()
        clock = [0.0]
        sleeps = []

        def sleep(s):
            sleeps.append(s)
            clock[0] += DT
            if len(sleeps) >= 200:
                stop.set()
        life = HeadLife(Bad(), rng=random.Random(0), clock=lambda: clock[0], sleep=sleep)
        with self.assertLogs("robot.head_life", level="ERROR") as logs:
            life.run(stop)                                  # returns when stopped: never died on the way
        self.assertEqual(len(sleeps), 200)
        self.assertGreater(life.errors, 1)
        self.assertEqual(len(logs.records), 1)              # 4 s of failures: logged once (HEAD_ERROR_LOG_S)


class ListeningTest(unittest.TestCase):
    """Selim: 'like people nod while you speak' - small slow nods at the pauses, a slight head tilt, silent."""

    def listen(self, script, mode="listen", seed=0, seconds=None):
        """script(t) -> hearing() value. Returns [(t, nod depth now)], the life."""
        clock = [0.0]
        life = HeadLife(None, rng=random.Random(seed), hearing=lambda: script(clock[0]))
        life.set_focus(10, 20, mode)
        out = []
        for i in range(int((seconds or 20.0) / DT)):
            clock[0] = (i + 1) * DT
            life.step(clock[0])
            out.append((clock[0], life._nod))
        return out, life

    @staticmethod
    def talk(t):
        """1.5 s of speech, 0.5 s pause, over and over (None after 12 s: they stopped)."""
        if t >= 12.0:
            return None
        k = t % 2.0
        return 0.0 if k < 1.5 else k - 1.5

    def nods(self, out):
        starts = sorted({n[0] for _, n in out if n is not None})
        return starts, [n for _, n in out if n is not None]

    def test_nods_at_pauses_not_too_often_and_stops_when_they_stop(self):
        for seed in range(5):
            out, life = self.listen(self.talk, seed=seed)
            starts, nods = self.nods(out)
            self.assertGreaterEqual(len(starts), 2, seed)
            for s in starts:
                self.assertGreaterEqual(s % 2.0, 1.5 + config.HEAD_NOD_PAUSE_S - 0.03)  # in a pause
                self.assertLess(s, 12.0)                                        # not after they stopped
            for a, b in zip(starts, starts[1:]):
                self.assertGreaterEqual(b - a, config.HEAD_NOD_EVERY_S)
            for _, depth, length in nods:
                self.assertTrue(config.HEAD_NOD_DEG[0] <= depth <= config.HEAD_NOD_DEG[1])
                self.assertTrue(config.HEAD_NOD_S[0] <= length <= config.HEAD_NOD_S[1])

    def test_a_nod_is_a_soft_dip_down_and_back(self):
        from robot.head_life import nod_offset
        self.assertEqual(nod_offset(0.0, 4.0, 0.8), 0.0)
        self.assertAlmostEqual(nod_offset(0.4, 4.0, 0.8), -4.0)
        self.assertAlmostEqual(nod_offset(0.8, 4.0, 0.8), 0.0)
        self.assertEqual(nod_offset(0.9, 4.0, 0.8), 0.0)

    def test_no_nods_when_not_listening_or_nobody_speaks_or_the_ears_break(self):
        self.assertEqual(self.nods(self.listen(self.talk, mode="follow")[0])[0], [])
        self.assertEqual(self.nods(self.listen(lambda t: None)[0])[0], [])
        self.assertEqual(self.nods(self.listen(lambda t: 1 / 0)[0])[0], [])

    def test_sometimes_a_slight_head_tilt_held_while_they_talk_then_back(self):
        tilted = 0
        for seed in range(12):
            out, life = self.listen(self.talk, seed=seed, seconds=11.0)
            if abs(life._cock[0]) > 0.8 * config.HEAD_COCK_DEG[0]:
                tilted += 1
                run(life, 6.0, t0=11.0)                     # (hearing still says speech: the clock is stuck)
                life.hearing = lambda: None
                run(life, 6.0, t0=17.0)
                self.assertLess(abs(life._cock[0]), 0.1)                # eased back once they stopped
        self.assertTrue(0 < tilted < 12)                                # sometimes, not always


if __name__ == "__main__":
    unittest.main()


class BatteryOffTest(unittest.TestCase):
    """The battery goes off while the head moves: no crash, no more writes, read-only probes, back by itself."""

    class Board(FakeHead):
        def __init__(self):
            super().__init__()
            self.on, self.probes, self.inits = True, 0, 0

        def set(self, pan, tilt):
            if not self.on:
                raise OSError(121, "Remote I/O error")
            super().set(pan, tilt)

        def answers(self):
            self.probes += 1
            return self.on

        def init(self):
            self.inits += 1

    def test_stops_driving_then_comes_back_when_the_board_answers(self):
        board = self.Board()
        life = HeadLife(board, rng=random.Random(0))
        life.set_focus(20, 10, "follow")
        run(life, 1.0)
        board.on = False                                    # battery switched off
        n = len(board.calls)
        run(life, 12.0, t0=1.0)                             # no exception escapes
        self.assertEqual(len(board.calls), n)               # nothing written while it is off
        self.assertIsNotNone(life.lost_t)
        self.assertLessEqual(board.probes, 3)               # only every HEAD_PROBE_S, read-only
        board.on = True                                     # battery back on
        run(life, config.HEAD_PROBE_S + 1, t0=13.0)
        self.assertIsNone(life.lost_t)
        self.assertEqual(board.inits, 1)
        self.assertGreater(len(board.calls), n)             # moving again

    def test_pose_stays_where_the_head_stopped_while_the_board_is_lost(self):
        """Vision places faces with pose_at: a lost head does not move, so no made-up spring / life motion."""
        board = self.Board()
        life = HeadLife(board, rng=random.Random(0))
        run(life, 0.5)
        board.on = False
        run(life, 0.2, t0=0.5)                              # the first failed write: lost here
        stuck = board.calls[-1]
        life.set_focus(40, 10, "follow")                    # attention wants to look far away
        run(life, 2.0, t0=0.7)
        for t in (1.5, 2.0, 2.6):
            self.assertAlmostEqual(life.pose_at(t)[0], stuck[0], places=6)
            self.assertAlmostEqual(life.pose_at(t)[1], stuck[1], places=6)
        self.assertEqual(life.speed_at(1.0, 2.6), 0.0)
