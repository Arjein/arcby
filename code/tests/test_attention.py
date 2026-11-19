"""robot/attention.py: who to look at, holding, searching, idling, glances, body turns."""
import random
import unittest

from robot import config
from robot.attention import Attention
from robot.tracker import Sighting, Track


def track(tid, pan, tilt=10.0, size=6.0, person=None, unknown=False, engagement=0.0, looking=False):
    return Track(tid, Sighting(None, pan, tilt, size), 0.0, 0.0, person=person, unknown=unknown,
                 engagement=engagement, looking_since=0.0 if looking else None)


class AttentionTest(unittest.TestCase):
    def setUp(self):
        self.a = Attention(rng=random.Random(0), hour=lambda: 14)

    def test_follows_the_only_face(self):
        self.assertEqual(self.a.update([track(1, 20)], 0.0), (20, 10.0, "follow"))

    def test_a_thing_held_up_beats_the_face_then_back_to_it(self):     # robot/show.py, 2026-11-02
        self.assertEqual(self.a.update([track(1, 20, person="selim")], 0.0)[2], "follow")
        pan, tilt, mode = self.a.update([track(1, 20, person="selim")], 0.1, shown=(14.0, -2.0, "cell phone"))
        self.assertEqual((pan, tilt, mode), (14.0, -2.0, "show"))
        self.assertEqual(self.a.update([track(1, 20, person="selim")], 0.2)[:1] + ("x",), (20, "x"))
        self.assertEqual(self.a.mode, "follow")
        _, tilt, _ = self.a.update([], 0.3, shown=(0.0, -80.0, None))      # never past the head's soft limit
        self.assertEqual(tilt, config.HEAD_TILT_SOFT[0])

    def test_known_person_beats_a_stranger_of_the_same_size(self):
        pan, _, mode = self.a.update([track(1, -20), track(2, 20, person="selim")], 0.0)
        self.assertEqual((pan, mode), (20, "follow"))

    def test_sticks_with_the_current_face_unless_clearly_better(self):
        self.a.update([track(1, -20, size=6)], 0.0)
        both = [track(1, -20, size=6), track(2, 20, size=7)]
        self.a.update(both, 0.1)
        t0 = 0.1 + config.ATTN_NEW_MIN_S
        self.assertEqual(self.a.update(both, t0)[2], "glance")                # a glance at the newcomer...
        t = t0 + config.ATTN_GLANCE_S + 0.01
        pan, _, mode = self.a.update(both, t)
        self.assertEqual((pan, mode), (-20, "follow"))                         # ...then back
        closer = [track(1, -20, size=6), track(2, 20, size=12)]
        pan, _, _ = self.a.update(closer, t + 0.1)
        self.assertEqual(pan, -20)                     # 2x more interesting: not before ATTN_MIN_DWELL_S
        pan, _, _ = self.a.update(closer, config.ATTN_MIN_DWELL_S + 0.1)
        self.assertEqual(pan, 20)                                              # much closer now: switch

    def test_a_much_more_interesting_face_takes_over_at_once(self):
        self.a.update([track(1, -20, size=6)], 0.0)
        pan, _, _ = self.a.update([track(1, -20, size=6), track(2, 20, size=6, person="selim", engagement=1.0)], 0.1)
        self.assertEqual(pan, 20)                                              # 8x: no waiting

    def test_someone_looking_at_the_robot_wins_over_someone_who_is_not(self):
        pan, _, _ = self.a.update([track(1, -20, person="emre"), track(2, 20, person="selim", engagement=1.0)], 0.0)
        self.assertEqual(pan, 20)
        # the followed person turns away, the other looks at the robot: attention moves over
        self.a = Attention(rng=random.Random(0), hour=lambda: 14)
        self.a.update([track(1, -20, person="emre", engagement=1.0), track(2, 20, person="selim")], 0.0)
        pan, _, _ = self.a.update([track(1, -20, person="emre"), track(2, 20, person="selim", engagement=1.0)], 0.1)
        self.assertEqual(pan, 20)

    def test_looks_round_the_people_who_look_at_it(self):
        group = [track(1, 0, person="selim", engagement=1.0, looking=True),
                 track(2, -30, person="meltem", engagement=0.7, looking=True),
                 track(3, 30, person="emre", engagement=0.7, looking=True),
                 track(4, 45, person="halil")]                    # Halil ignores the robot
        modes, glanced = [], []
        t = 0.0
        while t < 40.0:
            pan, _, mode = self.a.update(group, t)
            modes.append(mode)
            if mode == "glance":
                glanced.append(pan)
            t += 0.1
        self.assertGreater(modes.count("follow"), 0.7 * len(modes))      # mostly with the main person
        self.assertEqual(set(glanced) - {-30, 30}, set())                 # glances only at people looking
        self.assertEqual(set(glanced), {-30, 30})                         # ...and at both of them, in turn

    def test_newcomer_gets_a_glance_then_back(self):
        self.a.update([track(1, 0)], 0.0)
        pan, _, mode = self.a.update([track(1, 0), track(2, 30)], 0.1)
        self.assertEqual((pan, mode), (0, "follow"))                           # not for a 1-frame blip
        t = 0.1 + config.ATTN_NEW_MIN_S
        pan, _, mode = self.a.update([track(1, 0), track(2, 30)], t)
        self.assertEqual((pan, mode), (30, "glance"))
        pan, _, mode = self.a.update([track(1, 0), track(2, 30)], t + config.ATTN_GLANCE_S + 0.01)
        self.assertEqual((pan, mode), (0, "follow"))

    def test_a_face_found_again_where_one_was_lost_is_no_newcomer(self):
        """The tracker restarts real faces (6-13 new tracks a minute, 2026-11-01): no glance for each."""
        self.a.update([track(1, 0), track(2, 30)], 0.0)
        self.a.update([track(1, 0)], 0.1)                                      # #2 lost...
        for k in range(20):                                                    # ...#3 right there: the same face
            self.assertEqual(self.a.update([track(1, 0), track(3, 31)], 0.2 + k * 0.1)[2], "follow")

    def test_a_newcomer_next_to_the_followed_face_gets_no_glance(self):
        self.a.update([track(1, 0)], 0.0)
        for k in range(20):
            self.assertEqual(self.a.update([track(1, 0), track(2, 6)], 0.1 + k * 0.1)[2], "follow")

    def test_a_picture_on_the_wall_is_never_looked_at(self):
        photo = track(1, 30, person="bf1137bf")
        photo.picture = True
        for k in range(100):
            _, _, mode = self.a.update([photo], k * 0.1)
            self.assertEqual(mode, "idle")

    def test_a_face_lost_and_found_near_where_it_was_is_followed_again(self):
        self.a.update([track(1, 20, person="selim"), track(2, -30, size=20)], 0.0)
        self.assertEqual(self.a.target, 2)
        self.a.update([track(1, 20, person="selim")], 0.1)                      # the followed face lost a moment
        pan, _, mode = self.a.update([track(1, 20, person="selim"), track(3, -28, size=6)], 0.3)
        self.assertEqual((pan, mode), (-28, "follow"))                         # the one where it was, not Selim

    def test_the_followed_face_dropping_out_a_moment_is_waited_for(self):
        self.a.update([track(1, 20, person="selim"), track(2, -30, size=20)], 0.0)
        for k in range(10):                                                   # #2 unseen for 1 s
            self.assertEqual(self.a.update([track(1, 20, person="selim")], 0.1 + 0.1 * k)[2], "hold")
        pan, _, mode = self.a.update([track(1, 20, person="selim"), track(2, -30, size=20)], 1.1)
        self.assertEqual((pan, mode), (-30, "follow"))                        # back, no swing to Selim and back
        pan, _, mode = self.a.update([track(1, 20, person="selim")], 1.2 + config.ATTN_HOLD_S)
        self.assertEqual((pan, mode), (20, "follow"))                         # gone for good: Selim

    def test_lost_face_hold_then_search_then_idle(self):
        self.a.update([track(1, 20, 12)], 0.0)
        self.assertEqual(self.a.update([], 0.5), (20, 12, "hold"))
        pan, _, mode = self.a.update([], config.ATTN_HOLD_S + 0.5)
        self.assertEqual(mode, "search")
        self.assertLessEqual(abs(pan - 20), config.ATTN_SEARCH_DEG)                # a small look around the spot
        self.assertEqual(self.a.update([], config.ATTN_HOLD_S + config.ATTN_SEARCH_S + 0.1),
                         (20, 12, "idle"))                                          # then stays there (no jump)

    def test_idle_is_calm_rests_ahead_and_comes_back(self):
        """Selim 2026-11-01: 'the head moves randomly' - idle used to jump to a random place every 2.5-6 s
        (median 30 deg). Now: rest ahead, a slow look somewhere now and then, held, then back."""
        targets, t = [], 0.0
        while t < 600.0:
            pan, tilt, mode = self.a.update([], t)
            self.assertEqual(mode, "idle")
            self.assertTrue(config.ATTN_IDLE_TILT[0] <= tilt <= config.ATTN_IDLE_TILT[1])
            self.assertLessEqual(abs(pan), config.ATTN_IDLE_PAN_MAX)
            if not targets or targets[-1] != (pan, tilt):
                targets.append((pan, tilt))
            t += 0.5
        home = (0.0, config.ATTN_IDLE_HOME_TILT)
        self.assertLess(len(targets) - 1, 600 / 60 * 8)                   # < 8 changes a minute (was ~20)
        self.assertGreater(len(set(targets)), 10)                         # but it does look about
        for a, b in zip(targets, targets[1:]):
            self.assertIn(home, (a, b))                                   # every look starts or ends at rest

    def test_idle_looks_go_where_people_were(self):
        self.a.update([track(1, -25, tilt=15, person="selim")], 0.0)
        self.a.partner = None
        looks, t = [], 1.0
        while t < 900.0:
            pan, tilt, _ = self.a.update([], t)
            looks.append((pan, tilt))
            t += 0.5
        self.assertIn((-25, 15), looks)

    def test_bored_it_looks_away_at_once_not_holding_on_them(self):
        selim = [track(1, 10, person="selim")]
        modes, _ = [], None
        t = 0.0
        while t < 40.0:
            modes.append(self.a.update(selim, t)[2])
            t += 0.1
        i = modes.index("idle")
        self.assertNotIn("hold", modes[i - 1:i + 30])
        self.assertNotIn("search", modes[i - 1:i + 30])

    def test_body_turn_asked_after_following_far_to_the_side(self):
        self.a.update([track(1, 55)], 0.0)
        self.assertIsNone(self.a.body_turn)
        self.a.update([track(1, 55)], config.ATTN_BODY_TURN_S + 0.1)
        self.assertEqual(self.a.body_turn, 55)
        self.a.update([track(1, 10)], config.ATTN_BODY_TURN_S + 0.2)
        self.assertIsNone(self.a.body_turn)

    def test_a_face_missing_one_frame_is_not_a_newcomer(self):
        self.a.update([track(1, 0), track(2, 30)], 0.0)
        self.a.update([track(1, 0)], config.ATTN_GLANCE_S + 0.1)
        for k in range(10):
            _, _, mode = self.a.update([track(1, 0), track(2, 30)], config.ATTN_GLANCE_S + 0.2 + 0.1 * k)
            self.assertEqual(mode, "follow")


class AliveTest(unittest.TestCase):
    """Boredom, eye-contact breaks, turning to voices, night - the robot should not just stare at faces."""

    def setUp(self):
        self.a = Attention(rng=random.Random(1), hour=lambda: 14)

    def run_for(self, seconds, tracks, t0=0.0, step=0.1):
        modes, t = [], t0
        while t < t0 + seconds - 1e-9:
            modes.append(self.a.update(tracks() if callable(tracks) else tracks, t)[2])
            t += step
        return modes, t

    def test_someone_who_just_sits_there_gets_boring_then_interesting_again(self):
        selim = [track(1, 10, person="selim")]
        modes, t = self.run_for(25, selim)
        self.assertEqual(modes[0], "follow")
        self.assertNotEqual(modes[-1], "follow")                     # looked away after ~21 s
        self.assertGreater(modes.count("follow"), 180)
        modes, t = self.run_for(60, selim, t)
        self.assertIn("follow", modes[-300:])                        # ...and back again later

    def test_someone_looking_at_it_holds_attention_longer_with_short_breaks(self):
        selim = [track(1, 10, person="selim", engagement=1.0, looking=True)]
        modes, _ = self.run_for(30, selim)
        self.assertTrue(all(m in ("follow", "break") for m in modes))
        self.assertGreater(modes.count("break"), 3)
        self.assertGreater(modes.count("follow"), 4 * modes.count("break"))

    def test_never_bored_of_whom_it_talks_with(self):
        self.a.partner = "selim"
        modes, _ = self.run_for(120, [track(1, 10, person="selim")])
        self.assertTrue(all(m == "follow" for m in modes))

    def test_a_known_voice_turns_it_to_where_that_person_was(self):
        self.a.update([track(1, -40, person="meltem")], 0.0)          # Meltem was over there...
        self.a.update([track(2, 20, person="selim")], 30.0)           # ...now only Selim is in view
        self.a.hear(30.05, person="meltem")
        pan, _, mode = self.a.update([track(2, 20, person="selim")], 30.1)
        self.assertEqual((pan, mode), (-40, "orient"))
        pan, _, mode = self.a.update([track(2, 20, person="selim")], 30.5)
        self.assertEqual(mode, "orient")                             # not pulled back to Selim
        pan, _, mode = self.a.update([track(2, 20, person="selim"), track(3, -35, person="meltem")], 31.0)
        self.assertEqual((pan, mode), (-35, "follow"))

    def test_a_voice_in_view_is_looked_at(self):
        self.a.update([track(1, -30, person="emre"), track(2, 30, person="selim", size=12)], 0.0)
        self.a.hear(0.05, person="emre")
        pan, _, mode = self.a.update([track(1, -30, person="emre"), track(2, 30, person="selim", size=12)], 0.1)
        self.assertEqual((pan, mode), (-30, "follow"))

    def test_its_name_from_nowhere_makes_it_look_round_then_at_whoever_appears(self):
        self.a.update([], 0.0)
        self.a.hear(0.05, name=True)
        modes, t = self.run_for(1.0, [], 0.1)
        self.assertTrue(all(m == "scan" for m in modes))
        pan, _, mode = self.a.update([track(5, 45)], t)
        self.assertEqual((pan, mode), (45, "follow"))
        self.assertEqual([e["how"] for e in self.a.events], ["sweep"])         # for the dashboard

    def test_a_voice_it_cannot_place_gets_a_calm_look_ahead_not_a_sweep(self):
        """11 of 12 sweeps on 2026-11-01 were voices that did not call it (+-60 deg swings for nothing)."""
        self.a.update([track(1, 0, tilt=30, person="selim")], 0.0)
        self.a.update([], 30.0)
        self.a.hear(30.05)
        modes, _ = self.run_for(config.ATTN_ORIENT_S - 0.2, [], 30.1)
        self.assertTrue(all(m == "orient" for m in modes))
        self.assertEqual(self.a.update([], 31.0)[:2], (0.0, 30))                 # ahead, at face height
        self.assertEqual([e["how"] for e in self.a.events], ["ahead"])

    def test_it_looks_round_in_three_stops_ahead_then_far_left_and_right(self):
        self.a.update([track(1, 0, tilt=30, person="selim")], 0.0)        # Selim sits higher than the head
        self.a.update([], 30.0)
        self.a.hear(30.05, name=True)
        seen = []
        for k in range(int(3 * config.ATTN_SCAN_FIX_S / 0.1)):
            pan, tilt, mode = self.a.update([], 30.1 + k * 0.1)
            self.assertEqual(mode, "scan")
            if not seen or seen[-1] != (pan, tilt):
                seen.append((pan, tilt))
        self.assertEqual([p for p, _ in seen][0], 0.0)                    # straight ahead first...
        self.assertEqual({p for p, _ in seen}, {0.0, 60.0, -60.0})        # ...then both far sides
        self.assertEqual({t for _, t in seen}, {30 + config.ATTN_SCAN_UP_DEG})
        a = Attention(rng=random.Random(0), hour=lambda: 14)              # nobody seen yet
        a.update([], 0.0)
        a.hear(0.05, name=True)
        self.assertEqual(a.update([], 0.1)[1], config.ATTN_SCAN_TILT + config.ATTN_SCAN_UP_DEG)

    def test_a_voice_while_facing_someone_looking_at_it_does_not_turn_it(self):
        selim = [track(1, 0, person="selim", engagement=1.0, looking=True)]
        self.a.update(selim, 0.0)
        self.a.hear(0.05)
        self.assertIn(self.a.update(selim, 0.1)[2], ("follow", "break"))

    def test_people_talking_on_and_on_do_not_turn_it_every_sentence(self):
        self.a.update([], 0.0)
        self.a.hear(0.05)
        _, t = self.run_for(config.ATTN_ORIENT_S + 0.5, [], 0.1)
        self.a.hear(t)
        self.assertNotIn(self.a.update([], t + 0.1)[2], ("scan", "orient"))
        self.a.hear(t + 0.2, name=True)                              # its name always does
        self.assertEqual(self.a.update([], t + 0.3)[2], "scan")

    def test_an_unplaced_voice_in_a_conversation_looks_at_the_partner_not_round_the_room(self):
        self.a.partner = "selim"
        scene = [track(1, 12, person="selim", engagement=0.2), track(2, -40)]
        self.a.update(scene, 0.0)
        self.a.update(scene[1:], 0.1)                                # Selim out of view a moment
        self.a.hear(0.15)
        modes, _ = self.run_for(1.0, scene[1:], 0.2)
        self.assertNotIn("scan", modes)                              # never swings away mid-conversation
        self.a.hear(1.25)
        pan, _, mode = self.a.update(scene, 1.3)
        self.assertEqual((pan, mode), (12, "follow"))

    def test_an_unplaced_voice_with_one_known_face_in_view_is_theirs(self):
        scene = [track(1, 12, person="selim", engagement=0.2), track(2, -40, size=12)]
        self.a.update(scene, 0.0)
        self.a.hear(0.05)
        pan, _, mode = self.a.update(scene, 0.1)
        self.assertEqual((pan, mode), (12, "follow"))
        self.assertEqual(self.a.events[-1]["how"], "face")

    def test_no_head_no_turns(self):
        a = Attention(rng=random.Random(0), hour=lambda: 14, has_head=False)
        a.update([], 0.0)
        a.hear(0.05)
        a.hear(0.06, name=True)
        self.assertNotEqual(a.update([], 0.1)[2], "scan")
        self.assertEqual(a.events, [])

    def test_the_night_is_ankaras(self):
        a = Attention(rng=random.Random(0), tz="Europe/Istanbul")
        from datetime import datetime
        from zoneinfo import ZoneInfo
        self.assertEqual(a._hour(), datetime.now(ZoneInfo("Europe/Istanbul")).hour)

    def test_starting_to_look_at_it_wakes_up_a_bored_person(self):
        modes, t = self.run_for(25, [track(1, 10, person="selim")])
        self.assertNotEqual(modes[-1], "follow")
        _, _, mode = self.a.update([track(1, 10, person="selim", engagement=1.0, looking=True)], t)
        self.assertIn(mode, ("follow", "break"))

    def test_listening_keeps_its_eyes_on_the_speaker(self):
        speaking = [0.0]
        self.a.hearing = lambda: speaking[0]
        selim = [track(1, 10, person="selim", engagement=1.0, looking=True), track(2, -30, person="emre",
                                                                                 engagement=1.0, looking=True)]
        modes, t = self.run_for(30, selim)
        self.assertEqual(set(modes), {"listen"})                        # no breaks, no glances at Emre, no boredom
        speaking[0] = None                                              # they stopped
        modes, _ = self.run_for(30, selim, t)
        self.assertEqual(modes[:int(config.ATTN_LISTEN_HOLD_S / 0.1) - 1], ["listen"] * 14)
        self.assertNotIn("listen", modes[20:])
        self.assertIn("break", modes)

    def test_listening_only_to_the_partner_or_someone_looking(self):
        self.a.hearing = lambda: 0.0
        modes, _ = self.run_for(5, [track(1, 10, person="selim")])       # not looking, not talking with it
        self.assertNotIn("listen", modes)
        self.a.partner = "selim"
        self.assertEqual(self.a.update([track(1, 10, person="selim")], 5.1)[2], "listen")

    def test_no_head_no_listening_and_broken_ears_are_no_crash(self):
        a = Attention(rng=random.Random(0), hour=lambda: 14, has_head=False, hearing=lambda: 0.0)
        a.partner = "selim"
        self.assertEqual(a.update([track(1, 10, person="selim")], 0.0)[2], "follow")
        self.a.partner = "selim"
        self.a.hearing = lambda: 1 / 0
        self.assertEqual(self.a.update([track(1, 10, person="selim")], 0.0)[2], "follow")

    def test_night_idle_looks_lower(self):
        a = Attention(rng=random.Random(0), hour=lambda: 2)
        tilts = [a.update([], k * 0.5)[1] for k in range(200)]
        self.assertLessEqual(max(tilts), config.ATTN_NIGHT_TILT[1])


if __name__ == "__main__":
    unittest.main()
