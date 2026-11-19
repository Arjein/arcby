"""robot/tracker.py: face tracks across frames and identity votes."""
import unittest

from robot.tracker import Sighting, Tracker


def s(pan, tilt=10.0, size=6.0):
    return Sighting(face=None, pan=pan, tilt=tilt, size_deg=size)


class TrackingTest(unittest.TestCase):
    def test_same_face_moving_a_little_keeps_its_track(self):
        tr = Tracker()
        new, _ = tr.update([s(0)], 0.0)
        tid = new[0].id
        for k in range(1, 10):
            new, ended = tr.update([s(k * 0.8)], k * 0.1)          # walking slowly to the left
            self.assertEqual((new, ended), ([], []))
        self.assertEqual([t.id for t in tr.tracks], [tid])
        self.assertEqual(tr.tracks[0].hits, 10)

    def test_two_people_get_two_tracks_and_do_not_swap(self):
        tr = Tracker()
        a, b = tr.update([s(-20), s(20)], 0.0)[0]
        tr.update([s(18), s(-19)], 0.1)                             # order in the frame changed
        self.assertAlmostEqual(a.pan, -19)
        self.assertAlmostEqual(b.pan, 18)

    def test_a_jump_far_away_is_a_new_person(self):
        tr = Tracker()
        tr.update([s(0)], 0.0)
        new, _ = tr.update([s(40)], 0.1)
        self.assertEqual(len(new), 1)

    def test_track_ends_after_missing_for_a_while(self):
        tr = Tracker(max_missing_s=0.8)
        first = tr.update([s(0)], 0.0)[0][0]
        self.assertEqual(tr.update([], 0.5), ([], []))              # a short miss (blink, turned head) is fine
        self.assertEqual(tr.update([], 0.9)[1], [first])
        self.assertEqual(tr.tracks, [])

    def test_a_real_face_survives_a_short_miss_a_thing_does_not(self):
        tr = Tracker(max_missing_s=0.8, face_max_missing_s=2.5)
        face, thing = tr.update([s(0), s(40)], 0.0)[0]
        face.face_q = 0.8                                           # passed the big-stream check
        self.assertEqual(tr.update([], 1.5)[1], [thing])            # 1.5 s unseen: the thing is gone...
        self.assertEqual(tr.update([s(0.5)], 2.0), ([], []))        # ...the face comes back as the same track
        self.assertEqual(tr.tracks, [face])
        self.assertEqual(tr.update([], 4.6)[1], [face])             # gone for 2.6 s: ended

    def test_remembers_how_far_a_track_moved(self):
        tr = Tracker()
        t = tr.update([s(0)], 0.0)[0][0]
        for k in range(1, 5):
            tr.update([s(k * 0.5)], k * 0.1)
        self.assertEqual(t.pos0, (0, 10.0))
        self.assertAlmostEqual(t.max_dev, 2.0)

    def test_closer_face_gates_wider(self):
        tr = Tracker()
        tr.update([s(0, size=20)], 0.0)                              # close: 20 deg wide
        new, _ = tr.update([s(15, size=20)], 0.1)
        self.assertEqual(new, [])

    def test_a_moving_head_widens_the_gate(self):
        tr = Tracker()
        tr.update([s(0)], 0.0)
        self.assertEqual(len(tr.update([s(10)], 0.1)[0]), 1)       # still head: 10 deg off is someone else
        tr = Tracker()
        tr.update([s(0)], 0.0)
        self.assertEqual(tr.update([s(10)], 0.1, slack_deg=6)[0], [])   # moving head: the pose is unsure


class VotingTest(unittest.TestCase):
    def setUp(self):
        self.tr = Tracker(n_votes=3, match_sim=0.3, embed_every_s=0.4, recheck_s=3.0)
        self.t = self.tr.update([s(0)], 0.0)[0][0]

    def test_majority_decides_the_person(self):
        self.assertFalse(self.tr.vote(self.t, "selim", 0.35, 0.0))
        self.assertFalse(self.tr.vote(self.t, None, 0.0, 0.4))
        self.assertTrue(self.tr.vote(self.t, "selim", 0.33, 0.8))
        self.assertEqual(self.t.person, "selim")

    def test_very_confident_first_look_decides_at_once(self):
        self.assertTrue(self.tr.vote(self.t, "selim", 0.6, 0.0))
        self.assertEqual(self.t.person, "selim")

    def test_matching_nobody_means_unknown(self):
        self.tr.vote(self.t, "selim", 0.1, 0.0)                      # below threshold = nobody
        self.assertFalse(self.tr.vote(self.t, None, 0.0, 0.4))      # 2 of 3 is not enough for 'unknown'
        self.assertTrue(self.tr.vote(self.t, "ayse", 0.2, 0.8))
        self.assertTrue(self.t.unknown)
        self.assertIsNone(self.t.person)

    def test_a_recognised_track_never_turns_into_a_stranger(self):
        self.tr.vote(self.t, "selim", 0.9, 0.0)
        for k in range(1, 5):
            self.tr.vote(self.t, None, 0.0, k * 0.4)                # bad light for a while
        self.assertEqual(self.t.person, "selim")
        self.assertFalse(self.t.unknown)

    def test_rename_after_a_merge(self):
        for k, pid in enumerate(["m1", "m2", "m2"]):
            self.tr.vote(self.t, pid, 0.4, k * 0.4)
        self.tr.rename("m2", "m1")
        self.assertEqual(self.t.person, "m1")
        self.assertNotIn("m2", self.t.votes)
        self.assertEqual(self.t.seen_as["m1"], 3)

    def test_one_good_look_among_bad_ones_is_not_unknown(self):
        self.tr.vote(self.t, None, 0.0, 0.0)
        self.tr.vote(self.t, None, 0.0, 0.4)
        self.tr.vote(self.t, "selim", 0.32, 0.8)                     # a known face, briefly badly lit
        self.assertFalse(self.t.unknown)

    def test_embedding_rhythm(self):
        self.assertTrue(self.tr.wants_embedding(self.t, 0.0))
        self.tr.vote(self.t, None, 0.0, 0.0)
        self.tr.update([s(0)], 0.2)
        self.assertFalse(self.tr.wants_embedding(self.t, 0.2))      # undecided: wait 0.4 s
        self.tr.update([s(0)], 0.4)
        self.assertTrue(self.tr.wants_embedding(self.t, 0.4))
        self.tr.vote(self.t, "selim", 0.9, 0.4)                      # decided: re-check only every 3 s
        self.tr.update([s(0)], 1.0)
        self.assertFalse(self.tr.wants_embedding(self.t, 1.0))
        self.tr.update([s(0)], 3.5)
        self.assertTrue(self.tr.wants_embedding(self.t, 3.5))

    def test_a_track_matching_two_ids_is_evidence_they_are_one_person(self):
        for k, pid in enumerate(["m1", "m2", "m1", "m2", "m2"]):
            self.tr.vote(self.t, pid, 0.4, k * 0.4)
        self.assertEqual(Tracker.same_person_evidence(self.t), [("m1", "m2")])

    def test_one_stray_match_is_not_evidence(self):
        for k, pid in enumerate(["m1", "m1", "m2"]):
            self.tr.vote(self.t, pid, 0.4, k * 0.4)
        self.assertEqual(Tracker.same_person_evidence(self.t), [])

    def test_two_decided_people_in_one_frame_are_apart(self):
        other = self.tr.update([s(0), s(30)], 0.1)[0][0]
        self.tr.vote(self.t, "selim", 0.9, 0.1)
        self.tr.vote(other, "ayse", 0.9, 0.1)
        self.assertEqual(self.tr.seen_together(0.1), [("ayse", "selim")])

    def test_one_confident_look_does_not_overturn_a_decision(self):
        self.tr.vote(self.t, "selim", 0.9, 0.0)
        self.tr.vote(self.t, "m2", 0.9, 3.0)
        self.assertEqual(self.t.person, "selim")
        self.tr.vote(self.t, "m2", 0.9, 3.4)                        # but a majority does
        self.assertEqual(self.t.person, "m2")

    def test_a_look_that_disagrees_brings_the_next_look_sooner(self):
        self.tr.vote(self.t, "selim", 0.9, 0.0)                      # decided
        for k in range(1, 35):
            self.tr.update([s(0)], k * 0.1)                         # the face stays in view
        self.tr.vote(self.t, "m2", 0.4, 3.0)                        # a recheck names someone else
        self.tr.update([s(0)], 3.5)
        self.assertTrue(self.tr.wants_embedding(self.t, 3.5))       # not 3 s later: now

    def test_not_embedded_when_not_seen_this_frame(self):
        self.tr.update([], 0.5)
        self.assertFalse(self.tr.wants_embedding(self.t, 0.5))


if __name__ == "__main__":
    unittest.main()


class GateTest(unittest.TestCase):
    """A huge false box must not swallow a small face far away; size changes have a limit."""

    def test_a_big_false_box_does_not_steal_a_small_track(self):
        tr = Tracker()
        tr.update([Sighting(None, 47.4, 17.7, 3.0)], 0.0)
        new, _ = tr.update([Sighting(None, 20.2, 9.8, 61.9)], 0.07)   # 28 deg away, 62 deg wide (2026-11-01)
        self.assertEqual(len(new), 1)
        self.assertAlmostEqual(tr.tracks[0].pan, 47.4)

    def test_size_jump_starts_a_new_track(self):
        tr = Tracker()
        tr.update([Sighting(None, 10.0, 0.0, 4.0)], 0.0)
        new, _ = tr.update([Sighting(None, 10.5, 0.0, 9.0)], 0.07)     # same place, twice as big
        self.assertEqual(len(new), 1)
        new, _ = tr.update([Sighting(None, 10.6, 0.0, 4.4)], 0.14)     # a normal frame-to-frame change
        self.assertEqual(new, [])

    def test_still_needs_a_still_head_and_time(self):
        tr = Tracker()
        for i in range(200):                                            # 20 s at one place, head moving
            tr.update([Sighting(None, 30.0, 5.0, 3.0)], i * 0.1, still=False)
        self.assertFalse(tr.tracks[0].still)
        for i in range(200, 320):                                       # 12 s with a still head
            tr.update([Sighting(None, 30.0 + 0.1 * (i % 3), 5.0, 3.0)], i * 0.1)
        self.assertTrue(tr.tracks[0].still)
        self.assertFalse(tr.tracks[0].moved)
