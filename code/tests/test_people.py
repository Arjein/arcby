"""robot/people.py: the people database on disk."""
import os
import tempfile
import unittest

import numpy as np

from robot.people import PeopleStore
from tests.fakes import FakeClock


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def near(v, noise, rng):
    """A vector of the same 'person' seen again: v plus a little noise."""
    return unit(v + rng.normal(0, noise, v.shape))


class PeopleTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.clock = FakeClock(1000.0)
        self.rng = np.random.default_rng(0)
        self.a = unit(self.rng.normal(size=64))
        self.b = unit(self.rng.normal(size=64))

    def tearDown(self):
        self.dir.cleanup()

    def store(self, **kw):
        return PeopleStore(self.dir.name, clock=self.clock, **kw)

    def test_empty_store_matches_nobody(self):
        self.assertEqual(self.store().match(self.a), (None, 0.0))

    def test_enrol_then_match_the_right_person(self):
        s = self.store()
        pa = s.enrol([self.a])
        pb = s.enrol([self.b])
        who, sim = s.match(near(self.a, 0.05, self.rng))
        self.assertEqual(who, pa)
        self.assertGreater(sim, 0.9)
        self.assertEqual(s.match(self.b)[0], pb)

    def test_names_and_data_survive_a_restart(self):
        s = self.store()
        pid = s.enrol([self.a, near(self.a, 0.1, self.rng)], photo=np.zeros((20, 20, 3), np.uint8))
        s.set_name(pid, "Selim")
        again = self.store()
        self.assertEqual(again.name(pid), "Selim")
        self.assertEqual(again.match(self.a)[0], pid)
        self.assertEqual(again.people(), [(pid, "Selim", 2)])
        self.assertTrue(os.path.exists(os.path.join(self.dir.name, pid, "face.jpg")))

    def test_how_to_address_someone(self):
        s = self.store()
        pid = s.enrol([self.a])
        s.set_name(pid, "Selim")
        self.assertEqual(s.call(pid), "Selim")                       # by name unless asked otherwise
        s.set_call(pid, "Boss")
        self.assertEqual((self.store().call(pid), self.store().name(pid)), ("Boss", "Selim"))
        s.set_call(pid, None)
        self.assertEqual(s.call(pid), "Selim")

    def test_learn_keeps_new_angles_only(self):
        s = self.store(learn_min=0.4, learn_max=0.85)
        pid = s.enrol([self.a])
        self.assertFalse(s.learn(pid, self.a))                      # copy of a stored vector
        self.assertFalse(s.learn(pid, self.b))                      # not this person (low similarity)
        other_angle = unit(self.a + 0.6 * self.b)                   # same person, quite different view
        sim = float(self.a @ other_angle)
        self.assertTrue(0.4 < sim < 0.85)
        self.assertTrue(s.learn(pid, other_angle))
        self.assertEqual(len(s.vectors[pid]), 2)

    def test_learn_drops_the_most_redundant_when_full(self):
        s = self.store(max_vectors=3, learn_min=-1, learn_max=0.999)
        pid = s.enrol([self.a])
        for v in (unit(self.a + 0.01 * self.b), self.b, unit(self.rng.normal(size=64))):
            s.learn(pid, v, sim=1.0)
        m = s.vectors[pid]
        self.assertEqual(len(m), 3)
        g = m @ m.T
        np.fill_diagonal(g, 0)
        self.assertLess(g.max(), 0.99)                              # one of the two near-twins was dropped

    def test_forget_deletes_everything(self):
        s = self.store()
        pid = s.enrol([self.a], photo=np.zeros((8, 8, 3), np.uint8))
        s.forget(pid)
        self.assertEqual(s.match(self.a), (None, 0.0))
        self.assertFalse(os.path.exists(os.path.join(self.dir.name, pid)))
        self.assertEqual(self.store().people(), [])

    def test_merge_two_ids_of_one_person(self):
        s = self.store()
        p1 = s.enrol([self.a])
        p2 = s.enrol([near(self.a, 0.3, self.rng)])
        s.set_name(p2, "Selim")
        s.merge(p1, p2)
        self.assertEqual(s.name(p1), "Selim")
        self.assertEqual(len(s.vectors[p1]), 2)
        self.assertEqual([p for p, _, _ in s.people()], [p1])
        self.assertEqual(self.store().origins[p1], [p1, p2])     # provenance survives a restart

    def test_seen_counts_and_times(self):
        s = self.store()
        pid = s.enrol([self.a])
        self.clock.t = 2000.0
        s.seen(pid)
        self.assertEqual(s.index[pid]["seen"], 2)
        self.assertEqual(s.index[pid]["last_seen"], 2000.0)


class MergeTest(unittest.TestCase):
    """One person enrolled twice: evidence, candidates, merge and undo."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.clock = FakeClock(1000.0)
        self.rng = np.random.default_rng(1)
        self.s = PeopleStore(self.dir.name, clock=self.clock, match_sim=0.3)
        self.selim = unit(self.rng.normal(size=64))
        self.other = unit(self.rng.normal(size=64))
        self.m1 = self.s.enrol([near(self.selim, 0.08, self.rng) for _ in range(3)])
        self.clock.t += 10
        self.m2 = self.s.enrol([near(self.selim, 0.08, self.rng) for _ in range(3)])   # Selim again, bad light
        self.clock.t += 10
        self.o = self.s.enrol([self.other])

    def tearDown(self):
        self.dir.cleanup()

    def test_set_similarity_same_person_high_strangers_low(self):
        self.assertGreater(self.s.set_similarity(self.m1, self.m2), 0.4)
        self.assertLess(self.s.set_similarity(self.m1, self.o), 0.3)

    def test_similar_sets_alone_mean_ask_not_auto(self):
        c = self.s.merge_candidates(ask_sim=0.4)
        self.assertEqual([(x["keep"], x["drop"], x["auto"]) for x in c], [(self.m1, self.m2, False)])

    def test_track_evidence_makes_it_automatic(self):
        self.s.link(self.m1, self.m2)
        self.s.link(self.m2, self.m1)
        c = self.s.merge_candidates(auto_links=2)[0]
        self.assertTrue(c["auto"])
        self.assertEqual(c["links"], 2)

    def test_same_name_and_alike_is_automatic(self):
        self.s.set_name(self.m1, "Selim")
        self.s.set_name(self.m2, "Selim")
        self.assertTrue(self.s.merge_candidates()[0]["auto"])

    def test_seen_together_never_merged(self):
        self.s.link(self.m1, self.m2, n=5)
        self.assertTrue(self.s.apart(self.m1, self.m2))
        self.assertFalse(self.s.apart(self.m2, self.m1))            # already known
        self.assertEqual(self.s.merge_candidates(), [])
        with self.assertRaises(ValueError):
            self.s.merge(self.m1, self.m2)

    def test_keeps_the_named_id(self):
        self.s.set_name(self.m2, "Selim")
        c = self.s.merge_candidates(ask_sim=0.4)[0]
        self.assertEqual((c["keep"], c["drop"]), (self.m2, self.m1))

    def test_merge_moves_evidence_to_the_kept_id(self):
        self.s.apart(self.m2, self.o)                               # m2 was seen next to o
        self.s.merge(self.m1, self.m2)
        self.assertIn("|".join(sorted((self.m1, self.o))), self.s.apart_pairs)

    def test_unmerge_restores_both_and_marks_them_apart(self):
        before1 = self.s.vectors[self.m1].copy()
        before2 = self.s.vectors[self.m2].copy()
        self.s.set_name(self.m2, "Ayse")
        self.s.merge(self.m1, self.m2)
        self.s.unmerge(self.m1, self.m2)
        np.testing.assert_array_equal(self.s.vectors[self.m1], before1)
        np.testing.assert_array_equal(self.s.vectors[self.m2], before2)
        self.assertEqual(self.s.name(self.m2), "Ayse")
        self.assertEqual(self.s.merge_candidates(), [])             # never suggested again
        again = PeopleStore(self.dir.name, clock=self.clock)
        self.assertEqual(again.name(self.m2), "Ayse")

    def test_unmerge_brings_back_ids_merged_into_the_dropped_one(self):
        m3 = self.s.enrol([near(self.selim, 0.08, self.rng)])
        self.s.merge(self.m2, m3)                                   # m3 -> m2, then m2 -> m1
        self.s.merge(self.m1, self.m2)
        self.s.unmerge(self.m1, self.m2)
        self.assertEqual(sorted(set(self.s.origins[self.m2])), sorted({self.m2, m3}))
        self.assertEqual(set(self.s.origins[self.m1]), {self.m1})

    def test_forget_clears_evidence(self):
        self.s.link(self.m1, self.m2)
        self.s.apart(self.m1, self.o)
        self.s.forget(self.m1)
        self.assertEqual(self.s.links, {})
        self.assertEqual(self.s.apart_pairs, set())



if __name__ == "__main__":
    unittest.main()


class RobustStoreTest(unittest.TestCase):
    """A power cut must never stop the robot from starting; a merge racing a forget must not crash vision."""

    def test_a_broken_index_falls_back_to_the_backup(self):
        with tempfile.TemporaryDirectory() as d:
            s = PeopleStore(d)
            pid = s.enrol([np.ones(4, np.float32) / 2])
            s.set_name(pid, "Selim")                                 # second write: the first one is the .bak
            with open(os.path.join(d, "index.json"), "w") as f:
                f.write('{"people": {"x')                           # cut off mid-write
            s2 = PeopleStore(d)
            self.assertIn(pid, s2.index)

    def test_no_index_at_all_starts_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "index.json"), "w") as f:
                f.write("")
            self.assertEqual(PeopleStore(d).people(), [])

    def test_merge_of_a_forgotten_id_is_refused_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            s = PeopleStore(d)
            a = s.enrol([np.ones(4, np.float32) / 2])
            b = s.enrol([np.ones(4, np.float32) / 2])
            s.forget(b)
            with self.assertRaises(KeyError):
                s.merge(a, b)
            self.assertIn(a, s.index)
