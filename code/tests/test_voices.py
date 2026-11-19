"""robot/voices.py: fbank features, the voice-print store (identify, learn, forget, merge)."""
import tempfile
import unittest

import numpy as np

from robot import voices
from robot.voices import VoiceStore


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


class FbankTest(unittest.TestCase):
    def test_shape_and_mean_normalised(self):
        rng = np.random.default_rng(0)
        f = voices.fbank(rng.normal(0, 0.1, 16000).astype(np.float32))
        self.assertEqual(f.shape, (98, 80))                 # 1 s: 1 + (16000 - 400) // 160 frames
        np.testing.assert_allclose(f.mean(axis=0), 0, atol=1e-4)

    def test_a_tone_lights_its_mel_band(self):
        t = np.arange(16000) / 16000
        f = voices.fbank_raw((0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
        mel = 1127 * np.log1p(1000 / 700)
        edges = np.linspace(1127 * np.log1p(20 / 700), 1127 * np.log1p(8000 / 700), 82)
        expect = int(np.argmin(abs(edges[1:-1] - mel)))
        self.assertLessEqual(abs(int(f.mean(axis=0).argmax()) - expect), 1)

    def test_loudness_does_not_matter(self):              # the mic's auto gain changes the level all the time
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.05, 16000).astype(np.float32)
        np.testing.assert_allclose(voices.fbank(x), voices.fbank(4 * x), atol=1e-3)


class VoiceStoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.rng = np.random.default_rng(0)
        self.selim_c, self.emre_c = unit(self.rng.normal(size=16)), unit(self.rng.normal(size=16))

    def tearDown(self):
        self.dir.cleanup()

    def like(self, centre, noise=0.6):
        return unit(centre + noise * unit(self.rng.normal(size=16)))

    def store(self, **kw):
        return VoiceStore(self.dir.name, **kw)

    def test_consistency_finds_the_odd_print_out(self):
        s = self.store(alike=0.999)
        for _ in range(5):
            s.learn("selim", self.like(self.selim_c, noise=0.3))
        s.learn("selim", self.like(self.emre_c, noise=0.3))       # one sentence was really Emre's
        c = s.consistency("selim")
        self.assertEqual(int(np.argmin(c)), 5)
        self.assertEqual(len(s.consistency("nobody")), 0)

    def test_learns_identifies_and_keeps_it_on_disk(self):
        s = self.store(match_sim=0.5, margin=0.08, alike=0.99)
        for _ in range(5):
            s.learn("selim", self.like(self.selim_c))
            s.learn("emre", self.like(self.emre_c))
        s2 = self.store(match_sim=0.5, margin=0.08)
        self.assertEqual(s2.identify(self.like(self.selim_c))[0], "selim")
        self.assertEqual(s2.identify(self.like(self.emre_c))[0], "emre")
        self.assertIsNone(s2.identify(unit(self.rng.normal(size=16)))[0])      # a stranger

    def test_too_close_to_call_is_nobody(self):
        s = self.store(match_sim=0.3, margin=0.08, alike=0.99)
        s.learn("selim", self.selim_c)
        s.learn("emre", unit(self.selim_c + 0.01 * self.emre_c))               # nearly the same voice
        self.assertIsNone(s.identify(self.selim_c)[0])

    def test_only_among_the_people_it_still_knows(self):
        s = self.store(match_sim=0.5, alike=0.99)
        s.learn("old", self.selim_c)
        self.assertIsNone(s.identify(self.selim_c, among={"selim"})[0])

    def test_alike_prints_add_nothing_and_old_ones_go(self):
        s = self.store(keep=3, alike=0.95)
        self.assertTrue(s.learn("selim", self.selim_c))
        self.assertFalse(s.learn("selim", self.selim_c))
        for _ in range(5):
            s.learn("selim", self.like(self.selim_c, 2.0))
        self.assertEqual(s.count("selim"), 3)

    def test_forget_and_merge(self):
        s = self.store(alike=0.99)
        s.learn("a", self.selim_c)
        s.learn("b", self.like(self.selim_c))
        s.merge("a", "b")
        self.assertEqual((s.count("a"), s.count("b")), (2, 0))
        s.forget("a")
        self.assertEqual(self.store().count("a"), 0)

    def test_one_person_only_needs_a_clear_match(self):          # Emre must not pass for Selim
        s = self.store(match_sim=0.5, margin=0.08, alike=0.99, solo_sim=0.6)
        s.learn("selim", self.selim_c)
        half = unit(self.selim_c + 1.0 * unit(self.emre_c - (self.emre_c @ self.selim_c) * self.selim_c))
        pid, sim = s.identify(half)                             # ~0.7 alike with only Selim enrolled... ok
        self.assertEqual(pid, "selim")
        mid = unit(self.selim_c + 1.5 * unit(self.emre_c - (self.emre_c @ self.selim_c) * self.selim_c))
        pid, sim = s.identify(mid)                              # ~0.55: with nobody to compare, not enough
        self.assertTrue(0.5 <= sim < 0.6)
        self.assertIsNone(pid)

    def test_each_print_keeps_when_it_was_learnt(self):
        now = [1000.0]
        s = self.store(alike=0.99, clock=lambda: now[0])
        s.learn("selim", self.selim_c)
        now[0] = 2000.0
        s.learn("selim", self.like(self.selim_c))
        s.learn("emre", self.emre_c)
        s.merge("selim", "emre")
        again = self.store()
        self.assertEqual(sorted(again.prints), ["selim"])        # the times are not taken for a person
        self.assertEqual(list(again.times["selim"]), [1000.0, 2000.0, 2000.0])

    def test_no_model_no_embedder(self):
        self.assertIsNone(voices.load_voice_embedder(models_dir=self.dir.name))


if __name__ == "__main__":
    unittest.main()
