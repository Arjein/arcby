"""hwtest/mouthshots.py analysis on made-up data: two faces, the one the schedule names talks (its mouth moves with
the sound), the other looks and stays quiet."""
import unittest

import numpy as np

from hwtest.mouthshots import CHUNK_S, analyse_arrays, mouth_box, sentences, tracks_of


def scene(seconds=120.0, fps=13.0, phase=30.0, seed=0):
    rng = np.random.default_rng(seed)
    meta = dict(t_start=100.0, phase_s=phase, first="left", lores=[576, 324])
    frame_t = 100.0 + np.arange(0, seconds, 1 / fps)
    chunk_t = 100.0 + np.arange(0, seconds, CHUNK_S)

    def talking(t):                         # 1.5 s sentences with 1 s pauses
        return ((t - 100.0) % 2.5) < 1.5

    def who(t):
        return "left" if int((t - 100.0) // phase) % 2 == 0 else "right"
    pcm = (rng.normal(size=len(chunk_t) * 512) * 1e-4).astype(np.float32)        # the room
    for i, t in enumerate(chunk_t):
        if talking(t):
            pcm[i * 512:(i + 1) * 512] += np.sin(np.arange(512) * 0.2).astype(np.float32) * 0.05
    rows, mouths = [], []
    base = {s: rng.integers(0, 255, (32, 48)).astype(np.float32) for s in ("left", "right")}
    for i, t in enumerate(frame_t):
        for s, x in (("left", 150.0), ("right", 420.0)):
            m = base[s] + rng.normal(size=(32, 48)) * 4                       # camera noise
            if talking(t) and who(t) == s:
                m = m + rng.normal(size=(32, 48)) * 60                      # lips moving
            rows.append([i, x - 30, 120, x + 30, 190] + [0.0] * 10 + [0.9])
            mouths.append(np.clip(m, 0, 255).astype(np.uint8))
    return frame_t, np.asarray(rows, np.float32), np.asarray(mouths), pcm, chunk_t, meta


class MouthShotsTest(unittest.TestCase):
    def test_the_moving_mouth_is_the_speaker(self):
        res = analyse_arrays(*scene(), log=lambda *a: None)
        self.assertGreater(len(res), 20)
        self.assertEqual(sum(r["ratio"] == r["truth"] for r in res), len(res))
        self.assertGreater(sum(r["corr"] == r["truth"] for r in res), 0.8 * len(res))

    def test_the_robots_own_voice_is_no_sentence(self):
        frame_t, rows, mouths, pcm, chunk_t, meta = scene()
        n = len(analyse_arrays(frame_t, rows, mouths, pcm, chunk_t, meta, log=lambda *a: None))
        meta["prompts"] = [(105.0, 106.6), (110.0, 111.6)]     # it spoke over two sentences
        self.assertEqual(len(analyse_arrays(frame_t, rows, mouths, pcm, chunk_t, meta, log=lambda *a: None)), n - 2)

    def test_two_faces_stay_two_tracks(self):
        frame_t, rows, *_ = scene(seconds=10)
        ids = tracks_of(frame_t, rows)
        self.assertEqual(len(set(ids)), 2)

    def test_sentences_from_the_loudness(self):
        t = np.arange(0, 10, CHUNK_S)
        db = np.where((t % 2.5) < 1.5, -40.0, -80.0)
        found, _ = sentences(t, db)
        self.assertEqual(len(found), 4)

    def test_mouth_box_is_around_the_corners(self):
        x1, y1, x2, y2 = mouth_box(np.array([[0, 0], [0, 0], [0, 0], [40, 100], [60, 100]]), 60)
        self.assertTrue(x1 < 40 and x2 > 60 and y1 < 100 < y2)


if __name__ == "__main__":
    unittest.main()
