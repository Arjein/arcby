"""robot/faces.py: geometry, detector decoding and embedder input, with fake ONNX sessions."""
import unittest

import numpy as np

from robot import faces
from robot.faces import ARCFACE_KPS, Embedder, ScrfdDetector, YunetDetector


class _Input:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: returns fixed outputs, remembers the last input."""

    def __init__(self, outputs, name="input"):
        self.outputs = outputs
        self.name = name
        self.fed = None

    def get_inputs(self):
        return [_Input(self.name)]

    def run(self, names, feed):
        self.fed = feed[self.name]
        return self.outputs


def scrfd_outputs(size, hits):
    """SCRFD-shaped outputs (scores x3, box distances x3, landmarks x3) with hits = list of
    (stride, cell x, cell y, anchor, score, distances (l, t, r, b), kps offsets (10,)) in stride units."""
    w, h = size
    outs = {}
    for s in (8, 16, 32):
        n = (w // s) * (h // s) * 2
        outs[s] = [np.zeros((n, 1), np.float32), np.zeros((n, 4), np.float32), np.zeros((n, 10), np.float32)]
    for s, cx, cy, a, score, dist, kps in hits:
        i = (cy * (w // s) + cx) * 2 + a
        outs[s][0][i, 0] = score
        outs[s][1][i] = dist
        outs[s][2][i] = kps
    return [outs[s][k] for k in range(3) for s in (8, 16, 32)]


def yunet_outputs(size, hits):
    """YuNet-shaped outputs (cls, obj, bbox, kps; x3 strides, batch dim 1) with
    hits = list of (stride, cell x, cell y, cls, obj, bbox (dx, dy, log w, log h), kps (10,))."""
    w, h = size
    outs = {}
    for s in (8, 16, 32):
        n = (w // s) * (h // s)
        outs[s] = [np.zeros((1, n, 1), np.float32), np.zeros((1, n, 1), np.float32),
                   np.zeros((1, n, 4), np.float32), np.zeros((1, n, 10), np.float32)]
    for s, cx, cy, cls, obj, bbox, kps in hits:
        i = cy * (w // s) + cx
        outs[s][0][0, i, 0], outs[s][1][0, i, 0] = cls, obj
        outs[s][2][0, i] = bbox
        outs[s][3][0, i] = kps
    return [outs[s][k] for k in range(4) for s in (8, 16, 32)]


class GeometryTest(unittest.TestCase):
    def test_fit_shrinks_keeps_shape_and_pads(self):
        rgb = np.full((360, 640, 3), 200, np.uint8)
        x, scale = faces.fit(rgb, (320, 192))
        self.assertEqual(x.shape, (192, 320, 3))
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual(x[179, 319, 0], 200)       # last image row (360 * 0.5 = 180 rows)
        self.assertEqual(x[180, 0, 0], 0)           # padding below

    def test_nms_drops_overlaps_keeps_separate(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
        scores = np.array([0.8, 0.9, 0.7], np.float32)
        self.assertEqual(faces.nms(boxes, scores), [1, 2])

    def test_grid_rows_then_columns(self):
        g = faces.grid(32, 16, 8, per_cell=2)
        self.assertEqual(len(g), 4 * 2 * 2)
        self.assertEqual(g[0].tolist(), [0, 0])
        self.assertEqual(g[2].tolist(), [8, 0])     # next cell in the row
        self.assertEqual(g[8].tolist(), [0, 8])     # second row

    def test_similarity_transform_recovers_scale_rotation_shift(self):
        a = np.radians(20)
        m_true = np.array([[2 * np.cos(a), -2 * np.sin(a), 30], [2 * np.sin(a), 2 * np.cos(a), -5]])
        src = ARCFACE_KPS
        dst = src @ m_true[:, :2].T + m_true[:, 2]
        np.testing.assert_allclose(faces.similarity_transform(src, dst), m_true, atol=1e-9)

    def test_similarity_transform_never_mirrors(self):
        src = ARCFACE_KPS
        dst = src * [-1, 1]                         # left-right mirror image
        m = faces.similarity_transform(src, dst)
        self.assertGreater(np.linalg.det(m[:, :2]), 0)

    def test_align_puts_landmarks_on_the_template(self):
        rgb = np.zeros((300, 300, 3), np.uint8)
        kps = ARCFACE_KPS * 2 + [60, 40]            # a face twice the template size
        nx, ny = np.round(kps[2]).astype(int)
        rgb[ny - 3:ny + 4, nx - 3:nx + 4] = 255     # bright nose tip
        out = faces.align(rgb, kps)
        self.assertEqual(out.shape, (112, 112, 3))
        w = out[:, :, 0].astype(float)
        ys, xs = np.mgrid[:112, :112]
        x, y = (xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()      # middle of the bright spot
        self.assertLessEqual(abs(x - ARCFACE_KPS[2, 0]), 1.0)
        self.assertLessEqual(abs(y - ARCFACE_KPS[2, 1]), 1.0)


class EvenLightTest(unittest.TestCase):
    def test_dark_face_is_stretched_keeping_colour_ratios(self):
        rng = np.random.default_rng(0)
        face = (rng.random((112, 112, 1)) * 40 + 10) * np.array([1.0, 0.8, 0.6])   # dark, warm
        out = faces.even_light(face.astype(np.uint8)).astype(float)
        luma = out[16:-16, 16:-16] @ [0.299, 0.587, 0.114]
        self.assertGreater(luma.max(), 200)
        self.assertLess(luma.min(), 40)
        r, b = out[30:80, 30:80, 0].mean(), out[30:80, 30:80, 2].mean()
        self.assertGreater(r, b)                    # still warm: same gain on every colour

    def test_blank_crop_unchanged(self):
        face = np.full((112, 112, 3), 90, np.uint8)
        np.testing.assert_array_equal(faces.even_light(face), face)


class ScrfdTest(unittest.TestCase):
    size = (320, 192)

    def test_decodes_a_face_back_to_frame_pixels(self):
        kps = np.array([1, 0, 3, 0, 2, 1, 1, 2, 3, 2], np.float32)
        outs = scrfd_outputs(self.size, [(8, 5, 3, 1, 0.9, (2, 3, 4, 5), kps)])
        det = ScrfdDetector(FakeSession(outs), size=self.size)
        found = det.detect(np.zeros((384, 640, 3), np.uint8))       # frame 2x the input
        self.assertEqual(len(found), 1)
        f = found[0]
        # cell (5, 3) at stride 8 = (40, 24); box = centre -+ distances x 8; x2 back to frame pixels
        self.assertEqual(f.box, (48.0, 0.0, 144.0, 128.0))
        self.assertAlmostEqual(f.score, 0.9, places=5)
        np.testing.assert_allclose(f.kps[0], [(40 + 8) * 2, 24 * 2])
        np.testing.assert_allclose(f.kps[4], [(40 + 24) * 2, (24 + 16) * 2])

    def test_input_is_rgb_scaled_around_zero(self):
        sess = FakeSession(scrfd_outputs(self.size, []))
        rgb = np.zeros((192, 320, 3), np.uint8)
        rgb[..., 0] = 255
        ScrfdDetector(sess, size=self.size).detect(rgb)
        self.assertEqual(sess.fed.shape, (1, 3, 192, 320))
        self.assertAlmostEqual(float(sess.fed[0, 0, 0, 0]), (255 - 127.5) / 128)   # R first
        self.assertAlmostEqual(float(sess.fed[0, 2, 0, 0]), -127.5 / 128)

    def test_weak_and_tiny_faces_dropped(self):
        outs = scrfd_outputs(self.size, [(8, 5, 3, 0, 0.3, (2, 2, 2, 2), np.zeros(10)),     # weak
                                         (8, 20, 10, 0, 0.9, (0.3, 0.3, 0.3, 0.3), np.zeros(10))])  # 5 px
        self.assertEqual(ScrfdDetector(FakeSession(outs), size=self.size).detect(
            np.zeros((192, 320, 3), np.uint8)), [])

    def test_size_must_fit_the_strides(self):
        with self.assertRaises(ValueError):
            ScrfdDetector(FakeSession([]), size=(300, 192))


class YunetTest(unittest.TestCase):
    size = (320, 192)

    def test_decodes_a_face(self):
        kps = np.array([0.5, 0.5, 1.5, 0.5, 1, 1, 0.5, 1.5, 1.5, 1.5], np.float32)
        bbox = (0.5, 0.25, np.log(4), np.log(6))   # centre offset in cells, size = exp() x stride
        outs = yunet_outputs(self.size, [(16, 4, 2, 0.81, 1.0, bbox, kps)])
        f = YunetDetector(FakeSession(outs), size=self.size).detect(np.zeros((192, 320, 3), np.uint8))[0]
        # cell (4, 2) at stride 16 = (64, 32); centre (72, 36), size 64 x 96
        self.assertEqual(f.box, (40.0, 0.0, 104.0, 84.0))           # y1 = 36 - 48 clipped to 0
        self.assertAlmostEqual(f.score, 0.9, places=5)              # sqrt(cls x obj)
        np.testing.assert_allclose(f.kps[2], [64 + 16, 32 + 16])

    def test_input_is_bgr_raw(self):
        sess = FakeSession(yunet_outputs(self.size, []))
        rgb = np.zeros((192, 320, 3), np.uint8)
        rgb[..., 0] = 200
        YunetDetector(sess, size=self.size).detect(rgb)
        self.assertEqual(float(sess.fed[0, 2, 0, 0]), 200)          # R ends up last
        self.assertEqual(float(sess.fed[0, 0, 0, 0]), 0)


class EmbedderTest(unittest.TestCase):
    def test_output_is_a_unit_vector(self):
        sess = FakeSession([np.array([[3.0, 4.0]], np.float32)], name="data")
        v = Embedder(sess).embed(np.zeros((112, 112, 3), np.uint8))
        np.testing.assert_allclose(v, [0.6, 0.8], rtol=1e-6)

    def test_input_normalisation_and_channel_order(self):
        sess = FakeSession([np.ones((1, 4), np.float32)])
        face = np.zeros((112, 112, 3), np.uint8)
        face[..., 0] = 255
        Embedder(sess, bgr=True, mean=127.5, std=127.5).embed(face)
        self.assertEqual(sess.fed.shape, (1, 3, 112, 112))
        self.assertAlmostEqual(float(sess.fed[0, 2, 0, 0]), 1.0)    # R moved to the last channel
        self.assertAlmostEqual(float(sess.fed[0, 0, 0, 0]), -1.0)


class RegistryTest(unittest.TestCase):
    def test_every_model_is_usable(self):
        for name, m in faces.MODELS.items():
            if m["kind"] == "detector":
                self.assertIn(name, faces.DETECTORS)
            elif m["kind"] == "embedder":
                self.assertIn(name, faces.EMBED_INPUT)
            else:
                self.assertIn(m["kind"], ("vad", "voice", "pose", "objects"))  # vad.py, voices.py, body.py, show.py

    def test_config_choices_exist(self):
        from robot import config
        self.assertEqual(faces.MODELS[config.FACE_DETECTOR]["kind"], "detector")
        self.assertEqual(faces.MODELS[config.FACE_EMBEDDER]["kind"], "embedder")


if __name__ == "__main__":
    unittest.main()
