"""robot/camera.py: geometry, YUV conversion, face exposure loop, and the camera with a fake picamera2."""
import math
import unittest

import numpy as np

from robot import camera, config
from robot.camera import Camera, FaceExposure


class FakeRequest:
    def __init__(self, cam):
        self.cam = cam
        self.released = False

    def make_array(self, name):
        return self.cam.arrays[name]

    def release(self):
        self.released = True


class FakePicamera2:
    """The parts of Picamera2 that robot/camera.py uses."""

    def __init__(self, main=config.CAMERA_MAIN, lores=config.CAMERA_LORES):
        w, h = lores
        self.arrays = {"main": np.zeros((main[1], main[0], 3), np.uint8),
                       "lores": np.full((h * 3 // 2, w), 128, np.uint8)}
        self.controls = {}
        self.config = None
        self.requests = []
        self.started = self.closed = False

    def create_video_configuration(self, **kw):
        return kw

    def configure(self, cfg):
        self.config = cfg

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def set_controls(self, c):
        self.controls.update(c)

    def capture_request(self):
        r = FakeRequest(self)
        self.requests.append(r)
        return r


def yuv(y, u, v, w=8, h=4):
    """A flat-colour I420 buffer."""
    buf = np.empty((h * 3 // 2, w), np.uint8)
    buf[:h] = y
    chroma = np.empty((w // 2) * (h // 2) * 2, np.uint8)
    q = (w // 2) * (h // 2)
    chroma[:q], chroma[q:] = u, v
    buf[h:] = chroma.reshape(h // 2, w)
    return buf


class GeometryTest(unittest.TestCase):
    def test_middle_is_straight_ahead_and_edge_is_half_the_view(self):
        self.assertEqual(camera.angles_of(288, 162, 576, 324), (0.0, 0.0))
        ax, _ = camera.angles_of(576, 162, 576, 324)
        self.assertAlmostEqual(ax, config.CAMERA_HFOV_DEG / 2)

    def test_up_and_right_signs(self):
        ax, ay = camera.angles_of(400, 50, 576, 324)
        self.assertGreater(ax, 0)
        self.assertGreater(ay, 0)

    def test_head_target_turns_toward_the_pixel(self):
        # face right of centre and above: head must turn RIGHT (pan down) and UP (tilt up)
        pan, tilt = camera.head_target(400, 100, 576, 324, (10.0, 5.0))
        self.assertLess(pan, 10.0)
        self.assertGreater(tilt, 5.0)

    def test_size_deg(self):
        self.assertAlmostEqual(camera.size_deg(576, 576), config.CAMERA_HFOV_DEG)


class PixelTest(unittest.TestCase):
    def test_greys(self):
        self.assertTrue((camera.yuv420_to_rgb(yuv(16, 128, 128), 8, 4) == 0).all())
        self.assertTrue((camera.yuv420_to_rgb(yuv(235, 128, 128), 8, 4) >= 254).all())

    def test_red_comes_out_red(self):
        rgb = camera.yuv420_to_rgb(yuv(63, 102, 240), 8, 4)      # Rec.709 limited red
        r, g, b = rgb[0, 0].astype(int)
        self.assertGreater(r, 200)
        self.assertLess(g, 40)
        self.assertLess(b, 40)

    def test_mean_luma_uses_the_middle_of_the_box(self):
        y = np.full((100, 100), 20.0)
        y[40:61, 40:61] = 200.0                                  # bright face middle, dark hair / background
        self.assertGreater(camera.mean_luma(y, (30, 30, 70, 70)), 150)


def room(seed=1, margin=64):
    """A synthetic room (rectangles = furniture, frames, doors; slight blur): its spectrum, to render shifted views."""
    rng = np.random.default_rng(seed)
    w, h = config.CAMERA_LORES
    hb, wb = h + 2 * margin, w + 2 * margin
    big = np.full((hb, wb), 100.0)
    for _ in range(80):
        x, y = rng.integers(0, wb), rng.integers(0, hb)
        big[y:y + rng.integers(8, 90), x:x + rng.integers(8, 90)] = rng.uniform(30, 220)
    ky, kx = np.fft.fftfreq(hb)[:, None], np.fft.fftfreq(wb)[None, :]
    return np.fft.fft2(big) * np.exp(-(kx ** 2 + ky ** 2) / (2 * 0.25 ** 2)), kx, ky, margin


def view(scene, pan, tilt, noise=2.0, rng=None):
    """The lores Y plane the camera sees with its TRUE pose (pan, tilt): content moves right as it pans left (+)."""
    spec, kx, ky, m = scene
    k = math.degrees(1.0 / camera.focal_px(config.CAMERA_LORES[0]))
    img = np.real(np.fft.ifft2(spec * np.exp(-2j * np.pi * (kx * pan / k + ky * tilt / k))))
    w, h = config.CAMERA_LORES
    img = img[m:m + h, m:m + w]
    if rng is not None:
        img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class BackgroundTest(unittest.TestCase):
    """camera.Background: the camera's true pose from the picture (servo jitter, 2026-11-01)."""

    def test_measures_the_servo_error_against_the_room(self):
        scene, rng = room(), np.random.default_rng(0)
        bg = camera.Background(every=1)
        self.assertEqual(bg.update(view(scene, 0, 0, rng=rng), (0, 0), 0.0), (0.0, 0.0))     # the baseline
        for i, (err_p, err_t) in enumerate([(0.8, -0.5), (-1.3, 1.1), (0.2, 0.1), (1.5, -1.4)]):
            corr = bg.update(view(scene, err_p, err_t, rng=rng), (0, 0), 0.1 * (i + 1))
            self.assertAlmostEqual(corr[0], err_p, delta=0.05)
            self.assertAlmostEqual(corr[1], err_t, delta=0.05)

    def test_references_agree_when_the_head_looks_elsewhere_and_back(self):
        scene, rng = room(), np.random.default_rng(1)
        bg = camera.Background(every=1)
        bg.update(view(scene, 0, 0, rng=rng), (0, 0), 0.0)
        for i, (cmd, err) in enumerate([((6, 0), (0.4, 0.3)), ((12, 3), (-0.6, 0.2)), ((0, 0), (0.7, -0.9)),
                                        ((12, 3), (0.1, 0.5))]):
            true = (cmd[0] + err[0], cmd[1] + err[1])
            corr = bg.update(view(scene, *true, rng=rng), cmd, 1.0 + i)
            self.assertAlmostEqual(corr[0], err[0], delta=0.08, msg=cmd)
            self.assertAlmostEqual(corr[1], err[1], delta=0.08, msg=cmd)

    def test_a_changed_scene_is_not_a_measurement(self):
        bg = camera.Background(every=1)
        bg.update(view(room(1), 0, 0), (0, 0), 0.0)
        self.assertIsNone(bg.update(view(room(2), 0, 0), (0, 0), 0.1))      # lights / furniture: new reference
        self.assertIsNotNone(bg.update(view(room(2), 0.3, 0), (0, 0), 0.2))

    def test_measures_every_nth_frame_only(self):
        scene = room()
        bg = camera.Background(every=2)
        out = [bg.update(view(scene, 0, 0), (0, 0), 0.1 * i) for i in range(6)]
        self.assertEqual([o is None for o in out], [True, False] * 3)


class FaceExposureTest(unittest.TestCase):
    def test_dark_face_brightens_bright_face_darkens(self):
        fe = FaceExposure()
        self.assertGreater(fe.update(30, 0.0), 0)                # 2 stops under -> +1 EV (gain 0.5)
        fe = FaceExposure()
        self.assertLess(fe.update(235, 0.0), 0)

    def test_converges_and_then_holds(self):
        fe = FaceExposure()
        luma = 40.0
        t = 0.0
        for _ in range(40):
            new = fe.update(luma, t)
            if new is not None:
                luma = 40.0 * 2 ** new                           # the scene answers the EV change
            t += 0.3
        self.assertAlmostEqual(luma, config.FACE_EXPOSURE_TARGET, delta=15)
        self.assertIsNone(fe.update(luma, t))                    # close enough: no more changes

    def test_rate_limited(self):
        fe = FaceExposure()
        fe.update(30, 0.0)
        self.assertIsNone(fe.update(30, 0.1))

    def test_ev_stays_in_range(self):
        fe = FaceExposure()
        t = 0.0
        for _ in range(50):
            fe.update(5, t)
            t += 0.3
        self.assertEqual(fe.ev, config.FACE_EXPOSURE_EV[1])

    def test_no_winding_up_when_the_sensor_gives_everything(self):
        """Night: longest exposure + top gain. A dark face must not push EV to the rail, no squint cycling."""
        dark = {"ExposureTime": 66001, "AnalogueGain": 16.0}
        fe = FaceExposure()
        t = 0.0
        for _ in range(20):
            fe.update(20, t, dark)
            t += 0.3
        self.assertEqual(fe.ev, 0.0)
        self.assertLess(fe.update(235, t, dark), 0)              # a lamp on: darkening still works
        fe, evs, t = FaceExposure(), [], 0.0
        while t < 12.0:
            if fe.update(None, t, dark) is not None:
                evs.append(t)
            t += 0.1
        self.assertEqual(evs, [])
        self.assertTrue(camera.saturated(dark))
        self.assertFalse(camera.saturated({"ExposureTime": 20000, "AnalogueGain": 16.0}))
        self.assertFalse(camera.saturated(None))

    def test_squints_when_no_face_then_back(self):
        fe = FaceExposure()
        evs = []
        t = 0.0
        while t < 12.0:
            new = fe.update(None, t)
            if new is not None:
                evs.append((round(t, 1), new))
            t += 0.1
        self.assertEqual(evs[0][1], config.FACE_SQUINT_EV)       # first squint after FACE_SQUINT_AFTER_S
        self.assertAlmostEqual(evs[0][0], config.FACE_SQUINT_AFTER_S, delta=0.11)
        self.assertEqual(evs[1][1], 0.0)                         # back after FACE_SQUINT_S
        self.assertGreaterEqual(len(evs), 4)                     # and again, and again

    def test_face_found_while_squinting_takes_over(self):
        fe = FaceExposure()
        t = 0.0
        while fe.ev == 0.0:
            fe.update(None, t)
            t += 0.1
        new = fe.update(60, t + 0.3)                             # found a face thanks to the squint
        self.assertIsNotNone(new)
        self.assertGreater(new, 0)
        self.assertIsNone(fe.squinting_since)
        for k in range(10):                                      # the face stays: no squint cycling
            fe.update(config.FACE_EXPOSURE_TARGET, t + 0.6 + k * 0.3)
        self.assertGreater(fe.ev, 0.5)

    def test_face_gone_returns_to_normal_after_a_while(self):
        fe = FaceExposure()
        fe.update(30, 0.0)
        self.assertGreater(fe.ev, 0)
        self.assertIsNone(fe.update(None, 1.0))                  # just lost: keep the exposure


class CameraTest(unittest.TestCase):
    def test_configures_two_streams_full_view(self):
        fake = FakePicamera2()
        Camera(picam=fake).start()
        self.assertEqual(fake.config["main"]["size"], config.CAMERA_MAIN)
        self.assertEqual(fake.config["lores"]["format"], "YUV420")
        self.assertEqual(fake.config["sensor"]["output_size"], config.CAMERA_MAIN)
        self.assertTrue(fake.started)
        self.assertIn("AeMeteringMode", fake.controls)

    def test_frame_releases_the_buffer(self):
        fake = FakePicamera2()
        cam = Camera(picam=fake, clock=lambda: 10.0).start()
        with cam.frame() as f:
            self.assertAlmostEqual(f.t, 10.0 - config.CAMERA_LATENCY_S)
            self.assertEqual(f.lores_rgb().shape, (config.CAMERA_LORES[1], config.CAMERA_LORES[0], 3))
        self.assertTrue(fake.requests[-1].released)
        with self.assertRaises(ZeroDivisionError):
            with cam.frame():
                1 / 0
        self.assertTrue(fake.requests[-1].released)              # released on errors too

    def test_frame_age_from_sensor_timestamp(self):
        class Req:
            def __init__(self, md):
                self.md = md

            def get_metadata(self):
                return self.md
        now = 5_000_000_000
        # readout started 30 ms ago, 66 ms exposure: the middle of the exposure was 63 ms ago
        self.assertAlmostEqual(camera.frame_age(Req({"SensorTimestamp": now - 30_000_000, "ExposureTime": 66000}),
                                                now), 0.063)
        self.assertEqual(camera.frame_age(Req({}), now), config.CAMERA_LATENCY_S)
        self.assertEqual(camera.frame_age(Req({"SensorTimestamp": now + 10**9}), now), config.CAMERA_LATENCY_S)

    def test_crop_main_maps_lores_box_to_big_pixels(self):
        fake = FakePicamera2()
        fake.arrays["main"][400:440, 800:840] = 255              # a marker in the big stream
        cam = Camera(picam=fake).start()
        s = config.CAMERA_MAIN[0] / config.CAMERA_LORES[0]
        with cam.frame() as f:
            crop, (x0, y0, scale) = f.crop_main((800 / s, 400 / s, 840 / s, 440 / s), pad=0.5)
        self.assertEqual(scale, s)
        self.assertEqual(crop[400 - y0 + 5, 800 - x0 + 5, 0], 255)
        self.assertEqual(crop.shape, (80 + 1, 80 + 1, 3))           # 40 px box + 50 % on each side

    def test_set_ev_and_close(self):
        fake = FakePicamera2()
        cam = Camera(picam=fake).start()
        cam.set_ev(1.5)
        self.assertEqual(fake.controls["ExposureValue"], 1.5)
        cam.close()
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
