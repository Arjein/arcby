"""robot/body.py: pose model decoding, where a body's face should be, waving, body tracks, faces on bodies; the
seek / wave rules in robot/attention.py and wave answers in robot/greeter.py. No model file needed (fake outputs);
the real model is checked by test_real_model when its file is there."""
import os
import random
import tempfile
import unittest

import numpy as np

from robot import body as B
from robot import config
from robot.attention import Attention
from robot.tracker import Sighting, Track

W, H = 576, 324                 # the small stream


def person(box, kp=None, score=0.8):
    """A Body with the given keypoints {index: (x, y)} visible, the rest invisible."""
    kps = np.zeros((17, 3), np.float32)
    for i, (x, y) in (kp or {}).items():
        kps[i] = (x, y, 0.9)
    return B.Body(np.array(box, np.float32), score, kps)


def standing(cx=288, top=40, scale=1.0):
    """A whole person facing the camera, ~250 px tall, centred at cx."""
    s = scale
    return person([cx - 50 * s, top, cx + 50 * s, top + 250 * s], {
        B.NOSE: (cx, top + 20 * s), B.L_EYE: (cx + 6 * s, top + 15 * s), B.R_EYE: (cx - 6 * s, top + 15 * s),
        B.L_SHOULDER: (cx + 30 * s, top + 55 * s), B.R_SHOULDER: (cx - 30 * s, top + 55 * s),
        B.L_ELBOW: (cx + 40 * s, top + 95 * s), B.R_ELBOW: (cx - 40 * s, top + 95 * s),
        B.L_WRIST: (cx + 42 * s, top + 130 * s), B.R_WRIST: (cx - 42 * s, top + 130 * s),
        B.L_HIP: (cx + 20 * s, top + 135 * s), B.R_HIP: (cx - 20 * s, top + 135 * s),
        B.L_KNEE: (cx + 20 * s, top + 190 * s), B.R_KNEE: (cx - 20 * s, top + 190 * s),
        B.L_ANKLE: (cx + 20 * s, top + 245 * s), B.R_ANKLE: (cx - 20 * s, top + 245 * s)})


def legs_only(cx=288):
    """Up close from 14 cm: knees and ankles, cut off at the top of the picture."""
    return person([cx - 60, 0, cx + 60, 320], {B.L_KNEE: (cx + 30, 90), B.R_KNEE: (cx - 30, 90),
                                               B.L_ANKLE: (cx + 30, 300), B.R_ANKLE: (cx - 30, 300)})


def arm_at_right():
    """Only a hand / forearm reaching in from the right edge."""
    return person([470, 120, W, 220], {B.R_WRIST: (480, 160), B.R_ELBOW: (560, 190)})


def waving(t, up=True, x0=288):
    """A standing person with the right forearm up (or down) swinging 3x a second around the elbow."""
    b = standing(cx=x0)
    ex, ey = b.kps[B.R_ELBOW, :2]
    swing = 30 * np.sin(2 * np.pi * 3.0 * t)
    b.kps[B.R_WRIST, :2] = (ex + swing, ey - 40 if up else ey + 35)
    return b


class DecodeTest(unittest.TestCase):
    def test_letterbox_centres_the_frame_on_grey(self):
        x, s, (px, py) = B.letterbox(np.zeros((H, W, 3), np.uint8), (320, 192))
        self.assertEqual(x.shape, (1, 3, 192, 320))
        self.assertAlmostEqual(s, 320 / W)
        self.assertEqual((px, py), (0, 6))                  # 576x324 -> 320x180, 6 px of grey above and below
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), 114 / 255, places=5)
        self.assertEqual(float(x[0, 0, 100, 100]), 0.0)

    def test_decode_maps_back_to_frame_pixels_and_drops_overlaps(self):
        s, pad = 320 / W, (0, 6)
        out = np.zeros((1, 56, 4), np.float32)

        def put(i, cx, cy, w, h, score):
            out[0, :5, i] = (cx, cy, w, h, score)
            out[0, 5:56, i] = np.tile([cx, cy, 0.9], 17)

        put(0, 160, 96, 50, 100, 0.9)
        put(1, 162, 97, 50, 100, 0.6)                       # the same person again: dropped by NMS
        put(2, 60, 96, 30, 80, 0.5)
        put(3, 260, 96, 30, 80, 0.1)                        # too unsure
        found = B.decode(out, s, pad)
        self.assertEqual([round(b.score, 1) for b in found], [0.9, 0.5])
        np.testing.assert_allclose(found[0].box, [(135) / s, (46 - 6) / s, 185 / s, (146 - 6) / s], rtol=1e-5)
        np.testing.assert_allclose(found[0].kps[0, :2], [160 / s, 90 / s], rtol=1e-5)

    def test_nothing_found(self):
        self.assertEqual(B.decode(np.zeros((1, 56, 10), np.float32), 1.0, (0, 0)), [])


class FaceHintTest(unittest.TestCase):
    def test_head_keypoints_are_the_face(self):
        pan, tilt, how = B.face_hint(standing(), W, H, (0.0, 0.0))
        self.assertEqual(how, "head")
        self.assertAlmostEqual(pan, 0.0, delta=0.5)
        self.assertGreater(tilt, 20)                         # high up in the picture

    def test_legs_only_means_look_up(self):
        pose = (10.0, 5.0)
        pan, tilt, how = B.face_hint(legs_only(), W, H, pose)
        self.assertEqual(how, "knees")
        self.assertAlmostEqual(pan, 10.0, delta=0.5)
        top_of_picture = 5.0 + 67 / 2 * 0.9                  # roughly where the picture's top edge is
        self.assertGreater(tilt, top_of_picture)            # above what it can see now

    def test_cut_off_at_the_top_without_knees_still_looks_above(self):
        b = person([250, 0, 330, 200], {B.L_ANKLE: (300, 190)})
        pan, tilt, how = B.face_hint(b, W, H, (0.0, 0.0))
        self.assertEqual(how, "top")
        self.assertGreater(tilt, 30)

    def test_an_arm_at_the_right_edge_means_look_right(self):
        pan, tilt, how = B.face_hint(arm_at_right(), W, H, (0.0, 10.0))
        self.assertEqual((how, tilt), ("edge", None))       # no idea how high: attention uses the face height
        self.assertLess(pan, -51 - config.BODY_EDGE_DEG + 1)     # right = pan -, past the edge (HFOV 102 deg)

    def test_shoulders_without_a_head_look_above_them(self):
        b = standing()
        b.kps[list(B.HEAD), 2] = 0.0                         # turned away: no face keypoints
        _, tilt, how = B.face_hint(b, W, H, (0.0, 0.0))
        sh_tilt = B.angles([b.kps[B.L_SHOULDER, :2]], W, H, (0, 0))[0, 1]
        self.assertEqual(how, "shoulders")
        self.assertGreater(tilt, sh_tilt + config.BODY_HINT_MIN_UP_DEG - 0.1)

    def test_nothing_useful(self):
        self.assertIsNone(B.face_hint(person([200, 100, 300, 200]), W, H, (0, 0)))


class WaveTest(unittest.TestCase):
    def run_wave(self, make, hz=4.0, seconds=2.0):
        d = B.WaveDetector()
        return any(d.update(make(i / hz), i / hz) for i in range(int(seconds * hz)))

    def test_a_raised_forearm_swinging_is_a_wave(self):
        self.assertTrue(self.run_wave(lambda t: waving(t)))

    def test_too_few_samples_a_second_miss_a_fast_wave(self):
        # why WristWatch exists: a 3 Hz wave seen 3.3x a second looks like a slow drift
        self.assertFalse(self.run_wave(lambda t: waving(t), hz=3.3))

    def test_arm_down_swinging_is_not(self):
        self.assertFalse(self.run_wave(lambda t: waving(t, up=False)))

    def test_walking_past_with_an_arm_up_is_not(self):
        def walk(t):
            b = waving(0.0, x0=150 + 120 * t)                # whole body moves, the forearm does not swing
            return b
        self.assertFalse(self.run_wave(walk))

    def test_a_raised_still_hand_is_not(self):
        self.assertFalse(self.run_wave(lambda t: waving(0.0)))

    def test_a_forearm_on_a_desk_is_not(self):
        # 2026-11-01 PlayStation session: Selim seen from the side, forearm level on the desk (wrist 4 px above the
        # elbow, 35 px forward), the hand moving the controller - the motion watcher called it a wave
        def desk(t):
            b = standing()
            ex, ey = b.kps[B.R_ELBOW, :2]
            b.kps[B.R_WRIST, :2] = (ex + 35 + 10 * np.sin(2 * np.pi * 3.0 * t), ey - 4)
            return b
        self.assertFalse(self.run_wave(desk))
        self.assertIsNone(B.raised_wrist(desk(0.0)))
        self.assertIsNotNone(B.raised_wrist(waving(0.0)))


class MotionFrame:
    """A grey picture with a bright hand-sized square at x (and camera noise)."""

    def __init__(self, t, hand_x=None, hand_y=95, rng=None, size=14):
        self.t = t
        rng = rng or np.random.default_rng(int(t * 1000))
        y = np.full((H, W), 90, np.int16) + rng.integers(-6, 7, (H, W))
        if hand_x is not None:
            y[hand_y - size // 2:hand_y + size // 2, int(hand_x) - size // 2:int(hand_x) + size // 2] = 200
        self.y_plane = np.clip(y, 0, 255).astype(np.uint8)

    def lores_rgb(self):
        return np.repeat(self.y_plane[:, :, None], 3, axis=2)


class WristWatchTest(unittest.TestCase):
    """The pose model sees the arm up a few times a second; the patch round the wrist is watched every frame."""

    def sense(self, pose_every=4):
        # the model: a standing person with the right forearm up (wrist at x 246, y 105), every pose_every-th frame
        b = waving(0.0)
        return B.BodySense(FakePose(lambda i: [b]), inline=True), b

    def run_frames(self, hand, seconds=2.0, fps=15.0, speed=0.0):
        sense, b = self.sense()
        wx, wy = b.kps[B.R_WRIST, :2]
        events = []
        for i in range(int(seconds * fps)):
            t = i / fps
            f = MotionFrame(t, hand(t, wx), hand_y=int(wy) - 5)
            events += sense.process(f, lambda t: (0.0, 0.0), speed, [])
        return [e for e in events if e["kind"] == "waved"]

    def test_a_3hz_wave_is_seen_at_the_frame_rate(self):
        waved = self.run_frames(lambda t, wx: wx + 15 * np.sin(2 * np.pi * 3.0 * t))
        self.assertEqual(len(waved), 1)
        self.assertEqual(waved[0]["how"], "motion")

    def test_a_still_raised_hand_is_not(self):
        self.assertEqual(self.run_frames(lambda t, wx: wx), [])

    def test_camera_noise_alone_is_not(self):
        self.assertEqual(self.run_frames(lambda t, wx: None), [])

    def test_not_while_the_head_moves(self):
        self.assertEqual(self.run_frames(lambda t, wx: wx + 15 * np.sin(2 * np.pi * 3.0 * t),
                                         speed=config.VISION_STILL_DPS + 5), [])


class TrackerTest(unittest.TestCase):
    def test_follows_one_person_and_ends_the_track(self):
        tr = B.BodyTracker()
        new, _ = tr.update([standing(200)], 0.0, W, H, (0, 0))
        self.assertEqual(len(new), 1)
        new, _ = tr.update([standing(210)], 0.3, W, H, (0, 0))
        self.assertEqual((new, tr.tracks[0].hits), ([], 2))
        _, ended = tr.update([], 0.3 + config.BODY_MAX_MISSING_S + 0.1, W, H, (0, 0))
        self.assertEqual(len(ended), 1)

    def test_the_head_turning_does_not_break_a_track(self):
        tr = B.BodyTracker()
        tr.update([standing(288)], 0.0, W, H, (0, 0))
        # the head turned 20 deg left: the same person is now further right in the picture
        x = 288 + np.tan(np.radians(20)) * __import__("robot.camera", fromlist=["x"]).focal_px(W)
        new, _ = tr.update([standing(x)], 0.3, W, H, (20.0, 0))
        self.assertEqual(new, [])

    def test_out_of_view_a_moment_is_the_same_person(self):
        tr = B.BodyTracker()
        tr.update([standing(200)], 0.0, W, H, (0, 0))
        tr.update([], 1.0, W, H, (0, 0))
        new, _ = tr.update([standing(230)], 3.0, W, H, (0, 0))     # back 3 s later, a little further on
        self.assertEqual(new, [])
        _, ended = tr.update([], 3.0 + config.BODY_MAX_MISSING_S + 0.1, W, H, (0, 0))
        self.assertEqual(len(ended), 1)

    def test_legs_then_legs_and_body_is_one_track(self):
        # the head tilts up: the box grows from legs only to the whole person, its centre jumps; the knees do not
        tr = B.BodyTracker()
        legs = legs_only(288)
        tr.update([legs], 0.0, W, H, (0, 0))
        whole = person([200, 0, 376, 324], {B.L_KNEE: (318, 90), B.R_KNEE: (258, 90),
                                            B.L_HIP: (310, 10), B.R_HIP: (266, 10)})
        new, _ = tr.update([whole], 0.3, W, H, (0, 0))
        self.assertEqual(new, [])

    def test_a_chair_that_never_moves_is_still(self):
        tr = B.BodyTracker()
        for i in range(40):
            tr.update([standing(288)], i * 0.3, W, H, (0, 0))
        self.assertTrue(tr.tracks[0].still)
        tr.tracks[0].face_t = 11.0                          # ...unless a face was on it (someone sitting still)
        self.assertFalse(tr.tracks[0].still)

    def test_given_up_body_rearms_once_it_moved(self):
        sense = B.BodySense(None)
        tr = B.BodyTracker()
        tr.update([standing(200), standing(450)], 0.0, W, H, (0, 0))
        tr.update([standing(200), standing(450)], 0.3, W, H, (0, 0))
        b = tr.tracks[0]
        sense.give_up(b)
        tr.update([standing(205), standing(450)], 0.6, W, H, (0, 0))
        self.assertTrue(b.given_up)                          # moved a little: still given up
        for i, x in enumerate((240, 280, 320)):
            tr.update([standing(x), standing(450)], 0.9 + 0.3 * i, W, H, (0, 0))
        self.assertFalse(b.given_up)


def face_track(tid, pan, tilt, size=4.0, person=None):
    return Track(tid, Sighting(None, pan, tilt, size), 0.0, 0.0, person=person)


class LinkTest(unittest.TestCase):
    def test_the_face_on_the_head_belongs_to_that_body(self):
        tr = B.BodyTracker()
        tr.update([standing(150), standing(420)], 0.0, W, H, (0, 0))
        left, right = tr.tracks
        hint = right.hint
        faces = [face_track(7, hint[0] + 1, hint[1]), face_track(8, 60.0, 0.0)]
        faces[0].person = "selim"
        B.link_faces(tr.tracks, faces, 0.0)
        self.assertEqual((left.face, right.face, right.person), (None, 7, "selim"))


class FakePose:
    def __init__(self, script):
        self.script, self.i = script, 0

    def detect(self, rgb):
        b = self.script(self.i)
        self.i += 1
        return b


class FakeFrame:
    def __init__(self, t):
        self.t = t

    def lores_rgb(self):
        return np.zeros((H, W, 3), np.uint8)


class SenseTest(unittest.TestCase):
    def test_waving_makes_one_event_until_it_starts_again(self):
        hz = 4.0
        sense = B.BodySense(FakePose(lambda i: [waving(i / hz)]), inline=True)
        events = []
        for i in range(int(6 * hz)):
            sense.next_t = -1                               # every frame
            events += sense.process(FakeFrame(i / hz), lambda t: (0.0, 0.0), 0.0, [])
        waved = [e for e in events if e["kind"] == "waved"]
        self.assertEqual(len(waved), 1)
        self.assertEqual(waved[0]["track"], None)           # no face track on it
        self.assertIsNotNone(waved[0]["tilt"])

    def test_seekable_skips_bodies_with_a_fresh_face(self):
        sense = B.BodySense(FakePose(lambda i: [standing(288)]), inline=True)
        for i in range(3):
            sense.next_t = -1
            sense.process(FakeFrame(i * 0.3), lambda t: (0.0, 0.0), 0.0, [])
        self.assertEqual(len(sense.seekable(0.6)), 1)
        bt = sense.tracker.tracks[0]
        face = face_track(3, bt.hint[0], bt.hint[1])
        sense.next_t = -1
        sense.process(FakeFrame(0.9), lambda t: (0.0, 0.0), 0.0, [face])
        self.assertEqual(sense.seekable(0.9), [])

    def test_runs_slower_while_every_body_has_its_face(self):
        sense = B.BodySense(FakePose(lambda i: [standing(288)]), inline=True)
        sense.process(FakeFrame(0.0), lambda t: (0.0, 0.0), 0.0, [])
        self.assertAlmostEqual(sense.next_t, config.BODY_EVERY_IDLE_S)    # nobody seen yet
        sense.next_t = -1
        sense.process(FakeFrame(0.2), lambda t: (0.0, 0.0), 0.0, [])
        self.assertAlmostEqual(sense.next_t, 0.2 + config.BODY_EVERY_S)   # a body without its face
        bt = sense.tracker.tracks[0]
        face = face_track(3, bt.hint[0], bt.hint[1])
        sense.next_t = -1
        sense.process(FakeFrame(0.5), lambda t: (0.0, 0.0), 0.0, [face])
        sense.next_t = -1
        sense.process(FakeFrame(0.6), lambda t: (0.0, 0.0), 0.0, [face])
        self.assertAlmostEqual(sense.next_t, 0.6 + config.BODY_EVERY_SLOW_S)

    def test_off_without_a_model(self):
        sense = B.BodySense(None)
        self.assertFalse(sense.enabled)
        self.assertEqual(sense.process(FakeFrame(0.0), lambda t: (0, 0), 0.0, []), [])

    @unittest.skipUnless(os.path.exists(os.path.expanduser(os.path.join(config.MODELS_DIR, "yolo26n-pose-320x192.onnx"))),
                         "pose model not downloaded")
    def test_real_model_runs(self):
        det = B.load_pose()
        found = det.detect(np.full((H, W, 3), 128, np.uint8))
        self.assertEqual(found, [])                          # a grey wall is nobody


class Seekable:
    """A body attention may look for a face on (BodyTrack's fields attention uses)."""

    def __init__(self, bid, pan, tilt, hint, h=40.0):
        self.id, self.pan, self.tilt, self.hint, self.height_deg = bid, pan, tilt, hint, h
        self.seek_t, self.given_up, self.given_up_at = None, False, None


class AttentionBodiesTest(unittest.TestCase):
    def setUp(self):
        self.a = Attention(rng=random.Random(0), hour=lambda: 14)

    def test_legs_in_view_look_up_for_the_face(self):
        legs = Seekable(1, 5.0, -10.0, (5.0, 40.0, "knees"))
        self.assertEqual(self.a.update([], 0.0, [legs]), (5.0, 40.0, "seek"))
        self.assertEqual(self.a.events[-1]["how"], "body")

    def test_the_face_showing_up_wins(self):
        legs = Seekable(1, 5.0, -10.0, (5.0, 40.0, "knees"))
        self.a.update([], 0.0, [legs])
        pan, tilt, mode = self.a.update([face_track(9, 6.0, 38.0)], 0.3, [legs])
        self.assertEqual((pan, tilt, mode), (6.0, 38.0, "follow"))

    def test_head_keypoints_are_followed_longer(self):
        turned = Seekable(1, 20.0, 0.0, (20.0, 40.0, "head"))
        self.a.update([], 0.0, [turned])
        self.assertEqual(self.a.update([], config.ATTN_SEEK_S + 1.0, [turned])[2], "seek")
        self.assertNotEqual(self.a.update([], config.ATTN_SEEK_HEAD_S + 0.1, [turned])[2], "seek")

    def test_gives_up_on_a_body_without_a_face(self):
        chair = Seekable(1, 20.0, 0.0, (20.0, 15.0, "shoulders"))
        self.a.update([], 0.0, [chair])
        _, _, mode = self.a.update([], config.ATTN_SEEK_S + 0.1, [chair])
        self.assertNotEqual(mode, "seek")
        self.assertTrue(chair.given_up)
        again = Seekable(2, 21.0, 0.5, (21.0, 15.0, "shoulders"))     # the same chair as a new track
        _, _, mode = self.a.update([], config.ATTN_SEEK_S + 0.5, [again])
        self.assertNotEqual(mode, "seek")

    def test_an_arm_at_the_edge_turns_it_at_face_height(self):
        arm = Seekable(1, -45.0, 5.0, (-75.0, None, "edge"))
        pan, tilt, mode = self.a.update([], 0.0, [arm])
        self.assertEqual((pan, mode), (-75.0, "seek"))
        self.assertEqual(tilt, config.ATTN_SCAN_TILT)       # nobody seen yet: the default face height

    def test_no_head_no_seeking(self):
        a = Attention(rng=random.Random(0), hour=lambda: 14, has_head=False)
        self.assertNotEqual(a.update([], 0.0, [Seekable(1, 5.0, -10.0, (5.0, 40.0, "knees"))])[2], "seek")

    def test_following_someone_ignores_other_bodies(self):
        selim = face_track(1, -10.0, 30.0, person="selim")
        self.a.update([selim], 0.0)
        _, _, mode = self.a.update([selim], 0.2, [Seekable(2, 40.0, 0.0, (40.0, 30.0, "shoulders"))])
        self.assertEqual(mode, "follow")

    def test_a_wave_turns_it_to_the_waver(self):
        selim = face_track(1, -10.0, 30.0, person="selim")
        self.a.update([selim], 0.0)
        self.a.wave(0.1, track=None, person=None, pan=40.0, tilt=25.0)
        pan, tilt, mode = self.a.update([selim], 0.2)
        self.assertEqual((pan, tilt, mode), (40.0, 25.0, "orient"))
        self.assertEqual(self.a.events[-1]["how"], "wave")

    def test_a_wave_from_a_face_in_view_follows_that_face(self):
        selim, emre = face_track(1, -10.0, 30.0, person="selim"), face_track(2, 30.0, 30.0, person="emre")
        self.a.update([selim, emre], 0.0)
        self.assertEqual(self.a.update([selim, emre], 0.1)[0], -10.0)
        self.a.wave(0.2, track=2, person="emre", pan=30.0, tilt=30.0)
        self.assertEqual(self.a.update([selim, emre], 0.3)[:2], (30.0, 30.0))


class GreeterWaveTest(unittest.TestCase):
    def setUp(self):
        from robot.greeter import Greeter
        from robot.people import PeopleStore
        from tests.test_greeter import FakeVoice
        self.dir = tempfile.TemporaryDirectory()
        self.store = PeopleStore(self.dir.name, clock=lambda: 0.0)
        self.selim = self.store.enrol([np.ones(8) / 3])
        self.store.set_name(self.selim, "Selim")
        self.voice = FakeVoice()
        self.now = [1000.0]
        self.g = Greeter(self.voice, self.store, clock=lambda: self.now[0], rng=random.Random(0))

    def tearDown(self):
        self.dir.cleanup()

    def test_known_waver_by_name_then_not_again_soon(self):
        said = self.g.update([], [dict(kind="waved", person=self.selim)], 0.0)
        self.assertEqual(len(said), 1)
        self.assertIn("Selim", said[0])
        self.now[0] += 5
        self.assertEqual(self.g.update([], [dict(kind="waved", person=self.selim)], 5.0), [])
        self.now[0] += config.GREET_WAVE_AGAIN_S
        self.assertEqual(len(self.g.update([], [dict(kind="waved", person=self.selim)], 40.0)), 1)

    def test_waits_for_the_name_of_a_face_being_recognised(self):
        class Face:
            def __init__(self, tid, person=None):
                self.id, self.person = tid, person
        self.assertEqual(self.g.update([Face(5)], [dict(kind="waved", person=None, track=5)], 0.0), [])
        self.now[0] += 0.5
        self.assertEqual(self.g.update([Face(5)], [], 0.5), [])                   # still deciding who it is
        self.now[0] += 0.5
        said = self.g.update([Face(5, self.selim)], [], 1.0)                       # recognised: by name
        self.assertEqual(len(said), 1)
        self.assertIn("Selim", said[0])

    def test_no_name_in_time_plain_hello(self):
        class Face:
            id, person = 5, None
        self.g.update([Face()], [dict(kind="waved", person=None, track=5)], 0.0)
        self.now[0] += config.GREET_WAVE_NAME_S + 0.1
        self.assertIn(self.g.update([Face()], [], 2.0)[0], ("Merhaba!", "Selam!"))

    def test_a_stranger_gets_a_plain_hello(self):
        self.assertIn(self.g.update([], [dict(kind="waved", person=None)], 0.0)[0], ("Merhaba!", "Selam!"))


class FarBodyCropTest(unittest.TestCase):
    """Far people: the model on a big-stream crop under a small face (robot shots 2026-10-31 at 3 m: whole frame
    0.37-0.70 confidence, crop 0.89-0.92)."""

    def test_the_crop_has_the_model_shape_holds_the_face_and_stays_inside(self):
        mw, mh = config.BODY_INPUT
        for box in ([280, 100, 292, 114], [2, 2, 14, 16], [560, 300, 574, 322], [100, 50, 160, 120]):
            x1, y1, x2, y2 = B.crop_for_face(box, (W, H))
            self.assertAlmostEqual((x2 - x1) / (y2 - y1), mw / mh, places=3)
            self.assertTrue(0 <= x1 < x2 <= W + 1e-6 and 0 <= y1 < y2 <= H + 1e-6)
            self.assertGreaterEqual(y2 - y1, min(config.BODY_CROP_MIN_PX, H) - 1e-6)
        x1, y1, x2, y2 = B.crop_for_face([280, 100, 292, 114], (W, H))
        self.assertTrue(x1 < 286 < x2 and y1 < 100 and y2 > 114 + 3 * 14)  # room for the body under the face

    def test_bodies_found_in_a_crop_come_back_in_small_stream_pixels(self):
        b = person([10, 20, 110, 220], {B.NOSE: (60, 40)})
        B.to_frame([b], (400, 200), 4.0)
        np.testing.assert_allclose(b.box, [102.5, 55, 127.5, 105])
        np.testing.assert_allclose(b.kps[B.NOSE, :2], [115, 60])

    def test_every_other_run_is_a_crop_under_a_far_face(self):
        seen = []

        class Pose:
            def detect(self, rgb):
                seen.append(rgb.shape)
                if rgb.shape[:2] == (H, W):
                    return []                           # the whole frame misses them (too small)
                h, w = rgb.shape[:2]                    # the crop finds them: head near its top centre
                return [person([w * 0.35, h * 0.05, w * 0.65, h * 0.98],
                               {B.NOSE: (w / 2, h * 0.12), B.L_SHOULDER: (w * 0.42, h * 0.3),
                                B.R_SHOULDER: (w * 0.58, h * 0.3)}, score=0.9)]

        class Frame(FakeFrame):
            def crop_main(self, box, pad=0.4):
                s = 4.0
                x1, y1, x2, y2 = (int(v * s) for v in box)
                return np.zeros((y2 - y1, x2 - x1, 3), np.uint8), (x1, y1, s)

        face = SimpleFace([280, 100, 292, 114])
        pan, tilt = B.angles([[286, 107]], W, H, (0.0, 0.0))[0]     # where vision puts that face
        ft = Track(5, Sighting(face, float(pan), float(tilt), 2.0), 0.0, 0.0)
        sense = B.BodySense(Pose(), inline=True)
        for i in range(4):
            sense.next_t = -1
            sense.process(Frame(i * 0.3), lambda t: (0.0, 0.0), 0.0, [ft])
        self.assertEqual([sh[:2] == (H, W) for sh in seen], [False, True, False, True])
        self.assertEqual(sense.stats["crops"], 2)
        (bt,) = sense.tracker.tracks
        self.assertEqual(bt.face, 5)                    # the body found in the crop is that face's
        self.assertTrue(bt.body.box[0] < 286 < bt.body.box[2])
        self.assertTrue(sense.last_size == (W, H))

    def test_near_faces_and_frames_without_a_big_stream_use_the_whole_frame(self):
        seen = []

        class Pose:
            def detect(self, rgb):
                seen.append(rgb.shape)
                return []
        near = Track(1, Sighting(SimpleFace([250, 80, 330, 170]), 0.0, 0.0, 14.0), 0.0, 0.0)
        far = Track(2, Sighting(SimpleFace([280, 100, 292, 114]), 0.0, 0.0, 2.0), 0.0, 0.0)
        sense = B.BodySense(Pose(), inline=True)
        sense.next_t = -1
        sense.process(FakeFrame(0.0), lambda t: (0.0, 0.0), 0.0, [far])    # no crop_main: whole frame
        sense.next_t = -1
        sense.process(FakeFrame(0.3), lambda t: (0.0, 0.0), 0.0, [near])
        self.assertEqual([sh[:2] for sh in seen], [(H, W), (H, W)])
        self.assertEqual(sense.far_faces([near, far, face_track(3, 0, 0)]), [far])


class SimpleFace:
    def __init__(self, box):
        self.box = np.array(box, np.float64)


if __name__ == "__main__":
    unittest.main()
