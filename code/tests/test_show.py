"""Showing things to the robot (robot/show.py): a hand held up + a thing in it + looking = sure; asking = sure."""
import unittest
from types import SimpleNamespace

import numpy as np

from robot import body as B
from robot import config
from robot import show as S

from tests.test_body import W, H, person, standing

PHONE = S.COCO.index("cell phone")
CHAIR = S.COCO.index("chair")


def holding(cx=288, top=40, dx=0.0, dy=0.0):
    """A standing person holding the right hand up in front of the chest (wrist above the elbow)."""
    b = standing(cx=cx, top=top)
    b.kps[B.R_ELBOW, :2] = (cx - 40, top + 110)
    b.kps[B.R_WRIST, :2] = (cx - 15 + dx, top + 80 + dy)
    return b


class Frame:
    def __init__(self, t):
        self.t = t

    def lores_rgb(self):
        return np.zeros((H, W, 3), np.uint8)

    def crop_main(self, box, pad=0.4):
        s = 4.0
        x1, y1 = max(0, int(box[0] * s)), max(0, int(box[1] * s))
        x2, y2 = min(W * 4, int(box[2] * s)), min(H * 4, int(box[3] * s))
        return np.full((max(1, y2 - y1), max(1, x2 - x1), 3), 128, np.uint8), (x1, y1, s)


class FakeObjects:
    """Finds `cls` in the middle of every crop it gets (or nothing)."""

    def __init__(self, cls=PHONE, score=0.8):
        self.cls, self.score, self.runs = cls, score, 0

    def detect(self, rgb):
        self.runs += 1
        if self.cls is None:
            return []
        h, w = rgb.shape[:2]
        return [S.Thing(np.array([w * 0.4, h * 0.4, w * 0.6, h * 0.6], np.float32), self.score, self.cls)]


def face(looking=True, person="selim", tid=5):
    return SimpleNamespace(id=tid, looking=looking, person=person, pan=0.0, tilt=10.0,
                           sighting=SimpleNamespace(size_deg=3.0))


class Scene:
    """15 fps for `seconds`; the pose model sees body_at(t) every frame; the face on the body is `face`."""

    def __init__(self, objects, body_at, faces=(face(),), seconds=2.0):
        self.tracker = B.BodyTracker()
        self.show = S.ShowSense.inline_for_tests(objects)
        self.events = []
        t = 0.0
        while t < seconds:
            self.tracker.update([body_at(t)], t, W, H, (0.0, 0.0))
            for tr in self.tracker.tracks:
                tr.face = faces[0].id if faces else None
                tr.person = faces[0].person if faces else None
            self.events += self.show.process(Frame(t), lambda _t: (0.0, 0.0), self.tracker.visible(t), list(faces))
            self.t = t
            t += 1 / 15


class HeldUpTest(unittest.TestCase):
    def wrists(self, b):
        tr = B.BodyTracker()
        tr.update([b], 0.0, W, H, (0.0, 0.0))
        return [w for w, _ in S.held_up(tr.tracks[0])]

    def test_a_hand_in_front_of_the_chest_is_held_up(self):
        self.assertEqual(self.wrists(holding()), [B.R_WRIST])

    def test_arms_hanging_down_are_not(self):
        self.assertEqual(self.wrists(standing()), [])

    def test_a_hand_far_out_to_the_side_is_not(self):
        b = holding()
        b.kps[B.R_WRIST, :2] = (288 - 200, 120)
        b.kps[B.R_ELBOW, :2] = (288 - 150, 150)
        self.assertEqual(self.wrists(b), [])

    def test_no_shoulders_no_idea(self):
        b = holding()
        b.kps[B.L_SHOULDER, 2] = 0.0
        self.assertEqual(self.wrists(b), [])


class ShowTest(unittest.TestCase):
    def test_held_up_still_with_a_phone_while_looking_is_sure(self):
        sc = Scene(FakeObjects(), lambda t: holding())
        shown = [e for e in sc.events if e["kind"] == "showing"]
        self.assertEqual(len(shown), 1)                         # once, not every frame
        self.assertEqual(shown[0]["what"], "cell phone")
        self.assertEqual(shown[0]["person"], "selim")
        self.assertFalse(shown[0]["asked"])
        pan, tilt, what = sc.show.target(sc.t)
        self.assertEqual(what, "cell phone")
        jpeg, what = sc.show.picture(t=sc.t)
        self.assertEqual(jpeg[:2], b"\xff\xd8")                 # a JPEG for the mind
        self.assertEqual(what, "cell phone")

    def test_waving_is_not_showing(self):
        sc = Scene(FakeObjects(), lambda t: holding(dx=30 * np.sin(2 * np.pi * 3 * t)))
        self.assertEqual([e for e in sc.events if e["kind"] == "showing"], [])

    def test_not_looking_at_the_robot_is_not_showing(self):          # using the phone themselves
        sc = Scene(FakeObjects(), lambda t: holding(), faces=(face(looking=False),))
        self.assertEqual([e for e in sc.events if e["kind"] == "showing"], [])

    def test_a_hand_without_a_thing_or_with_furniture_is_not_showing(self):
        for objects in (FakeObjects(cls=None), FakeObjects(cls=CHAIR), FakeObjects(score=0.3)):
            sc = Scene(objects, lambda t: holding())
            self.assertEqual([e for e in sc.events if e["kind"] == "showing"], [], objects.cls)
            self.assertGreater(objects.runs, 0)                 # it did look

    def test_no_object_runs_while_no_hand_is_held_up(self):
        objects = FakeObjects()
        Scene(objects, lambda t: standing())
        self.assertEqual(objects.runs, 0)

    def test_asked_is_sure_even_without_an_object(self):
        tracker = B.BodyTracker()
        show = S.ShowSense.inline_for_tests(FakeObjects(cls=None))
        tracker.update([holding()], 0.0, W, H, (0.0, 0.0))
        tracker.tracks[0].person = "selim"
        show.ask(0.0, person="selim")
        ev = show.process(Frame(0.0), lambda _t: (0.0, 0.0), tracker.visible(0.0), [face()])
        (e,) = [x for x in ev if x["kind"] == "showing"]
        self.assertTrue(e["asked"])
        self.assertIsNone(e["what"])
        pan, tilt, _ = show.target(0.1)
        wrist = tracker.tracks[0].kang[B.R_WRIST]
        self.assertAlmostEqual(pan, wrist[0], places=3)         # it looks at the raised hand
        self.assertIsNotNone(show.picture(t=0.1))
        self.assertEqual(show.process(Frame(0.1), lambda _t: (0.0, 0.0), tracker.visible(0.1), [face()]), [])

    def test_asked_without_a_body_looks_under_the_face(self):
        show = S.ShowSense.inline_for_tests(FakeObjects(cls=None))
        show.ask(0.0, person="selim")
        (e,) = show.process(Frame(0.0), lambda _t: (0.0, 0.0), [], [face()])
        self.assertLess(e["tilt"], 10.0)

    def test_ask_words(self):
        for text in ("Bak!", "Bu ne?", "Şuna bak", "Bunu gördün mü?", "Ne bu?", "What is this?"):
            self.assertTrue(S.asked_to_look(text), text)
        for text in ("Bakkala gittim", "Bugün ne yapalım?", "Saat kaç?"):
            self.assertFalse(S.asked_to_look(text), text)


class DecodeObjectsTest(unittest.TestCase):
    def test_raw_layout_with_nms(self):
        out = np.zeros((1, 84, 3), np.float32)
        for i, (cx, score) in enumerate(((100, 0.9), (102, 0.7), (200, 0.6))):
            out[0, :4, i] = (cx, 50, 20, 40)
            out[0, 4 + PHONE, i] = score
        things = S.decode_objects(out, 1.0, (0, 0))
        self.assertEqual(sorted(round(float(t.box[0])) for t in things), [90, 190])   # the overlap merged
        self.assertTrue(all(t.name == "cell phone" for t in things))

    def test_end_to_end_layout(self):
        out = np.array([[[10, 20, 30, 40, 0.8, PHONE], [0, 0, 5, 5, 0.1, 0]]], np.float32)
        (t,) = S.decode_objects(out, 0.5, (2, 4))
        np.testing.assert_allclose(t.box, [16, 32, 56, 72])
        self.assertEqual(t.name, "cell phone")


class PixelTest(unittest.TestCase):
    def test_pixel_of_inverts_angles(self):
        for xy, pose in (((100, 50), (0, 0)), ((400, 250), (20, -10))):
            ang = B.angles([xy], W, H, pose)[0]
            x, y = S.pixel_of(ang, W, H, pose)
            self.assertAlmostEqual(x, xy[0], places=3)
            self.assertAlmostEqual(y, xy[1], places=3)


class VocabularyModelTest(unittest.TestCase):
    """YOLOE with the home word list (the showing session's winner): its own names, its own threshold."""
    NAMES = ("phone", "sheet of paper", "book")

    def test_yoloe_layout_with_mask_coefficients(self):
        out = np.zeros((1, 4 + 3 + 32, 2), np.float32)
        out[0, :4, 0] = (100, 50, 20, 40)
        out[0, 4 + 1, 0] = 0.3                              # "sheet of paper" at 0.3
        out[0, 4 + 3:, 0] = 5.0                             # mask coefficients: never read as a class
        out[0, :4, 1] = (200, 50, 20, 40)
        out[0, 4 + 0, 1] = 0.2                              # a phone at 0.2: under YOLOE's 0.25
        (t,) = S.decode_objects(out, 1.0, (0, 0), min_score=0.25, names=self.NAMES)
        self.assertEqual(t.name, "sheet of paper")

    def test_the_first_model_there_is_used(self):
        import tempfile, os
        from robot import faces as F
        d = tempfile.mkdtemp()
        self.assertIsNone(S.load_objects(d))                # none there: only asking
        open(os.path.join(d, F.MODELS["yolo26n"]["file"]), "wb").close()
        calls = []
        orig = F._session
        F._session = lambda path, threads: calls.append(path) or SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="images")])
        try:
            det = S.load_objects(d)
            self.assertEqual((det.name, det.names, det.min_score), ("yolo26n", None, None))
            open(os.path.join(d, F.MODELS["yoloe_home"]["file"]), "wb").close()
            det = S.load_objects(d)                         # the session's winner wins when it is there
            self.assertEqual(det.name, "yoloe_home")
            self.assertEqual(det.min_score, 0.25)
            self.assertIn("sheet of paper", det.names)
            self.assertNotIn("glasses", det.names)          # it fired on Selim's own glasses
        finally:
            F._session = orig

    def test_the_upper_body_box_holds_head_arms_and_hands(self):
        tr = B.BodyTracker()
        tr.update([holding()], 0.0, W, H, (0.0, 0.0))
        box = S.upper_body(tr.tracks[0], (W, H), (0.0, 0.0), 60.0)
        x1, y1, x2, y2 = box
        b = holding()
        for i in (B.NOSE, B.R_WRIST, B.L_ELBOW):
            self.assertTrue(x1 <= b.kps[i, 0] <= x2 and y1 <= b.kps[i, 1] <= y2, i)
        self.assertLess(y2, b.kps[B.L_ANKLE, 1])            # not the legs


if __name__ == "__main__":
    unittest.main()
