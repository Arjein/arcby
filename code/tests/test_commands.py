"""robot/commands.py: the dashboard's changes to what the robot knows (People tab), through request files."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from robot import commands as C
from robot.people import PeopleStore
from robot.voices import VoiceStore


def colour_vec(rgb):
    v = np.array([*rgb, 1.0, 0, 0, 0, 0], np.float32) + 0.01
    return v / np.linalg.norm(v)


class ColourEmbedder:
    """A face picture of one colour -> a vector of that colour (the same picture = the same vector)."""

    def embed(self, img):
        return colour_vec(tuple(float(c) / 255 for c in img.reshape(-1, 3).mean(0)))


class World:
    def __init__(self, root):
        self.root = Path(root)
        self.people = self.root / "people"
        self.data = self.root / "data"
        self.store = PeopleStore(str(self.people), clock=lambda: 0.0)
        self.voices = VoiceStore(str(self.people))
        self.selim = self.store.enrol([colour_vec((1, 0, 0))])
        self.store.set_name(self.selim, "Selim")
        self.other = self.store.enrol([colour_vec((0, 1, 0)), colour_vec((0, 0, 1))])
        self.voices.learn(self.selim, colour_vec((1, 0, 0)))
        self.voices.learn(self.other, colour_vec((0, 1, 0)))
        faces = self.data / "2026-11-02" / "120000" / "faces"
        (faces / "2026-11-02").mkdir(parents=True)
        lines = []
        for i, (rgb, person, match, sim) in enumerate((((255, 0, 0), self.selim, self.selim, 0.6),
                                                      ((0, 255, 0), None, self.other, 0.5),
                                                      ((0, 0, 255), None, self.other, 0.2))):
            name = f"2026-11-02/120000-t{i}-1"
            Image.new("RGB", (112, 112), rgb).save(faces / f"{name}-face.png")
            Image.new("RGB", (40, 40), rgb).save(faces / f"{name}-crop.jpg")
            lines.append(json.dumps(dict(track=i, face=f"{name}-face.png", crop=f"{name}-crop.jpg", person=person,
                                         match=match, sim=sim)))
        (faces / "looks.jsonl").write_text("\n".join(lines) + "\n")
        self.faces = faces
        self.cmds = C.PeopleCommands(self.store, self.voices, ColourEmbedder(), labels=self.root / "labels.jsonl",
                                     roots=[self.data.resolve()])

    def face(self, i):
        return str(self.faces / f"2026-11-02/120000-t{i}-1-face.png")

    def labels(self):
        return [json.loads(x) for x in (self.root / "labels.jsonl").read_text().splitlines()]


class PeopleCommandsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.w = World(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_name_and_call(self):
        ok, msg = self.w.cmds.apply(dict(op="name", id=self.w.other, name="Emre"))
        self.assertTrue(ok, msg)
        self.assertEqual(self.w.store.name(self.w.other), "Emre")
        self.assertTrue(self.w.cmds.apply(dict(op="call", id=self.w.other, call="Emre Bey"))[0])
        self.assertEqual(self.w.store.call(self.w.other), "Emre Bey")
        self.assertEqual(self.w.labels()[0]["op"], "name")

    def test_merge_moves_vectors_and_voice_prints(self):
        ok, msg = self.w.cmds.apply(dict(op="merge", keep=self.w.selim, drop=self.w.other))
        self.assertTrue(ok, msg)
        self.assertNotIn(self.w.other, self.w.store.index)
        self.assertEqual(len(self.w.store.vectors[self.w.selim]), 3)
        self.assertEqual(self.w.voices.count(self.w.selim), 2)

    def test_forget_deletes_face_voice_and_the_recorded_pictures(self):
        ok, msg = self.w.cmds.apply(dict(op="forget", id=self.w.other))
        self.assertTrue(ok, msg)
        self.assertNotIn(self.w.other, self.w.store.index)
        self.assertEqual(self.w.voices.count(self.w.other), 0)
        self.assertFalse(Path(self.w.face(1)).exists())             # matched it confidently: gone
        self.assertFalse(Path(self.w.face(1).replace("-face.png", "-crop.jpg")).exists())
        self.assertTrue(Path(self.w.face(2)).exists())              # a weak match (0.2): not theirs, kept
        self.assertTrue(Path(self.w.face(0)).exists())              # Selim's: kept

    def test_relabel_a_face_to_someone_moves_it(self):
        ok, msg = self.w.cmds.apply(dict(op="relabel_look", face=self.w.face(1), to=self.w.selim))
        self.assertTrue(ok, msg)
        self.assertEqual(len(self.w.store.vectors[self.w.other]), 1)   # the green face removed from them...
        self.assertEqual(len(self.w.store.vectors[self.w.selim]), 2)    # ...and stored for Selim
        self.assertIn("removed 1", msg)
        self.assertEqual(self.w.labels()[-1]["person"], self.w.selim)

    def test_relabel_to_a_new_person(self):
        n = len(self.w.store.index)
        ok, msg = self.w.cmds.apply(dict(op="relabel_look", face=self.w.face(2), to="new"))
        self.assertTrue(ok, msg)
        self.assertEqual(len(self.w.store.index), n + 1)

    def test_delete_a_face_removes_the_files_and_the_stored_copy(self):
        ok, msg = self.w.cmds.apply(dict(op="delete_look", face=self.w.face(2)))
        self.assertTrue(ok, msg)
        self.assertFalse(Path(self.w.face(2)).exists())
        self.assertEqual(len(self.w.store.vectors[self.w.other]), 1)   # the blue vector was that face

    def test_bad_requests_are_refused_with_a_reason(self):
        for cmd in (dict(op="rm -rf"), dict(op="name", id="nobody00", name="x"),
                    dict(op="name", id=self.w.selim, name="x" * 41),
                    dict(op="delete_look", face="/etc/passwd"),
                    dict(op="delete_look", face=str(Path(self.dir.name) / "elsewhere-face.png")),
                    dict(op="merge", keep=self.w.selim, drop=self.w.selim),
                    dict(op="picture", id=self.w.selim, state="maybe")):
            ok, msg = self.w.cmds.apply(cmd)
            self.assertFalse(ok, cmd)
            self.assertTrue(msg)
        self.assertTrue(Path(self.w.face(0)).exists())

    def test_seen_together_cannot_be_merged(self):
        self.w.store.apart(self.w.selim, self.w.other)
        ok, msg = self.w.cmds.apply(dict(op="merge", keep=self.w.selim, drop=self.w.other))
        self.assertFalse(ok)
        self.assertIn("seen together", msg)


class CommandBoxTest(unittest.TestCase):
    def test_a_request_file_gets_its_answer(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            box = C.CommandBox(w.cmds, folder=Path(d) / "cmds")
            box.dir.mkdir()
            (box.dir / "1793-a.json").write_text(json.dumps(dict(op="name", id=w.other, name="Emre")))
            (box.dir / "1793-b.json").write_text("{broken")
            self.assertEqual(box.poll(), 2)
            a = json.loads((box.dir / "1793-a.done.json").read_text())
            b = json.loads((box.dir / "1793-b.done.json").read_text())
            self.assertTrue(a["ok"])
            self.assertFalse(b["ok"])
            self.assertEqual(sorted(p.name for p in box.dir.iterdir()), ["1793-a.done.json", "1793-b.done.json"])
            self.assertEqual(box.poll(), 0)                          # answered once


if __name__ == "__main__":
    unittest.main()


class DriveCommandsTest(unittest.TestCase):
    def setUp(self):
        from robot.commands import DriveCommands

        class People:
            def apply(self, cmd):
                return True, "people: " + cmd["op"]

        class Legs:
            def __init__(self):
                self.calls = []

            def stop(self, why):
                self.calls.append(("stop", why))
                return True

            def start(self, mode, why="", max_s=None):
                self.calls.append(("start", mode, max_s))
                return True, ""
        self.legs = Legs()
        self.cmds = DriveCommands(People(), self.legs)
        self.off = DriveCommands(People(), None, "battery off")

    def test_drive_ops_go_to_the_legs_others_to_people(self):
        from robot import config
        self.assertEqual(self.cmds.apply({"op": "name"}), (True, "people: name"))
        self.assertEqual(self.cmds.apply({"op": "drive_stop"}), (True, "stopped"))
        ok, msg = self.cmds.apply({"op": "drive_start", "mode": "explore"})
        self.assertTrue(ok)
        self.assertEqual(self.legs.calls[-1], ("start", "explore", config.WANDER_MAX_S))
        self.assertFalse(self.cmds.apply({"op": "drive_start", "mode": "fly"})[0])

    def test_no_drive(self):
        self.assertEqual(self.off.apply({"op": "drive_stop"}), (True, "no drive (battery off)"))
        self.assertFalse(self.off.apply({"op": "drive_start"})[0])
