"""pi/dashboard/arcby_archive.py: the People / Heard tabs - reading the robot's files (stdlib only), serving pictures
from its folders only, and changes as requests to the robot (robot/commands.py)."""
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from robot.people import PeopleStore
from robot.voices import VoiceStore

ARCH = Path(__file__).resolve().parents[2] / "pi" / "dashboard" / "arcby_archive.py"
spec = importlib.util.spec_from_file_location("arcby_archive_test", ARCH)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        r = Path(self.dir.name)
        self.old = {k: getattr(A, k) for k in ("PEOPLE_DIR", "DATA_DIR", "COLLECT_DIR", "HEARD_DIR", "COMMAND_DIR",
                                                "TUNED_FILE", "LABELS_FILE")}
        A.PEOPLE_DIR, A.DATA_DIR, A.COLLECT_DIR = r / "people", r / "data", r / "collect"
        A.HEARD_DIR, A.COMMAND_DIR, A.TUNED_FILE = r / "heard", r / "cmds", r / "tuned.json"
        A.LABELS_FILE = r / "labels.jsonl"
        A._labels_cache["sig"] = None
        A._looks_cache["t"] = 0.0
        store = PeopleStore(str(A.PEOPLE_DIR), clock=lambda: 1000.0)
        self.selim = store.enrol([unit([1, 0, 0]), unit([1, 0.1, 0])], photo=np.zeros((8, 8, 3), np.uint8))
        store.set_name(self.selim, "Selim")
        self.other = store.enrol([unit([0, 1, 0])])
        VoiceStore(str(A.PEOPLE_DIR)).learn(self.selim, unit([1, 0, 0, 0]))
        faces = A.DATA_DIR / "2026-11-02" / "120000" / "faces"
        (faces / "2026-11-02").mkdir(parents=True)
        lines = []
        for i, (person, match, sim) in enumerate(((self.selim, self.selim, 0.6), (None, self.other, 0.5),
                                                   (None, self.other, 0.2))):
            name = f"2026-11-02/1200{i:02d}-t{i}-1"
            Image.new("RGB", (16, 16)).save(faces / f"{name}-face.png")
            lines.append(json.dumps(dict(track=i, face=f"{name}-face.png", person=person, match=match, sim=sim,
                                         time=1000.0 + i)))
        (faces / "looks.jsonl").write_text("\n".join(lines) + "\n")
        A.HEARD_DIR.mkdir()
        (A.HEARD_DIR / "20261102-153719-heard-merhaba-selim.wav").write_bytes(b"RIFF....")
        (A.HEARD_DIR / "201923-dropped.wav").write_bytes(b"RIFF....")

    def tearDown(self):
        for k, v in self.old.items():
            setattr(A, k, v)
        self.dir.cleanup()

    def test_people_with_counts_photo_and_their_pictures(self):
        d = A.people()
        by = {p["id"]: p for p in d["people"]}
        self.assertEqual(by[self.selim]["name"], "Selim")
        self.assertEqual((by[self.selim]["vectors"], by[self.selim]["voices"], by[self.selim]["looks"]), (2, 1, 1))
        self.assertTrue(by[self.selim]["photo"].startswith("/media/data/"))      # the newest recorded face
        self.assertTrue(by[self.other]["photo"].startswith("/media/data/"))
        self.assertIsNotNone(A.media_file(f"/media/people/{self.selim}/face.jpg"))   # the old photo still served
        self.assertEqual(by[self.other]["looks"], 1)              # the 0.5 match counts, the 0.2 one does not
        self.assertEqual(d["unknown_looks"], 1)
        self.assertEqual(A.looks_of("unknown")["total"], 1)

    def test_this_is_moves_the_picture_and_merges_follow(self):    # Selim 2026-11-02: it stayed under the old one
        face = A.looks_of(self.other)["looks"][0]["face"]
        A.LABELS_FILE.write_text(json.dumps({"op": "relabel", "face": face, "person": self.selim}) + "\n")
        A._looks_cache["t"] = 0.0
        self.assertEqual(A.looks_of(self.other)["total"], 0)
        self.assertEqual(A.looks_of(self.selim)["total"], 2)
        with open(A.LABELS_FILE, "a") as f:                    # then Selim was merged into the other one
            f.write(json.dumps({"op": "merge", "keep": self.other, "drop": self.selim}) + "\n")
        self.assertEqual(A.looks_of(self.other)["total"], 2)
        self.assertEqual(A.looks_of(self.selim)["total"], 0)

    def test_a_card_shows_the_newest_good_picture(self):
        photo = {p["id"]: p["photo"] for p in A.people()["people"]}
        self.assertTrue(photo[self.selim].endswith("-face.png"))   # a recorded face, not the old enrolment photo

    def test_the_learnt_threshold_decides_whose_a_picture_is(self):
        A.TUNED_FILE.write_text(json.dumps({"values": {"FACE_MATCH_SIM": 0.55}}))
        A._looks_cache["t"] = 0.0
        self.assertEqual(A.looks_of(self.other)["total"], 0)      # 0.5 < 0.55 now: not theirs

    def test_media_only_from_the_robots_folders(self):
        url = A.people()["people"][0]["photo"] or A.looks_of(self.selim)["looks"][0]["face_url"]
        self.assertIsNotNone(A.media_file(url))
        for bad in ("/media/people/../../etc/passwd", "/media/etc/passwd", "/media/heard/x.sh",
                    "/media/data/../people/index.json"):
            self.assertIsNone(A.media_file(bad), bad)

    def test_heard_newest_first_and_delete_only_clips(self):
        clips = A.heard()["clips"]
        self.assertEqual({c["kind"] for c in clips}, {"heard", "dropped"})
        heard = next(c for c in clips if c["kind"] == "heard")
        self.assertEqual(heard["text"], "merhaba selim")
        self.assertTrue(A.heard_delete(json.dumps({"name": "201923-dropped.wav"}).encode())["ok"])
        for bad in ("../people/index.json", "x.sh", "a/b.wav"):
            with self.assertRaises((ValueError, OSError)):
                A.heard_delete(json.dumps({"name": bad}).encode())

    def test_requests_are_checked_before_they_leave(self):
        face = A.looks_of(self.selim)["looks"][0]["face"]
        self.assertEqual(A.check_command({"op": "delete_look", "face": face, "x": 1}), {"op": "delete_look", "face": face})
        for bad in ({"op": "drop_table"}, {"op": "merge", "keep": "a;rm", "drop": self.selim},
                    {"op": "delete_look", "face": "/etc/passwd-face.png"}, {"op": "name", "id": self.selim, "name": "x" * 41},
                    {"op": "relabel_look", "face": face, "to": "everyone"}, []):
            with self.assertRaises(ValueError):
                A.check_command(bad)

    def test_robot_not_running_people_cli_applies_it(self):
        calls = []

        def run(cmd, **kw):                             # people_cli.py apply FILE, done by hand here
            calls.append(cmd)
            req = Path(cmd[-1])
            req.with_name(req.name[:-5] + ".done.json").write_text(json.dumps({"ok": True, "message": "named"}))
            req.unlink()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        res = A.command(json.dumps({"op": "name", "id": self.selim, "name": "Selim"}).encode(), lambda: False, run=run)
        self.assertEqual(res, {"ok": True, "message": "named"})
        self.assertEqual(calls[0][2], "apply")
        self.assertEqual(list(A.COMMAND_DIR.iterdir()), [])

    def test_robot_running_answers_through_the_command_box(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from robot.commands import CommandBox, PeopleCommands
        store = PeopleStore(str(A.PEOPLE_DIR))
        box = CommandBox(PeopleCommands(store, labels=Path(self.dir.name) / "labels.jsonl"), folder=A.COMMAND_DIR,
                         every_s=0.05)
        box.dir.mkdir(parents=True, exist_ok=True)
        box.start()
        try:
            res = A.command(json.dumps({"op": "name", "id": self.other, "name": "Emre"}).encode(), lambda: True,
                            wait_s=5)
        finally:
            box.close()
        self.assertTrue(res["ok"], res)
        self.assertEqual(store.name(self.other), "Emre")

    def test_the_drive_card_stop_and_start_go_to_alive(self):
        self.assertEqual(A.check_command({"op": "drive_start", "mode": "explore", "x": 1}),
                         {"op": "drive_start", "mode": "explore"})
        with self.assertRaises(ValueError):
            A.check_command({"op": "drive_start", "mode": "follow_everyone"})
        res = A.command(json.dumps({"op": "drive_stop"}).encode(), lambda: False)     # alive.py not running
        self.assertTrue(res["ok"])
        self.assertIn("not running", res["message"])
        self.assertFalse(A.command(json.dumps({"op": "drive_start"}).encode(), lambda: False)["ok"])
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from robot.commands import CommandBox, DriveCommands

        class Legs:
            stopped = []

            def stop(self, why):
                self.stopped.append(why)
                return True
        legs = Legs()
        box = CommandBox(DriveCommands(None, legs), folder=A.COMMAND_DIR, every_s=0.05)
        box.dir.mkdir(parents=True, exist_ok=True)
        box.start()
        try:
            res = A.command(json.dumps({"op": "drive_stop"}).encode(), lambda: True)
        finally:
            box.close()
        self.assertEqual(res, {"ok": True, "message": "stopped"})
        self.assertEqual(legs.stopped, ["dashboard STOP"])

    def test_robot_too_slow_says_so(self):
        res = A.command(json.dumps({"op": "name", "id": self.selim, "name": "x"}).encode(), lambda: True, wait_s=0.1,
                        sleep=lambda s: time.sleep(0.02))
        self.assertFalse(res["ok"])
        self.assertIn("did not answer", res["message"])

    def test_npy_header_rows(self):
        import io
        for shape in ((0, 512), (7, 512), (40, 192)):
            buf = io.BytesIO()
            np.save(buf, np.zeros(shape, np.float32))
            self.assertEqual(A._npy_rows(buf.getvalue()[:256]), shape[0])


class DashboardRoutesTest(unittest.TestCase):
    def test_the_page_has_the_tabs_and_the_routes_exist(self):
        dash_path = ARCH.with_name("arcby_dashboard.py")
        spec = importlib.util.spec_from_file_location("arcby_dashboard_tabs", dash_path)
        d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(d)
        for tab in ("live", "people", "heard", "learning", "system"):
            self.assertIn(f'data-tab="{tab}"', d.PAGE)
        self.assertIn("api/people/looks", d.PAGE)
        self.assertTrue(callable(d.archive.people) and callable(d.archive.command))   # imported next to it


if __name__ == "__main__":
    unittest.main()
