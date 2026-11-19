"""robot/recorder.py + robot/sensors.py + code/data_cli.py: the dataset the robot records (fake sensors)."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot import config
from robot.recorder import Recorder
from robot.sensors import SensorLogger

import data_cli


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.now = [100.0]

    def tearDown(self):
        self.dir.cleanup()

    def rec(self, max_bytes=10**9, wall=1793462160.0):
        return Recorder(self.dir.name, max_bytes=max_bytes, meta=dict(program="test"), clock=lambda: self.now[0],
                        wall=lambda: wall)

    def test_session_layout_and_meta(self):
        r = self.rec()
        self.assertEqual(r.dir.parent.parent, Path(self.dir.name))
        meta = json.loads((r.dir / "meta.json").read_text())
        self.assertEqual(meta["program"], "test")
        self.assertIn("FACE_EMBEDDER", meta["config"])
        self.assertTrue((r.dir / "keyframes").is_dir())
        self.assertNotEqual(self.rec().dir, r.dir)                    # a second run in the same second

    def test_streams_are_readable_while_still_being_written(self):
        r = self.rec()
        for t in (0.0, 0.1, 0.2, 0.3, 0.4):
            r.write("frames", {"t": t, "faces": [], "head": np.array([1.5, 2.0])})
        r.write("events", {"kind": "said", "text": "Günaydın Selim!"})
        r.flush()                                                     # a power cut after this
        got = list(data_cli.records(r.dir / "frames.jsonl.gz"))
        self.assertEqual([x["t"] for x in got], [0, 0.1, 0.2, 0.3, 0.4])
        self.assertEqual(got[0]["head"], [1.5, 2.0])
        ev = list(data_cli.records(r.dir / "events.jsonl.gz"))[0]
        self.assertEqual((ev["text"], ev["t"], ev["wall"]), ("Günaydın Selim!", 100.0, 1793462160.0))
        r.close()
        self.assertEqual(len(list(data_cli.records(r.dir / "frames.jsonl.gz"))), 5)

    def test_keyframes_only_when_the_scene_changes(self):
        r = self.rec()
        rng = np.random.default_rng(0)
        room = rng.integers(0, 255, (54, 96, 3), np.uint8)
        kept = lambda frames: sum(x is not None for x in frames)          # noqa: E731
        self.assertEqual(kept([r.keyframe(room, t / 10, busy=True, who={"selim"}) for t in range(300)]), 1)
        busy = [r.keyframe(rng.integers(0, 255, (54, 96, 3), np.uint8), 30 + t / 10, busy=True, who={"selim"})
                for t in range(30)]                                       # 3 s of a scene that keeps changing
        self.assertEqual(kept(busy), 3)                                   # ...at most 1 per second
        self.assertIsNotNone(r.keyframe(room, 34.0, busy=True, who={"selim", "meltem"}))   # someone came
        quiet = [r.keyframe(room, 35 + t, busy=False, who={"selim", "meltem"}) for t in range(700)]
        self.assertEqual(kept(quiet), 2)                                  # unchanged: one per DATA_KEYFRAME_MAX_S
        self.assertGreater(r.keyframes_skipped, 50)

    def test_old_keyframes_go_first_faces_last_never_this_session(self):
        from robot.recorder import folder_bytes
        old = []
        for day in range(3):
            r = self.rec(wall=1793462160.0 + day * 86400)
            (r.dir / "keyframes" / "k.jpg").write_bytes(os.urandom(5000))
            r.write("frames", {"x": os.urandom(800).hex()})
            (r.dir / "faces").mkdir()
            (r.dir / "faces" / "f.png").write_bytes(os.urandom(300))
            r.close()
            old.append(r.dir)
        now = self.rec(wall=1793462160.0 + 3 * 86400)
        (now.dir / "keyframes" / "k.jpg").write_bytes(os.urandom(5000))
        everything = folder_bytes(self.dir.name)
        now.max_bytes = everything - 2 * 5000 + 100                   # room after dropping 2 old keyframe sets...
        now.retain()
        self.assertFalse((old[0] / "keyframes").exists() or (old[1] / "keyframes").exists())
        self.assertTrue((old[2] / "keyframes").exists())
        self.assertTrue(all((s / "faces" / "f.png").exists() and (s / "frames.jsonl.gz").exists() for s in old))
        now.max_bytes = 3 * 5000 // 2                                 # ...much less room: streams, then sessions
        now.retain()
        self.assertFalse(old[0].exists() or old[0].parent.exists())   # empty day folder removed too
        self.assertTrue(now.dir.exists() and (now.dir / "keyframes" / "k.jpg").exists())
        tiny = self.rec(max_bytes=10, wall=1793462160.0 + 4 * 86400)  # over the limit alone: kept
        self.assertTrue(tiny.dir.exists())

    def test_keeps_the_disk_free_and_trims_this_session_last(self):
        r = self.rec(max_bytes=10**9)
        self.assertEqual(r.limit(total=1000, free_bytes=10**12), 10**9)            # plenty free: max_bytes
        self.assertEqual(r.limit(total=1000, free_bytes=config.DATA_MIN_FREE_BYTES + 500), 1500)
        for i in range(5):
            (r.dir / "keyframes" / f"{i}.000.jpg").write_bytes(os.urandom(4000))
        r.max_bytes = 3 * 4000 + 2000                         # this session alone is over: oldest keyframes go
        r.retain(free_bytes=10**12)
        left = sorted(p.name for p in (r.dir / "keyframes").iterdir())
        self.assertEqual(left[-1], "4.000.jpg")                # the newest stays
        self.assertNotIn("0.000.jpg", left)

    def test_a_failing_disk_switches_recording_off_not_the_robot(self):
        r = self.rec()
        r.write("events", {"kind": "a"})
        with unittest.mock.patch("robot.recorder.gzip.open", side_effect=OSError(28, "No space left on device")):
            r.write("frames", {"t": 1.0})                     # no exception
        self.assertIn("No space", r.failed)
        r.write("events", {"kind": "b"})                      # ignored from now on
        self.assertIsNone(r.keyframe(np.zeros((54, 96, 3), np.uint8), 1.0, busy=True))
        r.close()
        self.assertEqual(r.counts, {"events": 1})

    def test_size_check_runs_in_its_own_thread(self):
        r = self.rec()
        started = threading.Event()
        release = threading.Event()

        def slow(free_bytes=None):
            started.set()
            release.wait(2)
        r.retain = slow
        self.now[0] += config.DATA_RETAIN_EVERY_S + 1
        t0 = time.monotonic()
        r.tick()                                              # returns at once
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertTrue(started.wait(1))
        r.tick()                                              # still running: not a second one
        release.set()
        r.close()

    def test_data_cli_lists_and_shows(self):
        r = self.rec()
        r.write("frames", {"t": 1.0})
        r.write("frames", {"t": 2.0})
        r.close()
        with unittest.mock.patch.object(data_cli, "ROOT", Path(self.dir.name)):
            self.assertEqual(data_cli.sessions(), [r.dir])
            streams, length = data_cli.summary(r.dir)
            self.assertEqual((len(streams["frames"]), length), (2, 1.0))


class FakeImu:
    def __init__(self, turning=True):
        self.n, self.turning = 0, turning

    def step(self):
        self.n += 1
        gyro = (0.1, 0.0, 30.0 if self.turning else 0.0)
        return SimpleNamespace(t=time.monotonic(), accel_g=(0.0, 0.0, 1.0), gyro_dps=gyro, roll_deg=1.0,
                               pitch_deg=-2.0, yaw_deg=3.0, tilt_deg=2.2, tipped=False, bump_g=0.01, temp_c=31.0)


class FakeSonar:
    def __init__(self):
        self.answers = iter([123.4, "nothing"] * 1000)

    def ping(self):
        return next(self.answers)


class BrokenSonar:
    def ping(self):
        raise OSError("GPIO busy")


class SensorLoggerTest(unittest.TestCase):
    def test_streams_from_every_sensor_and_a_broken_one_is_only_logged(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(d, meta=dict(program="test"))
            stop = threading.Event()
            logger = SensorLogger(rec, imu=FakeImu(), sonar=BrokenSonar(), power=lambda: (False, True, "fake"))
            with unittest.mock.patch.object(config, "DATA_SONAR_HZ", 50):
                logger.start(stop)
                time.sleep(0.5)
                stop.set()
                logger.join()
            rec.close()
            imu = list(data_cli.records(rec.dir / "imu.jsonl.gz"))
            self.assertGreater(len(imu), 6)                            # turning: ~25 Hz of the 100 Hz reads
            self.assertEqual(imu[0]["accel_g"], [0.0, 0.0, 1.0])
            system = list(data_cli.records(rec.dir / "system.jsonl.gz"))
            self.assertTrue(any(r.get("sensor") == "sonar" and "GPIO busy" in r["off"] for r in system))
            self.assertTrue(any(r.get("undervolt_since_boot") is True for r in system))
            self.assertTrue(logger.status["sonar"].startswith("off"))

    def test_a_robot_standing_still_makes_little_imu_data(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(d, meta=dict(program="test"))
            stop = threading.Event()
            logger = SensorLogger(rec, imu=FakeImu(turning=False), sonar=FakeSonar(), power=lambda: (False, False, "f"))
            logger.start(stop)
            time.sleep(1.5)
            stop.set()
            logger.join()
            rec.close()
            self.assertLessEqual(len(list(data_cli.records(rec.dir / "imu.jsonl.gz"))), 3)   # ~1 Hz, not 25

    def test_sonar_answers_and_misses(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(d, meta=dict(program="test"))
            stop = threading.Event()
            logger = SensorLogger(rec, imu=FakeImu(), sonar=FakeSonar(), power=lambda: (False, False, "fake"))
            with unittest.mock.patch.object(config, "DATA_SONAR_HZ", 50):
                logger.start(stop)
                time.sleep(0.3)
                stop.set()
                logger.join()
            rec.close()
            sonar = list(data_cli.records(rec.dir / "sonar.jsonl.gz"))
            self.assertIn(123.4, [r["cm"] for r in sonar])
            self.assertIn("nothing", [r.get("why") for r in sonar])


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
