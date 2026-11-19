"""Learning mode: robot/learning.py (jobs, progress, tuned.json) and robot/learn_faces.py (the face threshold job)."""
import gzip
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from robot import config
from robot import learn_faces as LF
from robot.learning import Change, Job, Progress, Result, Runner, State, Tuned, clamp_tunable

LIMITS = {"FACE_MATCH_SIM": (0.30, 0.50)}


class TunedTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "tuned.json")
        self.t = [1000.0]
        self.tuned = Tuned(self.path, limits=LIMITS, clock=lambda: self.t[0], defaults={"FACE_MATCH_SIM": 0.38})

    def tearDown(self):
        self.dir.cleanup()

    def test_only_tunable_keys_within_limits(self):
        self.assertEqual(self.tuned.offer(Change("MOTOR_MAX", 1.0, "no"), "x"), "refused")
        self.assertEqual(self.tuned.offer(Change("FACE_MATCH_SIM", 0.9, "too high", auto=True)), "applied")
        self.assertEqual(self.tuned.value("FACE_MATCH_SIM"), 0.50)
        self.assertIsNone(clamp_tunable("FACE_MATCH_SIM", float("nan"), LIMITS))

    def test_a_proposal_waits_until_decided(self):
        self.assertEqual(self.tuned.offer(Change("FACE_MATCH_SIM", 0.34, "data says so"), "faces"), "proposed")
        self.assertEqual(self.tuned.value("FACE_MATCH_SIM"), 0.38)          # not yet
        (p,) = self.tuned.data["proposals"]
        self.assertTrue(self.tuned.decide(p["id"], accept=True))
        self.assertEqual(self.tuned.value("FACE_MATCH_SIM"), 0.34)
        self.assertEqual(self.tuned.data["proposals"], [])
        self.assertFalse(self.tuned.decide(p["id"], accept=True))           # gone

    def test_rejected_proposal_changes_nothing_and_newest_proposal_wins(self):
        self.tuned.offer(Change("FACE_MATCH_SIM", 0.34, "a"))
        self.t[0] += 1
        self.tuned.offer(Change("FACE_MATCH_SIM", 0.36, "b"))
        (p,) = self.tuned.data["proposals"]
        self.assertEqual(p["new"], 0.36)
        self.tuned.decide(p["id"], accept=False)
        self.assertEqual(self.tuned.value("FACE_MATCH_SIM"), 0.38)

    def test_undo_and_kept_across_starts(self):
        self.tuned.offer(Change("FACE_MATCH_SIM", 0.36, "one", auto=True))
        self.tuned.offer(Change("FACE_MATCH_SIM", 0.34, "two", auto=True))
        again = Tuned(self.path, limits=LIMITS, defaults={"FACE_MATCH_SIM": 0.38})
        self.assertEqual(again.value("FACE_MATCH_SIM"), 0.34)
        self.assertTrue(again.undo("FACE_MATCH_SIM"))
        self.assertEqual(again.value("FACE_MATCH_SIM"), 0.36)
        self.assertTrue(again.undo("FACE_MATCH_SIM"))
        self.assertEqual(again.value("FACE_MATCH_SIM"), 0.38)               # the default
        self.assertNotIn("FACE_MATCH_SIM", again.data["values"])

    def test_config_reads_it_only_outside_the_tests(self):
        with open(self.path, "w") as f:
            json.dump({"values": {"FACE_MATCH_SIM": 0.9, "MOTOR_MAX": 1.0, "X": "a"}}, f)
        self.assertEqual(config._tuned_overrides(self.path, LIMITS), {})     # ARCBY_NO_PRIVATE is set in the tests
        old = os.environ.pop("ARCBY_NO_PRIVATE")
        try:
            self.assertEqual(config._tuned_overrides(self.path, LIMITS), {"FACE_MATCH_SIM": 0.5})
            self.assertEqual(config._tuned_overrides(self.path + ".missing", LIMITS), {})
        finally:
            os.environ["ARCBY_NO_PRIVATE"] = old


class ProgressTest(unittest.TestCase):
    def test_time_left_from_the_rate(self):
        now = [0.0]
        p = Progress(100, clock=lambda: now[0])
        self.assertIsNone(p.eta_s())
        for _ in range(10):
            now[0] += 1.0
            p.advance(2)
        self.assertAlmostEqual(p.eta_s(), 40.0, delta=1.0)                 # 80 left at 2 a second


class FakeJob(Job):
    def __init__(self, name, total=3, fail=False, change=None, stop_at=None):
        self.name = self.title = name
        self.total, self.fail, self.change, self.stop_at = total, fail, change, stop_at

    def prepare(self):
        return self.total

    def run(self, progress):
        for i in range(self.total):
            if self.stop_at is not None and i == self.stop_at:
                progress._stop.set()
            if progress.stopping:
                return Result(findings=["stopped"])
            if self.fail:
                raise RuntimeError("broken")
            progress.advance(1, step="working")
        return Result(findings=[f"{self.name} ok"], changes=[self.change] if self.change else [])


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        self.state = State(os.path.join(d, "state.json"), every_s=0.0)
        self.tuned = Tuned(os.path.join(d, "tuned.json"), limits=LIMITS, defaults={"FACE_MATCH_SIM": 0.38})

    def tearDown(self):
        self.dir.cleanup()

    def test_jobs_in_turn_a_broken_one_does_not_stop_the_others(self):
        r = Runner([FakeJob("a", fail=True), FakeJob("b", total=0),
                    FakeJob("c", change=Change("FACE_MATCH_SIM", 0.40, "c says", auto=True))],
                   state=self.state, tuned=self.tuned)
        with self.assertLogs("robot.learning", level="ERROR"):
            self.assertEqual(r.run(), [("a", "failed"), ("b", "nothing to do"), ("c", "done")])
        self.assertEqual(self.tuned.value("FACE_MATCH_SIM"), 0.40)
        st = json.loads(Path(self.state.path).read_text())
        self.assertFalse(st["running"])
        self.assertEqual(st["why"], "all done")
        self.assertEqual([x["job"] for x in st["results"]], ["a", "c"])
        self.assertEqual(st["results"][-1]["changes"][0]["how"], "applied")

    def test_stopped_mid_job_offers_nothing(self):
        r = Runner([FakeJob("a", total=5, stop_at=2, change=Change("FACE_MATCH_SIM", 0.40, "x", auto=True)),
                    FakeJob("b")], state=self.state, tuned=self.tuned)
        self.assertEqual(r.run(), [("a", "stopped"), ("b", "stopped")])
        self.assertEqual(self.tuned.value("FACE_MATCH_SIM"), 0.38)
        self.assertEqual(json.loads(Path(self.state.path).read_text())["why"], "stopped")

    def test_progress_reaches_the_state_file_while_it_runs(self):
        seen = []

        class Peek(FakeJob):
            def run(job, progress):
                progress.advance(1, step="half")
                seen.append(json.loads(Path(self.state.path).read_text()))
                return super().run(progress)
        Runner([Peek("p", total=2)], state=self.state, tuned=self.tuned).run()
        self.assertTrue(seen[0]["running"])
        self.assertEqual((seen[0]["job"], seen[0]["done"], seen[0]["total"], seen[0]["step"]), ("p", 1, 2, "half"))


# ---------- the face threshold job on made-up recordings ----------

def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


class World:
    """People A, B, C (known, in the store) and strangers; tracks of 10 looks; some tracks seen together."""

    def __init__(self, root, seed=0, spread=0.9):
        rng = np.random.default_rng(seed)
        self.root = Path(root)
        self.data = self.root / "data"
        self.people = self.root / "people"
        self.dims = 64
        centres = {p: unit(rng.normal(size=self.dims)) for p in ("A", "B", "C")}
        self.vec = {}

        def look(c):
            return unit(c + spread * unit(rng.normal(size=self.dims)) * rng.uniform(0.6, 1.0))
        # the store: 20 vectors per known person
        index = {}
        for p, c in centres.items():
            d = self.people / p
            d.mkdir(parents=True)
            np.save(d / "vectors.npy", np.array([look(c) for _ in range(20)]))
            index[p] = {"name": p, "created": 0}
        (self.people / "index.json").write_text(json.dumps({"people": index}))
        sess = self.data / "2026-11-02" / "120000"
        faces = sess / "faces"
        faces.mkdir(parents=True)
        lines, frames, tid = [], [], 0
        for k in range(30):                       # 30 scenes: a known person + a stranger in the frame together
            who = "ABC"[k % 3]
            stranger = unit(rng.normal(size=self.dims))
            pair = []
            for c in (centres[who], stranger):
                tid += 1
                pair.append(tid)
                for n in range(10):
                    name = f"t{tid}-{n}-face.png"
                    (faces / name).write_bytes(b"")
                    self.vec[str(faces / name)] = look(c)
                    lines.append(json.dumps(dict(track=tid, face=name, time=k * 100 + n * 3, quality=0.8)))
            frames.append(json.dumps({"tracks": [{"id": pair[0]}, {"id": pair[1]}]}))
        (faces / "looks.jsonl").write_text("\n".join(lines) + "\n")
        with gzip.open(sess / "frames.jsonl.gz", "wt") as f:
            f.write("\n".join(frames) + "\n")


class FakeEmbedder:
    def __init__(self, world):
        self.world, self.calls = world, 0

    def embed(self, path):
        self.calls += 1
        return self.world.vec[path]


class FaceJobTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.w = World(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def job(self, emb):
        return LF.FaceThresholdJob(embedder=emb, data_dir=self.w.data, collect_dir=self.w.root / "none",
                                   people_dir=self.w.people, cache_dir=self.w.root / "cache",
                                   load_image=lambda p: p)

    def test_finds_a_threshold_from_tracks_and_faces_seen_together(self):
        emb = FakeEmbedder(self.w)
        job = self.job(emb)
        self.assertEqual(job.prepare(), 601)                 # 600 looks + the analysis
        res = job.run(Progress(601))
        rep = job.report
        self.assertEqual(rep["known_tracks"], 30)
        self.assertEqual(rep["own"], 300)
        self.assertEqual(rep["covis"], 300)
        self.assertIsNotNone(rep["threshold"])
        row = next(r for r in rep["rows"] if r["t"] == rep["threshold"])
        now = next(r for r in rep["rows"] if abs(r["t"] - round(config.FACE_MATCH_SIM, 2)) < 1e-9)
        self.assertLessEqual(row["track_wrong"], LF.TRACK_WRONG_MAX)
        self.assertLessEqual(row["covis"], now["covis"] + LF.COVIS_SLACK)
        lower = [r for r in rep["rows"] if r["t"] < rep["threshold"]]
        self.assertTrue(all(r["track_wrong"] > LF.TRACK_WRONG_MAX or r["covis"] > now["covis"] + LF.COVIS_SLACK
                            for r in lower))                                # the lowest allowed
        self.assertTrue(res.findings)
        if res.changes:
            (ch,) = res.changes
            self.assertEqual(ch.key, "FACE_MATCH_SIM")
            self.assertEqual(ch.auto, abs(ch.new - config.FACE_MATCH_SIM) <= LF.AUTO_STEP)

    def test_a_second_run_embeds_only_new_looks(self):
        emb = FakeEmbedder(self.w)
        job = self.job(emb)
        job.run(Progress(job.prepare()))
        self.assertEqual(emb.calls, 600)
        again = self.job(emb)
        self.assertEqual(again.prepare(), 1)                 # nothing new: only the analysis
        again.run(Progress(1))
        self.assertEqual(emb.calls, 600)

    def test_stopping_keeps_what_was_embedded(self):
        emb = FakeEmbedder(self.w)
        job = self.job(emb)
        stop = threading.Event()
        p = Progress(job.prepare(), stop=stop)
        orig = p.advance

        def advance(n=1, step=None):
            orig(n, step)
            if p.done >= 250:
                stop.set()
        p.advance = advance
        res = job.run(p)
        self.assertIn("stopped", res.findings[0])
        self.assertEqual(self.job(emb).prepare(), 600 - 250 + 1)

    def test_a_recording_cut_by_a_power_loss_is_read_up_to_the_cut(self):   # Pi 2026-11-02, battery empty
        f = self.w.data / "2026-11-02" / "120000" / "frames.jsonl.gz"
        raw = f.read_bytes()
        for cut in (raw[:len(raw) // 2], raw[:20] + b"\x00" * 40 + raw[60:]):     # ended early / damaged inside
            f.write_bytes(cut)
            together = LF.seen_together(self.w.data, {"2026-11-02/120000"})     # no exception
            self.assertIsInstance(together, dict)

    def test_strangers_let_in_more_often_stop_it_going_lower(self):
        # two scenes: the known person's tracks match well; the stranger in the frame matches them at 0.33 -
        # going from 0.38 down to 0.32 would let every stranger look in: not allowed
        looks, rows = [], []
        for k in range(40):
            for n in range(5):
                looks.append(dict(session="s", track=2 * k, quality=0.8))
                rows.append([0.6])
                looks.append(dict(session="s", track=2 * k + 1, quality=0.8))
                rows.append([0.33])
        LF.MIN_OWN, old = 10, LF.MIN_OWN
        try:
            rep = LF.analyse(looks, np.array(rows), {("s", 2 * k): {2 * k + 1} for k in range(40)}, current=0.38)
        finally:
            LF.MIN_OWN = old
        self.assertGreaterEqual(rep["threshold"], 0.34)

    def test_too_little_data_changes_nothing(self):
        looks = [dict(session="s", track=1, quality=0.8)] * 5
        rep = LF.analyse(looks, np.full((5, 1), 0.6), {})
        self.assertIsNone(rep["threshold"])
        self.assertIn("not enough", rep["why"])


class FakeProc:
    def __init__(self):
        self.returncode, self.terminated, self.killed, self.pid = None, 0, 0, 42
        self.exit_on_term = True

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        if self.exit_on_term:
            self.returncode = 0

    def kill(self):
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            import subprocess
            raise subprocess.TimeoutExpired("learn.py", timeout)
        return self.returncode


class SupervisorTest(unittest.TestCase):
    def setUp(self):
        from robot.learning import LearnSupervisor
        self.dir = tempfile.TemporaryDirectory()
        self.battery, self.temp, self.procs = [True], [50.0], []

        def popen(cmd, **kw):
            self.procs.append(FakeProc())
            return self.procs[-1]
        self.s = LearnSupervisor(probe=lambda: self.battery[0], temp=lambda: self.temp[0], popen=popen,
                                 cmd=["learn.py"], log_path=os.path.join(self.dir.name, "learn.out"),
                                 every_s=60, off_checks=2, max_temp=75)
        self.t = 0.0

    def tearDown(self):
        self.dir.cleanup()

    def tick(self, n=1):
        for _ in range(n):
            self.s.tick(self.t)
            self.t += 60

    def test_battery_on_never_learns(self):
        self.tick(5)
        self.assertEqual(self.procs, [])
        self.assertIn("battery on", self.s.why)

    def test_wall_power_twice_in_a_row_starts_it_battery_back_stops_it(self):
        self.battery[0] = False
        self.tick()
        self.assertEqual(self.procs, [])                    # once could be a glitch
        self.tick()
        self.assertTrue(self.s.running)
        self.battery[0] = True
        self.tick()
        self.assertEqual(self.procs[0].terminated, 1)
        self.tick()
        self.assertFalse(self.s.running)
        self.battery[0] = False                             # wall power again: it goes on where it stopped
        self.tick(2)
        self.assertEqual(len(self.procs), 2)

    def test_too_hot_waits_or_stops(self):
        self.battery[0], self.temp[0] = False, 80.0
        self.tick(3)
        self.assertEqual(self.procs, [])
        self.temp[0] = 60.0
        self.tick()
        self.assertTrue(self.s.running)
        self.temp[0] = 76.0
        self.tick()
        self.assertEqual(self.procs[0].terminated, 1)

    def test_all_done_is_not_started_again_until_the_next_launch(self):
        self.battery[0] = False
        self.tick(2)
        self.procs[0].returncode = 0                        # finished by itself
        self.tick(5)
        self.assertEqual(len(self.procs), 1)
        self.assertTrue(self.s.finished)

    def test_a_learner_that_does_not_stop_is_killed_without_blocking(self):
        self.battery[0] = False
        self.tick(2)
        p = self.procs[0]
        p.exit_on_term = False
        self.battery[0] = True
        self.s.tick(self.t)                                 # SIGTERM, returns at once
        self.assertEqual((p.terminated, p.killed), (1, 0))
        self.t += 60                                        # past the 15 s grace
        self.s.tick(self.t)
        self.assertEqual(p.killed, 1)

    def test_a_probe_error_counts_as_battery_on(self):
        def broken():
            raise OSError("bus")
        self.s.probe = broken
        self.tick(3)
        self.assertEqual(self.procs, [])


if __name__ == "__main__":
    unittest.main()
