"""
Learning mode (docs/vision.md, part L): while the Pi runs on the wall adapter (no battery = no head,
no motors), code/learn.py works through jobs that improve the robot from its own recordings - one after another,
each resumable, at low priority.

    runner = Runner([FaceThresholdJob(), ...], state=State(), tuned=Tuned())
    runner.run()                         # until done or SIGTERM (alive.py stops it when the battery comes back)

- A Job says how much work it has (`prepare` -> units, e.g. face looks to embed), does it (`run`, calling
  `progress.advance(n)` and checking `progress.stopping`), and returns a Result: findings (lines for the dashboard)
  and Changes (a learnt value: key, new value, reason, auto or proposed).
- State: ~/scratch/learn/state.json (at most 2x a second) - the dashboard's "Learning" card: the job, its progress,
  the time left (work left / the measured rate), the queue, the last results, the proposals waiting.
- Tuned: ~/.local/share/arcby/tuned.json - the only place learnt values go. Only keys in config.TUNABLE, each within
  its limits; every change kept with the old value and the reason (`learn.py undo KEY`). config.py reads it at start.
  A change applies by itself when the job says so (small, clear evidence); else it waits as a proposal for Selim
  (dashboard / `learn.py apply ID`).
"""
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


@dataclass
class Change:
    key: str
    new: float
    reason: str
    auto: bool = False          # apply without asking (the job's evidence is clear and the step small)


@dataclass
class Result:
    findings: list = field(default_factory=list)    # short lines for the dashboard / log
    changes: list = field(default_factory=list)     # [Change]


class Job:
    """One piece of learning. Subclasses: name, title, prepare(), run()."""
    name = "job"
    title = "a job"

    def prepare(self):
        """How many work units (looks, clips...) there are to do now - 0 = nothing to do."""
        return 1

    def run(self, progress):
        """Do the work: progress.advance(n) as units are done; return early (checkpointed) when progress.stopping.
        Returns a Result."""
        progress.advance(1)
        return Result()


def _atomic_write(path, data):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------- learnt values ----------

def clamp_tunable(key, value, limits=None):
    """value inside config.TUNABLE[key]'s limits, or None (not tunable / not a number)."""
    limits = config.TUNABLE if limits is None else limits
    if key not in limits:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    lo, hi = limits[key]
    return min(max(v, lo), hi)


class Tuned:
    """tuned.json: {"values": {key: value}, "history": [...], "proposals": [...]}."""

    def __init__(self, path=None, limits=None, clock=time.time, defaults=None):
        self.path = Path(path or config.TUNED_FILE).expanduser()
        self.limits = config.TUNABLE if limits is None else limits
        self.defaults = defaults if defaults is not None else {k: getattr(config, k, None) for k in self.limits}
        self._clock = clock
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self):
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        if not isinstance(d, dict):
            d = {}
        values = {k: v for k, v in (d.get("values") or {}).items() if clamp_tunable(k, v, self.limits) == v}
        return dict(values=values, history=list(d.get("history") or [])[-200:],
                    proposals=[p for p in (d.get("proposals") or []) if isinstance(p, dict)][-50:])

    def _save(self):
        _atomic_write(self.path, self.data)

    def value(self, key):
        return self.data["values"].get(key, self.defaults.get(key))

    def offer(self, change, job=""):
        """A job's Change: applied now (auto) or kept as a proposal. Returns "applied" / "proposed" / "same" /
        "refused" (not tunable)."""
        v = clamp_tunable(change.key, change.new, self.limits)
        if v is None:
            return "refused"
        if v != change.new:
            change = Change(change.key, v, change.reason + f" (limited to {v})", change.auto)
        old = self.value(change.key)
        if old is not None and abs(float(old) - v) < 1e-9:
            return "same"
        with self._lock:
            if change.auto:
                self._apply(change.key, v, change.reason, job, how="auto")
                return "applied"
            pid = f"p{int(self._clock() * 1000)}"
            self.data["proposals"] = [p for p in self.data["proposals"] if p["key"] != change.key]   # newest wins
            self.data["proposals"].append(dict(id=pid, key=change.key, old=old, new=v, reason=change.reason, job=job,
                                               t=round(self._clock(), 1)))
            self._save()
        return "proposed"

    def _apply(self, key, value, reason, job, how):
        self.data["history"].append(dict(key=key, old=self.value(key), new=value, reason=reason, job=job, how=how,
                                         t=round(self._clock(), 1)))
        self.data["history"] = self.data["history"][-200:]
        self.data["values"][key] = value
        self._save()

    def decide(self, pid, accept):
        """Selim's answer to a proposal (dashboard / learn.py apply|reject ID). True if it was there."""
        with self._lock:
            for p in self.data["proposals"]:
                if p["id"] == pid:
                    self.data["proposals"].remove(p)
                    if accept and clamp_tunable(p["key"], p["new"], self.limits) is not None:
                        self._apply(p["key"], clamp_tunable(p["key"], p["new"], self.limits), p["reason"],
                                    p.get("job", ""), how="approved")
                    else:
                        self._save()
                    return True
        return False

    def undo(self, key):
        """Back to the value before the last change of key (the default when that was the first). True if changed."""
        with self._lock:
            skip = 0                                # changes already undone (undo after undo goes further back)
            for i in range(len(self.data["history"]) - 1, -1, -1):
                h = self.data["history"][i]
                if h["key"] != key:
                    continue
                if h.get("how") == "undo":
                    skip += 1
                    continue
                if skip:
                    skip -= 1
                    continue
                old = h.get("old")
                if old is None or old == self.defaults.get(key):
                    self.data["values"].pop(key, None)
                else:
                    self.data["values"][key] = old
                self.data["history"].append(dict(key=key, old=h["new"], new=old, reason="undo", how="undo",
                                                 t=round(self._clock(), 1)))
                self._save()
                return True
        return False


def load_overrides(path=None, limits=None):
    """{key: value} from tuned.json as config.py reads it at start (tunable keys only, within limits)."""
    return config._tuned_overrides(path or config.TUNED_FILE, limits)


# ---------- progress + state ----------

class Progress:
    """What the running job reports; the time left from the measured rate (units per second, smoothed)."""

    def __init__(self, total, on_change=None, clock=time.monotonic, stop=None):
        self.total, self.done = max(0, int(total)), 0
        self._clock, self._on_change = clock, on_change
        self.started = clock()
        self.rate = None
        self._last = (self.started, 0)
        self._stop = stop or threading.Event()
        self.step = ""

    @property
    def stopping(self):
        return self._stop.is_set()

    def advance(self, n=1, step=None):
        self.done = min(self.total, self.done + n) if self.total else self.done + n
        if step is not None:
            self.step = step
        now = self._clock()
        t0, d0 = self._last
        if now - t0 >= 1.0 and self.done > d0:
            r = (self.done - d0) / (now - t0)
            self.rate = r if self.rate is None else 0.7 * self.rate + 0.3 * r
            self._last = (now, self.done)
        elif self.rate is None and now > self.started and self.done:
            self.rate = self.done / (now - self.started)
        if self._on_change:
            self._on_change()

    def eta_s(self):
        if not self.total or self.rate is None or self.rate <= 0:
            return None
        return max(0.0, (self.total - self.done) / self.rate)


class State:
    """~/scratch/learn/state.json for the dashboard (written at most every `every_s`, and at each job's end)."""

    def __init__(self, path=None, every_s=0.5, clock=time.monotonic, wall=time.time):
        self.path = Path(path or config.LEARN_STATE_FILE).expanduser()
        self.every_s, self._clock, self._wall = every_s, clock, wall
        self._last = -1e9
        self.data = dict(running=False, job=None, title=None, step="", done=0, total=0, eta_s=None, queue=[],
                         results=[], proposals=[], started=None, updated=None, why=None)

    def update(self, force=False, **kw):
        self.data.update(kw)
        now = self._clock()
        if force or now - self._last >= self.every_s:
            self._last = now
            self.data["updated"] = round(self._wall(), 1)
            try:
                _atomic_write(self.path, self.data)
            except OSError as e:
                log.warning("learn: could not write %s (%s)", self.path, e)


class Runner:
    def __init__(self, jobs, state=None, tuned=None, stop=None, clock=time.monotonic, wall=time.time):
        self.jobs, self.state, self.tuned = list(jobs), state or State(), tuned or Tuned()
        self.stop = stop or threading.Event()
        self._clock, self._wall = clock, wall

    def install_signals(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop.set())

    def run(self):
        """Every job in turn. Returns [(job name, "done" / "nothing to do" / "stopped" / "failed")]."""
        out = []
        results = list(self.state.data.get("results") or [])[-10:]
        self.state.update(force=True, running=True, started=round(self._wall(), 1), why=None,
                          queue=[j.name for j in self.jobs], proposals=self.tuned.data["proposals"])
        for i, job in enumerate(self.jobs):
            if self.stop.is_set():
                out.append((job.name, "stopped"))
                continue
            queue = [j.name for j in self.jobs[i + 1:]]
            try:
                total = job.prepare()
            except Exception:  # noqa: BLE001 - one broken job must not stop the others
                log.exception("learn: %s could not start", job.name)
                out.append((job.name, "failed"))
                continue
            if not total:
                out.append((job.name, "nothing to do"))
                continue
            progress = Progress(total, clock=self._clock, stop=self.stop)

            def report(p=progress, job=job, queue=queue):
                eta = p.eta_s()
                self.state.update(job=job.name, title=job.title, step=p.step, done=p.done, total=p.total,
                                  eta_s=None if eta is None else round(eta), queue=queue)
            progress._on_change = report
            report()
            self.state.update(force=True)
            log.info("learn: %s - %d to do", job.name, total)
            try:
                res = job.run(progress) or Result()
            except Exception as e:  # noqa: BLE001
                log.exception("learn: %s failed", job.name)
                results.append(dict(job=job.name, title=job.title, how="failed", findings=[str(e)[:200]],
                                    changes=[], t=round(self._wall(), 1)))
                out.append((job.name, "failed"))
                continue
            if self.stop.is_set() and progress.done < progress.total:
                out.append((job.name, "stopped"))
                continue
            changes = []
            for ch in res.changes:
                how = self.tuned.offer(ch, job=job.name)
                changes.append(dict(key=ch.key, new=ch.new, reason=ch.reason, how=how))
                log.info("learn: %s %s -> %s (%s): %s", job.name, ch.key, ch.new, how, ch.reason)
            results.append(dict(job=job.name, title=job.title, how="done", findings=list(res.findings)[:20],
                                changes=changes, t=round(self._wall(), 1)))
            self.state.update(force=True, results=results[-10:], proposals=self.tuned.data["proposals"])
            out.append((job.name, "done"))
        self.state.update(force=True, running=False, job=None, title=None, step="", eta_s=None, queue=[],
                          results=results[-10:], proposals=self.tuned.data["proposals"],
                          why="stopped" if self.stop.is_set() else "all done")
        return out


# ---------- alive.py: start / stop learn.py by the power source ----------

def pi_temp_c(path="/sys/class/thermal/thermal_zone0/temp"):
    try:
        with open(path) as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


class LearnSupervisor:
    """In alive.py: every LEARN_CHECK_S the battery probe (read-only, robot/power.battery_on) and the Pi's temperature.
    Battery off LEARN_OFF_CHECKS times in a row (the Pi on the wall adapter) and cool enough -> start code/learn.py
    (its own process, nice 19); battery back or too hot -> stop it (SIGTERM: the job checkpoints, the next start goes
    on). Once it finished all its jobs it is not started again until alive.py starts again."""

    def __init__(self, probe=None, temp=pi_temp_c, popen=None, clock=time.monotonic, cmd=None, log_path=None,
                 every_s=None, off_checks=None, max_temp=None):
        import subprocess
        import sys
        if probe is None:
            from .power import battery_on as probe
        self.probe, self.temp, self.popen = probe, temp, popen or subprocess.Popen
        self._clock = clock
        here = Path(__file__).resolve().parent.parent
        self.cmd = cmd or [sys.executable, str(here / "learn.py")]
        self.log_path = Path(log_path or "~/scratch/learn/learn.out").expanduser()
        self.every_s = config.LEARN_CHECK_S if every_s is None else every_s
        self.off_checks = config.LEARN_OFF_CHECKS if off_checks is None else off_checks
        self.max_temp = config.LEARN_MAX_TEMP_C if max_temp is None else max_temp
        self.proc, self.off, self.finished, self.next_t = None, 0, False, -1e9
        self._stopping, self._kill_at, self._now = False, 0.0, None
        self.why = "waiting for the first check"

    def tick(self, now=None):
        now = self._clock() if now is None else now
        self._now = now
        if now < self.next_t:
            return
        self.next_t = now + self.every_s
        if self.proc is not None and self._stopping:
            self._reap(now)
        if self.proc is not None and self.proc.poll() is not None:
            rc = self.proc.returncode
            self.finished = rc == 0 and not self._stopping
            log.info("learn: learn.py ended (%s)%s", rc, " - all jobs done for this start" if self.finished else "")
            self.proc, self._stopping = None, False
            self.why = "all done" if self.finished else f"ended ({rc})"
        try:
            on = bool(self.probe())
        except Exception:  # noqa: BLE001 - a probe error = do not know: treat as battery on (no learning)
            on = True
        t = self.temp() if self.temp else None
        hot = t is not None and t >= self.max_temp
        self.off = 0 if on else self.off + 1
        if self.proc is not None:
            if (on or hot) and not self._stopping:
                self.stop("battery on" if on else f"Pi at {t:.0f} C")
            return
        if self.finished:
            return
        if on:
            self.why = "battery on: the robot is in use"
        elif hot:
            self.why = f"waiting: Pi at {t:.0f} C"
        elif self.off >= self.off_checks:
            self.start()
        else:
            self.why = "wall power? checking again"

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        out = open(self.log_path, "a")
        try:
            self.proc = self.popen(self.cmd, stdout=out, stderr=out, stdin=None)
        except OSError as e:
            log.warning("learn: could not start learn.py (%s)", e)
            self.why = f"could not start ({e})"
            return
        finally:
            out.close()
        self._stopping = False
        self.why = "learning (wall power)"
        log.info("learn: on wall power - learning started (pid %s)", getattr(self.proc, "pid", "?"))

    def stop(self, why="", wait=False, timeout=15.0):
        """SIGTERM to learn.py (it checkpoints and ends). wait=False (the camera loop): return at once - the next
        tick reaps it, or kills it after `timeout`. wait=True (alive.py stopping): wait for it."""
        if self.proc is None:
            return
        import subprocess
        if not self._stopping:
            log.info("learn: stopping learn.py (%s)", why)
            self._stopping, self._kill_at = True, (self._now if self._now is not None else self._clock()) + timeout
            self.why = f"stopped: {why}" if why else "stopped"
            try:
                self.proc.terminate()
            except OSError:
                pass
        if wait:
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc, self._stopping = None, False

    def _reap(self, now):
        """A learn.py that was told to stop: gone yet? Killed when it takes too long."""
        if self.proc.poll() is not None:
            self.proc, self._stopping = None, False
        elif now >= self._kill_at:
            log.warning("learn: learn.py did not stop in time - killed")
            self.proc.kill()

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None
