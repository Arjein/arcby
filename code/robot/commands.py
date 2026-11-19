"""
Changes Selim makes on the dashboard (pi/dashboard "People" / "Heard" tabs) to what the robot knows. alive.py holds the
people store in memory, so the dashboard never edits its files itself: it drops a request into COMMAND_DIR and the
robot carries it out (`CommandBox`, a thread in alive.py) - or, when alive.py is not running,
`people_cli.py apply FILE` does the same with the same code.

    <id>.json        {"op": "merge", "keep": "009cf37c", "drop": "7e416de7"}      (the dashboard writes it)
    <id>.done.json   {"ok": true, "message": "merged ..."}                          (written back, request removed)

Ops: name / call / merge / forget / picture (a person), relabel_look / delete_look (one recorded face);
drive_stop / drive_start {"mode": "explore"} (the Live tab's Drive card: robot/legs.py, alive.py only - DriveCommands).
Every label Selim gives is also kept in LABELS_FILE: ground truth for the learning jobs (robot/learning.py).
Forgetting removes the person's vectors, voice prints, photo AND the face crops recorded of them (KVKK).
"""
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

PID = re.compile(r"^[0-9a-f]{8}$")
OPS = ("name", "call", "merge", "forget", "picture", "relabel_look", "delete_look")


class CommandError(ValueError):
    pass


def _name(v, field="name"):
    if v is None or v == "":
        return None
    if not isinstance(v, str) or len(v) > 40 or not v.strip() or any(ord(c) < 32 for c in v):
        raise CommandError(f"{field}: up to 40 printable characters")
    return v.strip()


def _pid(cmd, key, store):
    v = cmd.get(key)
    if not isinstance(v, str) or not PID.match(v) or v not in store.index:
        raise CommandError(f"{key}: no such person {v!r}")
    return v


def look_roots():
    """Where recorded faces may live (a look path must be inside one of them)."""
    return [Path(config.DATA_DIR).expanduser().resolve(), Path(config.COLLECT_DIR).expanduser().resolve()]


def _look_path(cmd, roots=None):
    p = cmd.get("face")
    if not isinstance(p, str) or not p.endswith("-face.png"):
        raise CommandError("face: the path of a recorded *-face.png")
    real = Path(p).expanduser().resolve()
    if not any(str(real).startswith(str(r) + os.sep) for r in (roots or look_roots())):
        raise CommandError("face: not a recorded face")
    return real


class PeopleCommands:
    """Carries out one command on the people store (+ voice prints, recorded crops, labels)."""

    def __init__(self, store, voices=None, embedder=None, labels=None, roots=None, clock=time.time):
        self.store, self.voices, self._embedder = store, voices, embedder
        self.labels = Path(labels or config.LABELS_FILE).expanduser()
        self.roots = roots
        self._clock = clock

    @property
    def embedder(self):
        if self._embedder is None:
            from .faces import load_embedder
            self._embedder = load_embedder(threads=1)
        return self._embedder

    def _label(self, **row):
        row.setdefault("t", round(self._clock(), 1))
        self.labels.parent.mkdir(parents=True, exist_ok=True)
        with open(self.labels, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _vector_of(self, face):
        from PIL import Image
        return self.embedder.embed(np.asarray(Image.open(face).convert("RGB")))

    def apply(self, cmd):
        """-> (ok, message). Never raises for a bad command (the dashboard shows the message)."""
        try:
            op = cmd.get("op")
            if op not in OPS:
                raise CommandError(f"unknown op {op!r}")
            return True, getattr(self, "_" + op)(cmd)
        except (CommandError, KeyError, ValueError) as e:
            return False, str(e)
        except OSError as e:
            return False, f"file error: {e}"

    def _name(self, cmd):
        pid = _pid(cmd, "id", self.store)
        name = _name(cmd.get("name"))
        self.store.set_name(pid, name)
        self._label(op="name", person=pid, name=name)
        return f"{pid} is now {name or '(no name)'}"

    def _call(self, cmd):
        pid = _pid(cmd, "id", self.store)
        call = _name(cmd.get("call"), "call")
        self.store.set_call(pid, call)
        return f"{pid} is called {call or 'by their name'}"

    def _merge(self, cmd):
        keep, drop = _pid(cmd, "keep", self.store), _pid(cmd, "drop", self.store)
        if keep == drop:
            raise CommandError("merge: the same person twice")
        self.store.merge(keep, drop)            # ValueError when they were seen together (two people)
        if self.voices is not None:
            self.voices.merge(keep, drop)
        self._label(op="merge", keep=keep, drop=drop)
        return f"{drop} merged into {keep} ({self.store.name(keep) or 'no name'})"

    def _forget(self, cmd):
        pid = _pid(cmd, "id", self.store)
        name = self.store.name(pid)
        self.store.forget(pid)
        if self.voices is not None:
            self.voices.forget(pid)
        crops = forget_recorded(pid, roots=self.roots)
        self._label(op="forget", person=pid)
        return f"{name or pid} forgotten: face, voice prints and {crops} recorded face pictures deleted"

    def _picture(self, cmd):
        pid = _pid(cmd, "id", self.store)
        state = cmd.get("state")
        if state not in ("on", "off", "auto"):
            raise CommandError("picture: state on / off / auto")
        self.store.set_picture(pid, {"on": True, "off": False, "auto": None}[state])
        return f"{pid}: photo on the wall = {state}"

    def _relabel_look(self, cmd):
        face = _look_path(cmd, self.roots)
        to = cmd.get("to")
        vec = self._vector_of(face)
        if to == "new":
            pid = self.store.enrol([vec])
            removed = self.store.remove_like(vec, keep=pid)
            msg = f"a new person {pid}"
        else:
            pid = _pid(cmd, "to", self.store)
            removed = self.store.remove_like(vec, keep=pid)
            kept = self.store.add_vector(pid, vec)
            msg = f"{self.store.name(pid) or pid}" + ("" if kept else " (already had this face)")
        self._label(op="relabel", face=str(face), person=pid)
        gone = ", ".join(f"{n} from {self.store.name(p) or p}" for p, n in removed.items())
        return f"this face is {msg}" + (f"; removed {gone}" if gone else "")

    def _delete_look(self, cmd):
        face = _look_path(cmd, self.roots)
        removed = {}
        if face.exists():
            try:
                removed = self.store.remove_like(self._vector_of(face))
            except Exception as e:  # noqa: BLE001 - an unreadable picture: still delete the files
                log.warning("commands: %s not embedded (%s)", face, e)
        n = _delete_files(face)
        self._label(op="delete", face=str(face))
        gone = ", ".join(f"{k} from {self.store.name(p) or p}" for p, k in removed.items())
        return f"deleted ({n} files)" + (f"; stored face removed: {gone}" if gone else "")


def _delete_files(face):
    """face.png and its crop.jpg next to it."""
    n = 0
    for p in (face, face.with_name(face.name[:-len("-face.png")] + "-crop.jpg")):
        try:
            p.unlink()
            n += 1
        except FileNotFoundError:
            pass
    return n


def forget_recorded(pid, roots=None, match_sim=None):
    """Delete every recorded face picture of pid: looks.jsonl lines whose person is pid, or whose best match was pid
    and confident (>= FACE_MATCH_SIM). The lines stay (the files are gone: the dashboard skips them)."""
    match_sim = config.FACE_MATCH_SIM if match_sim is None else match_sim
    n = 0
    for root in roots or look_roots():
        for lj in list(root.glob("*/*/faces/looks.jsonl")) + list(root.glob("looks.jsonl")):
            try:
                lines = lj.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("person") == pid or (m.get("match") == pid and (m.get("sim") or 0) >= match_sim):
                    if _delete_files(lj.parent / m.get("face", "x-face.png")):
                        n += 1
    return n


DRIVE_OPS = ("drive_stop", "drive_start")
DRIVE_START_MODES = ("explore",)


class DriveCommands:
    """The Drive card's requests go to alive.py's drive (robot/legs.py); every other op to the people handler."""

    def __init__(self, people, legs=None, off_why="no drive"):
        self.people, self.legs, self.off_why = people, legs, off_why

    def apply(self, cmd):
        op = cmd.get("op") if isinstance(cmd, dict) else None
        if op not in DRIVE_OPS:
            return self.people.apply(cmd)
        if self.legs is None:
            return op == "drive_stop", f"no drive ({self.off_why})"     # a stop with nothing to stop is fine
        if op == "drive_stop":
            return True, "stopped" if self.legs.stop("dashboard STOP") else "it was not driving"
        mode = cmd.get("mode", "explore")
        if mode not in DRIVE_START_MODES:
            return False, f"mode: one of {', '.join(DRIVE_START_MODES)}"
        ok, why = self.legs.start(mode, why="dashboard", max_s=config.WANDER_MAX_S)
        return ok, f"{mode} started (at most {config.WANDER_MAX_S:.0f} s)" if ok else why


class CommandBox:
    """COMMAND_DIR, checked every `every_s` in its own thread (alive.py): each request applied once, the result
    written next to it."""

    def __init__(self, handler, folder=None, every_s=1.0):
        self.handler = handler
        self.dir = Path(folder or config.COMMAND_DIR).expanduser()
        self.every_s = every_s
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="commands", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.every_s):
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - a bad request must not stop the next ones
                log.exception("commands: poll failed")

    def poll(self):
        done = 0
        for req in sorted(self.dir.glob("*.json")):
            if req.name.endswith(".done.json") or req.name.endswith(".tmp.json"):
                continue
            done += apply_file(req, self.handler)
        return done

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def apply_file(req, handler):
    """One request file -> its .done.json (the request removed). 1 if handled."""
    req = Path(req)
    try:
        cmd = json.loads(req.read_text(encoding="utf-8"))
        if not isinstance(cmd, dict):
            raise ValueError("not a JSON object")
        ok, msg = handler.apply(cmd)
    except (OSError, ValueError) as e:
        ok, msg = False, f"bad request: {e}"
    log.info("commands: %s -> %s", req.name, msg)
    out = req.with_name(req.name[:-len(".json")] + ".done.json")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({"ok": ok, "message": msg}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    try:
        req.unlink()
    except FileNotFoundError:
        pass
    return 1
