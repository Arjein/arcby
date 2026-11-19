"""
The dashboard's archive tabs (People, Heard): what the robot knows and kept, to look at and correct. Stdlib only,
like arcby_dashboard.py; it only READS the robot's files - every change goes to the robot as a request
(code/robot/commands.py: alive.py carries it out, or `people_cli.py apply` when alive.py is not running), so the
people store in alive.py's memory and its files never disagree. The one direct write: deleting a heard clip
(a file nothing else keeps state about).

  GET  /api/people                       people + counts (vectors, voice prints), photo, last seen
  GET  /api/people/looks?id=PID|unknown   that person's recorded faces, newest first (&offset=&n=)
  GET  /api/heard                         the kept sentences (~/scratch/alive/heard), newest first
  GET  /media/<people|data|collect|heard>/<path>   a picture / clip from those folders only
  POST /api/command                       {"op": ...} -> robot/commands.py (name, call, merge, forget, picture,
                                          relabel_look, delete_look)
  POST /api/heard-delete                  {"name": "<clip>.wav"}
"""
import json
import os
import re
import secrets
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HOME = Path.home()
SHARE = Path(os.environ.get("ARCBY_SHARE", HOME / ".local/share/arcby"))
PEOPLE_DIR = Path(os.environ.get("ARCBY_PEOPLE_DIR", SHARE / "people-r50_int8"))
DATA_DIR = Path(os.environ.get("ARCBY_DATA_DIR", SHARE / "data"))
COLLECT_DIR = Path(os.environ.get("ARCBY_COLLECT_DIR", SHARE / "collect"))
HEARD_DIR = Path(os.environ.get("ARCBY_HEARD_DIR", HOME / "scratch/alive/heard"))
COMMAND_DIR = Path(os.environ.get("ARCBY_COMMAND_DIR", HOME / "scratch/alive/commands"))
TUNED_FILE = Path(os.environ.get("ARCBY_TUNED_FILE", SHARE / "tuned.json"))
LABELS_FILE = Path(os.environ.get("ARCBY_LABELS_FILE", SHARE / "labels.jsonl"))   # robot/commands.py writes it
PEOPLE_CLI = Path(os.environ.get("ARCBY_PEOPLE_CLI", HOME / "projects/arcby/code/people_cli.py"))
DEFAULT_MATCH_SIM = 0.38            # robot/config.py FACE_MATCH_SIM (tuned.json wins)
OPS = ("name", "call", "merge", "forget", "picture", "relabel_look", "delete_look", "drive_stop", "drive_start")
DRIVE_OPS = ("drive_stop", "drive_start")      # the Live tab's Drive card: only alive.py can carry them out
PID = re.compile(r"^[0-9a-f]{8}$")
CLIP = re.compile(r"^[\w.\-]{1,120}\.wav$")
TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".wav": "audio/wav"}


def roots():
    return {"people": PEOPLE_DIR, "data": DATA_DIR, "collect": COLLECT_DIR, "heard": HEARD_DIR}


def media_url(path):
    """A file inside one of the roots -> its /media URL (None outside them)."""
    real = Path(path).resolve()
    for key, root in roots().items():
        r = root.resolve()
        if str(real).startswith(str(r) + os.sep):
            return f"/media/{key}/" + str(real.relative_to(r))
    return None


def media_file(url_path):
    """/media/<root>/<rel> -> (Path, content type) or None (outside the roots, missing, a type not served)."""
    parts = url_path.split("/", 3)                  # "", "media", root, rel
    if len(parts) < 4 or parts[1] != "media" or parts[2] not in roots():
        return None
    root = roots()[parts[2]].resolve()
    p = (root / parts[3]).resolve()
    if not str(p).startswith(str(root) + os.sep) or not p.is_file() or p.suffix.lower() not in TYPES:
        return None
    return p, TYPES[p.suffix.lower()]


def _npy_rows(data):
    """Rows of a .npy array from its header (no numpy here)."""
    if data[:6] != b"\x93NUMPY":
        return 0
    major = data[6]
    hlen = struct.unpack("<H", data[8:10])[0] if major == 1 else struct.unpack("<I", data[8:12])[0]
    head = data[10:10 + hlen] if major == 1 else data[12:12 + hlen]
    m = re.search(rb"'shape':\s*\((\d+)", head)
    return int(m.group(1)) if m else 0


def voice_counts(path=None):
    """{pid: voice prints} from voices.npz (a zip of .npy files)."""
    path = Path(path or PEOPLE_DIR / "voices.npz")
    out = {}
    try:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".npy") and not n.endswith("__t.npy"):     # robot/voices.py TIMES = "__t"
                    with z.open(n) as f:
                        out[n[:-4]] = _npy_rows(f.read(256))
    except (OSError, zipfile.BadZipFile, KeyError, struct.error):
        pass
    return out


def match_sim():
    try:
        return float(json.loads(TUNED_FILE.read_text())["values"]["FACE_MATCH_SIM"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_MATCH_SIM


def people():
    """The people the robot knows, most recently seen first."""
    try:
        index = json.loads((PEOPLE_DIR / "index.json").read_text()).get("people", {})
    except (OSError, ValueError, AttributeError):
        return {"people": [], "why": "no people store yet"}
    voices = voice_counts()
    counts, newest = look_counts()
    out = []
    for pid, p in index.items():
        d = PEOPLE_DIR / pid
        try:
            with open(d / "vectors.npy", "rb") as f:
                vecs = _npy_rows(f.read(256))
        except OSError:
            vecs = 0
        photo = d / "face.jpg"
        out.append(dict(id=pid, name=p.get("name"), call=p.get("call"), created=p.get("created"),
                        last_seen=p.get("last_seen"), seen=p.get("seen", 0), vectors=vecs, voices=voices.get(pid, 0),
                        picture=bool(p["picture_set"]) if "picture_set" in p else bool(p.get("picture")),
                        picture_by_hand="picture_set" in p, looks=counts.get(pid, 0),
                        photo=newest.get(pid) or (media_url(photo) if photo.exists() else None)))
    out.sort(key=lambda p: -(p["last_seen"] or 0))
    return {"people": out, "match_sim": match_sim(), "unknown_looks": counts.get(None, 0)}


# ---------- recorded faces ----------

_looks_cache = {"t": 0.0, "looks": []}


def all_looks(max_age_s=20.0, now=None):
    """Every recorded face whose picture still exists (the recordings' looks.jsonl + collect), newest first.
    Cached for max_age_s."""
    now = time.time() if now is None else now
    if now - _looks_cache["t"] < max_age_s:
        return _looks_cache["looks"]
    looks = []
    sources = [(lj, f"{lj.parent.parent.parent.name}/{lj.parent.parent.name}") for lj in DATA_DIR.glob("*/*/faces/looks.jsonl")]
    if (COLLECT_DIR / "looks.jsonl").exists():
        sources.append((COLLECT_DIR / "looks.jsonl", "collect"))
    for lj, session in sources:
        try:
            lines = lj.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                m = json.loads(line)
                face = lj.parent / m["face"]
            except (ValueError, KeyError, TypeError):
                continue
            if not face.exists():
                continue                            # deleted / forgotten
            crop = lj.parent / m["crop"] if m.get("crop") else None
            looks.append(dict(face=str(face), face_url=media_url(face),
                              crop_url=media_url(crop) if crop is not None and crop.exists() else None,
                              time=m.get("time"), track=m.get("track"), session=session, person=m.get("person"),
                              match=m.get("match"), sim=m.get("sim"), width_px=m.get("width_px"),
                              quality=m.get("quality")))
    looks.sort(key=lambda lk: -(lk["time"] or 0))
    _looks_cache.update(t=now, looks=looks)
    return looks


_labels_cache = {"sig": None, "faces": {}, "merged": {}}


def labels():
    """Selim's corrections (LABELS_FILE): {face path: person} from "this is..." and {dropped id: kept id} from merges.
    Re-read only when the file changed."""
    try:
        st = LABELS_FILE.stat()
        sig = (st.st_mtime, st.st_size)
    except OSError:
        return {}, {}
    if _labels_cache["sig"] != sig:
        faces, merged = {}, {}
        for line in LABELS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("op") == "relabel" and row.get("face"):
                faces[row["face"]] = row.get("person")
            elif row.get("op") == "merge" and row.get("drop"):
                merged[row["drop"]] = row.get("keep")
        _labels_cache.update(sig=sig, faces=faces, merged=merged)
    return _labels_cache["faces"], _labels_cache["merged"]


def owner(look, thr, faces=None, merged=None):
    """Whose a recorded face is: Selim's label if he gave one, else the person the robot decided, else a confident
    best match, else None - merged ids followed to the person they were merged into."""
    if faces is None:
        faces, merged = labels()
    pid = faces.get(str(Path(look["face"]).resolve())) or faces.get(look["face"])
    if pid is None:
        if look["person"]:
            pid = look["person"]
        elif look["match"] and (look["sim"] or 0) >= thr:
            pid = look["match"]
    seen = set()
    while pid in (merged or {}) and pid not in seen:
        seen.add(pid)
        pid = merged[pid]
    return pid


def look_counts():
    """{person: pictures} and {person: their newest good picture URL} (the cards show a fresh face)."""
    thr = match_sim()
    faces, merged = labels()
    out, newest = {}, {}
    for lk in all_looks():                          # newest first
        k = owner(lk, thr, faces, merged)
        out[k] = out.get(k, 0) + 1
        if k not in newest and (lk.get("quality") is None or lk["quality"] >= 0.6):
            newest[k] = lk["face_url"]
    return out, newest


def looks_of(pid, offset=0, n=60):
    thr = match_sim()
    faces, merged = labels()
    want = None if pid == "unknown" else pid
    sel = [lk for lk in all_looks() if owner(lk, thr, faces, merged) == want]
    return {"id": pid, "total": len(sel), "offset": offset, "looks": sel[offset:offset + n]}


# ---------- heard ----------

def heard(n=80):
    """The kept sentences, newest first: [{name, url, kind, text, time}] (names: [YYYYMMDD-]HHMMSS-kind[-words])."""
    out = []
    try:
        files = sorted(HEARD_DIR.glob("*.wav"), key=lambda p: -p.stat().st_mtime)
    except OSError:
        files = []
    for p in files[:n]:
        parts = p.stem.split("-")
        if parts and len(parts[0]) == 8 and parts[0].isdigit():
            parts = parts[1:]
        kind = parts[1] if len(parts) > 1 else "?"
        out.append(dict(name=p.name, url=media_url(p), kind=kind, text=" ".join(parts[2:]),
                        time=p.stat().st_mtime, bytes=p.stat().st_size))
    return {"clips": out, "why": None if out else "nothing kept yet (the ears are off while the robot is quiet)"}


def heard_delete(raw):
    d = json.loads(raw.decode("utf-8"))
    name = d.get("name") if isinstance(d, dict) else None
    if not isinstance(name, str) or not CLIP.match(name):
        raise ValueError("send {\"name\": \"<clip>.wav\"}")
    p = (HEARD_DIR / name).resolve()
    if p.parent != HEARD_DIR.resolve():
        raise ValueError("not a kept clip")
    p.unlink()
    return {"ok": True, "message": f"{name} deleted"}


# ---------- changes: requests to the robot ----------

def alive_running(proc="/proc"):
    """Is code/alive.py running (it holds the people store - changes must go through it)?"""
    try:
        pids = [d for d in os.listdir(proc) if d.isdigit()]
    except OSError:
        return False
    for pid in pids:
        try:
            with open(f"{proc}/{pid}/cmdline", "rb") as f:
                if b"code/alive.py" in f.read().replace(b"\0", b" "):
                    return True
        except OSError:
            continue
    return False


def check_command(cmd):
    """Only well-formed requests leave the dashboard (robot/commands.py checks them again)."""
    if not isinstance(cmd, dict) or cmd.get("op") not in OPS:
        raise ValueError(f"op must be one of {', '.join(OPS)}")
    for key in ("id", "keep", "drop"):
        if key in cmd and not (isinstance(cmd[key], str) and PID.match(cmd[key])):
            raise ValueError(f"{key}: a person id")
    if "to" in cmd and not (cmd["to"] == "new" or (isinstance(cmd["to"], str) and PID.match(cmd["to"]))):
        raise ValueError("to: a person id or 'new'")
    if "face" in cmd and (not isinstance(cmd["face"], str) or media_url(cmd["face"]) is None
                          or not cmd["face"].endswith("-face.png")):
        raise ValueError("face: one of the recorded faces")
    for key in ("name", "call"):
        if key in cmd and cmd[key] is not None and (not isinstance(cmd[key], str) or len(cmd[key]) > 40):
            raise ValueError(f"{key}: up to 40 characters")
    if "mode" in cmd and cmd["mode"] not in ("explore",):
        raise ValueError("mode: explore")
    return {k: cmd[k] for k in ("op", "id", "keep", "drop", "to", "face", "name", "call", "state", "mode") if k in cmd}


def command(raw, alive_running, run=subprocess.run, wait_s=20.0, sleep=time.sleep):
    """A change from the page -> the robot. alive.py running: a request file it picks up within ~1 s (its answer
    awaited); not running: people_cli.py apply carries it out now."""
    cmd = check_command(json.loads(raw.decode("utf-8")))
    drive = cmd["op"] in DRIVE_OPS
    if drive and not alive_running():
        return {"ok": cmd["op"] == "drive_stop", "message": "alive.py is not running - nothing drives"}
    if drive:
        wait_s = min(wait_s, 5.0)
    COMMAND_DIR.mkdir(parents=True, exist_ok=True)
    rid = f"{time.time():.3f}-{secrets.token_hex(3)}"
    req = COMMAND_DIR / f"{rid}.json"
    tmp = COMMAND_DIR / f"{rid}.tmp.json"
    tmp.write_text(json.dumps(cmd, ensure_ascii=False), encoding="utf-8")
    if not alive_running():
        tmp.replace(req)
        r = run([sys.executable, str(PEOPLE_CLI), "apply", str(req)], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return {"ok": False, "message": (r.stderr or r.stdout or "people_cli.py failed").strip()[-300:]}
    else:
        tmp.replace(req)
    done = COMMAND_DIR / f"{rid}.done.json"
    deadline = time.time() + wait_s
    while not done.exists():
        if time.time() > deadline:
            return {"ok": False, "message": "the robot did not answer in time (it will still do it)", "id": rid}
        sleep(0.2)
    try:
        res = json.loads(done.read_text(encoding="utf-8"))
    finally:
        done.unlink(missing_ok=True)
    _looks_cache["t"] = 0.0                         # pictures may be gone / moved: read them again
    return res
