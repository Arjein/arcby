"""
Learning job: FACE_MATCH_SIM from the robot's own recordings (docs/vision.md L5; first done by hand
on 2026-11-02). No labels needed:

  - the same person   = looks of one face track (the tracker followed one face);
  - different people  = looks of two tracks seen in the same frame (frames.jsonl);
  - against the people store, as the robot decides: a track's person = the stored person its good looks match
    clearly (median); each look is scored against that person (leave-one-out: a stored vector that IS this look is
    skipped) and against everyone else.

Threshold = judged as the robot decides: by TRACKS (the first 3 good looks vote; vision.py). The lowest value where the
vote names the wrong person for at most TRACK_WRONG_MAX of the known tracks, and looks of whoever was in the frame with
person X reach X at most COVIS_SLACK more often than at today's value - more of each person's tracks recognised,
strangers not let in more than now. (First rule, single looks: others <= 1 %, wrong <= 0.5 % - never reachable on the
robot's real recordings of 2026-11-02, 3369 looks: at 0.50 still 3.7 % / 4.2 %; mostly the brothers, a duplicate of
one of them and the framed photo on the wall.) Embeddings of the looks are cached (LEARN_CACHE_DIR/faces-<embedder>.npz): a second run only embeds the
new ones.
"""
import gzip
import json
import zlib
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config
from .learning import Change, Job, Result

log = logging.getLogger(__name__)

GOOD_Q = 0.5            # looks at least this good count (lamps / hands / blur are below)
IDENT_MIN = 0.40        # a track is person X when its good looks' median similarity to X is this high...
IDENT_MARGIN = 0.08     # ...and this much above the next person's
IDENT_LOOKS = 3
TRACK_WRONG_MAX = 0.01  # at most 1 % of known tracks voted to the wrong person
COVIS_SLACK = 0.02      # others in the frame may reach X at most this much more often than at today's threshold
VOTE_LOOKS = 3          # the first good looks of a track that vote (vision.py decides a track's person the same way)
MIN_OWN, MIN_COVIS = 200, 100   # fewer looks than this: no change
AUTO_STEP = 0.03        # a change this small applies by itself; larger ones wait for Selim
GRID = np.round(np.arange(0.30, 0.5001, 0.01), 2)


# ---------- reading the recordings ----------

def find_looks(data_dir=None, collect_dir=None):
    """Every recorded look with its aligned face file: [dict(session, track, time, quality, width_px, path)]."""
    out = []
    share = Path(data_dir or config.DATA_DIR).expanduser()
    sources = [(lj.parent, f"{lj.parent.parent.parent.name}/{lj.parent.parent.name}", lj)
               for lj in sorted(share.glob("*/*/faces/looks.jsonl"))]
    cj = Path(collect_dir or config.COLLECT_DIR).expanduser() / "looks.jsonl"
    if cj.exists():
        sources.append((cj.parent, "collect", cj))
    for root, session, lj in sources:
        try:
            lines = lj.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                m = json.loads(line)
                p = root / m["face"]
            except (ValueError, KeyError, TypeError):
                continue
            q = m.get("quality")
            out.append(dict(session=session, track=m.get("track"), time=float(m.get("time") or 0.0),
                            quality=1.0 if q is None else float(q), width_px=float(m.get("width_px") or 0.0),
                            path=str(p)))
    return out


def seen_together(data_dir, sessions):
    """{(session, track): {other tracks in the same frame}} from each session's frames.jsonl.gz."""
    together = defaultdict(set)
    share = Path(data_dir or config.DATA_DIR).expanduser()
    for sess in sessions:
        f = share / sess / "frames.jsonl.gz"
        try:
            fh = gzip.open(f, "rt")
        except OSError:
            continue
        with fh:
            try:
                for line in fh:
                    try:
                        ids = [t["id"] for t in json.loads(line).get("tracks", [])]
                    except (ValueError, KeyError, TypeError, AttributeError):
                        continue
                    for a in ids:
                        for b in ids:
                            if a != b:
                                together[(sess, a)].add(b)
            except (OSError, EOFError, zlib.error):   # a session cut by a power loss: what was read counts
                pass
    return together


# ---------- the analysis (pure: tests feed it numbers) ----------

def store_scores(V, store):
    """(looks x people) best similarity to each person's vectors, leave-one-out (a stored copy of the look itself,
    sim > 0.995, is skipped). store = {pid: vectors}. Returns (S, pids)."""
    pids = sorted(p for p, v in store.items() if len(v))
    S = np.full((len(V), len(pids)), -1.0)
    for j, pid in enumerate(pids):
        s = V @ np.asarray(store[pid], np.float32).T
        s[s > 0.995] = -1.0
        S[:, j] = s.max(1)
    return S, pids


def analyse(looks, S, together, current=None):
    """looks = [dict(session, track, quality)], S = store_scores(). Returns a report dict with the threshold."""
    current = config.FACE_MATCH_SIM if current is None else current
    good = np.array([lk["quality"] >= GOOD_Q for lk in looks], bool)
    tracks = defaultdict(list)
    for i, lk in enumerate(looks):
        tracks[(lk["session"], lk["track"])].append(i)
    ident = {}
    for k, ii in tracks.items():
        g = [i for i in ii if good[i]]
        if len(g) < IDENT_LOOKS or S.shape[1] < 1:
            continue
        med = np.median(S[g], axis=0)
        o = np.argsort(med)[::-1]
        second = med[o[1]] if len(o) > 1 else -1.0
        if med[o[0]] >= IDENT_MIN and med[o[0]] - second >= IDENT_MARGIN:
            ident[k] = int(o[0])
    own, oth = [], []
    for k, j in ident.items():
        for i in tracks[k]:
            if good[i]:
                own.append(S[i, j])
                oth.append(np.max(np.delete(S[i], j)) if S.shape[1] > 1 else -1.0)
    own, oth = np.array(own), np.array(oth)
    covis = []
    for k, j in ident.items():
        for other in together.get(k, ()):
            k2 = (k[0], other)
            if ident.get(k2) == j:
                continue
            covis += [S[i, j] for i in tracks.get(k2, []) if good[i]]
    covis = np.array(covis)
    first = {k: [i for i in tracks[k] if good[i]][:VOTE_LOOKS] for k in ident}

    def vote(T):
        """(right, wrong, undecided) shares of the known tracks at threshold T."""
        r = w = u = 0
        for k, j in ident.items():
            votes = [int(np.argmax(S[i])) if S[i].max() >= T else -1 for i in first[k]]
            top = max(set(votes), key=votes.count)
            if top >= 0 and votes.count(top) * 2 > len(votes):
                r, w = r + (top == j), w + (top != j)
            else:
                u += 1
        n = max(1, r + w + u)
        return r / n, w / n, u / n

    rows = []
    for T in list(GRID) + ([round(current, 2)] if round(current, 2) not in GRID else []):
        right, wrong_t, undecided = vote(T)
        rows.append(dict(t=float(T), own=float(np.mean(own >= T)) if len(own) else 0.0,
                         wrong=float(np.mean((oth >= T) & (oth > own))) if len(own) else 0.0,
                         covis=float(np.mean(covis >= T)) if len(covis) else 0.0,
                         track_right=right, track_wrong=wrong_t, track_undecided=undecided))
    rows.sort(key=lambda r: r["t"])
    rep = dict(looks=len(looks), good=int(good.sum()), tracks=len(tracks), known_tracks=len(ident),
               own=len(own), covis=len(covis), rows=rows, current=current, threshold=None, why=None,
               own_p50=float(np.median(own)) if len(own) else None,
               covis_p95=float(np.percentile(covis, 95)) if len(covis) else None)
    if len(own) < MIN_OWN or len(covis) < MIN_COVIS:
        rep["why"] = f"not enough data ({len(own)} looks of known people, {len(covis)} of others in the frame)"
        return rep
    now = next(r for r in rows if abs(r["t"] - round(current, 2)) < 1e-9)
    ok = [r for r in rows if r["track_wrong"] <= TRACK_WRONG_MAX and r["covis"] <= now["covis"] + COVIS_SLACK]
    if not ok:
        rep["why"] = "no threshold keeps the wrong-person tracks under 1 % - kept as it is"
        return rep
    best = min(ok, key=lambda r: r["t"])
    rep["threshold"] = best["t"]
    return rep


def findings(rep):
    out = [f"{rep['looks']} looks ({rep['good']} good) in {rep['tracks']} tracks, {rep['known_tracks']} of known people"]
    if rep["own_p50"] is not None:
        line = f"a person's own looks: median {rep['own_p50']:.2f}"
        if rep["covis_p95"] is not None:
            line += f"; someone else in the frame: 95 % under {rep['covis_p95']:.2f}"
        out.append(line)
    cur = next((r for r in rep["rows"] if abs(r["t"] - round(rep["current"], 2)) < 1e-9), None)
    if cur:
        out.append(f"at today's {rep['current']:.2f}: tracks {cur['track_right']:.0%} right / "
                   f"{cur['track_wrong']:.1%} wrong / {cur['track_undecided']:.0%} undecided; "
                   f"{cur['covis']:.1%} of others in the frame let in")
    if rep["threshold"] is None:
        out.append(rep["why"])
    else:
        r = next(r for r in rep["rows"] if r["t"] == rep["threshold"])
        out.append(f"best: {rep['threshold']:.2f} -> tracks {r['track_right']:.0%} right / {r['track_wrong']:.1%} "
                   f"wrong / {r['track_undecided']:.0%} undecided; {r['covis']:.1%} of others let in")
    return out


# ---------- the job ----------

class FaceThresholdJob(Job):
    name = "faces"
    title = "Face threshold from the recordings"

    def __init__(self, embedder=None, data_dir=None, collect_dir=None, people_dir=None, cache_dir=None,
                 load_image=None):
        self._embedder = embedder
        self.data_dir, self.collect_dir = data_dir, collect_dir
        self.people_dir = people_dir or config.PEOPLE_DIR
        self.cache = Path(cache_dir or config.LEARN_CACHE_DIR).expanduser() / f"faces-{config.FACE_EMBEDDER}.npz"
        self._load_image = load_image
        self.looks, self.vecs, self.todo = [], {}, []
        self.report = None

    def _read_cache(self):
        try:
            with np.load(self.cache, allow_pickle=False) as z:
                return dict(zip((str(p) for p in z["paths"]), z["vecs"]))
        except (OSError, ValueError, KeyError):
            return {}

    def _write_cache(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache.with_name(self.cache.stem + ".tmp.npz")
        np.savez(tmp, paths=np.array(list(self.vecs), dtype=str),
                 vecs=np.array(list(self.vecs.values()), np.float32).reshape(len(self.vecs), -1))
        tmp.replace(self.cache)

    def prepare(self):
        self.looks = [lk for lk in find_looks(self.data_dir, self.collect_dir) if Path(lk["path"]).exists()]
        self.vecs = self._read_cache()
        self.todo = [lk for lk in self.looks if lk["path"] not in self.vecs]
        return len(self.todo) + 1          # + 1 = the analysis

    def _embedder_now(self):
        if self._embedder is None:
            from .faces import load_embedder
            self._embedder = load_embedder(threads=config.LEARN_THREADS)
        return self._embedder

    def _image(self, path):
        if self._load_image is not None:
            return self._load_image(path)
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))

    def run(self, progress):
        emb = self._embedder_now() if self.todo else None
        for n, lk in enumerate(self.todo):
            if progress.stopping:
                self._write_cache()
                return Result(findings=[f"stopped after {n} new looks (kept, the next run goes on)"])
            try:
                self.vecs[lk["path"]] = emb.embed(self._image(lk["path"]))
            except Exception as e:  # noqa: BLE001 - one unreadable file: skip it
                log.warning("learn faces: %s (%s)", lk["path"], e)
            progress.advance(1, step="embedding the recorded faces")
            if n % 100 == 99:
                self._write_cache()
        if self.todo:
            self._write_cache()
        progress.advance(0, step="comparing")
        looks = [lk for lk in self.looks if lk["path"] in self.vecs]
        if not looks:
            progress.advance(1)
            return Result(findings=["no recorded faces yet"])
        V = np.array([self.vecs[lk["path"]] for lk in looks], np.float32)
        store = load_store(self.people_dir)
        S, _ = store_scores(V, store)
        together = seen_together(self.data_dir, {lk["session"] for lk in looks if lk["session"] != "collect"})
        self.report = rep = analyse(looks, S, together)
        progress.advance(1)
        res = Result(findings=findings(rep))
        if rep["threshold"] is not None and abs(rep["threshold"] - rep["current"]) >= 0.005:
            step = abs(rep["threshold"] - rep["current"])
            res.changes.append(Change("FACE_MATCH_SIM", rep["threshold"], res.findings[-1], auto=step <= AUTO_STEP))
        return res


def load_store(people_dir):
    """{pid: vectors} of the people store (read-only; no PeopleStore - it may be in use by alive.py)."""
    root = Path(people_dir).expanduser()
    try:
        index = json.loads((root / "index.json").read_text(encoding="utf-8")).get("people", {})
    except (OSError, ValueError, AttributeError):
        return {}
    out = {}
    for pid in index:
        try:
            v = np.load(root / pid / "vectors.npy")
        except (OSError, ValueError):
            continue
        if v.ndim == 2 and len(v):
            out[pid] = v.astype(np.float32)
    return out
