"""
People the robot knows: face vectors (robot/faces.py embeddings) + a photo + a name per person, on disk.

    store = PeopleStore()                       # loads config.PEOPLE_DIR
    pid, sim = store.match(vec)                 # best person and similarity (None, 0.0 when nobody)
    pid = store.enrol([vec1, vec2], photo)      # a new person, no name yet
    store.set_name(pid, "Selim")
    store.learn(pid, vec)                       # after a confident match: keep vectors from new angles / light
    store.forget(pid)                           # by hand only (people_cli.py forget): the robot never forgets anyone itself

One person seen as two ids (a bad first look -> enrolled again) is fixed with evidence, not one threshold:
  link(a, b)    one face track matched both a and b -> same person (a track follows one face without a gap)
  apart(a, b)   a and b were in the same frame at the same time -> never the same person
  merge_candidates()  pairs worth merging: automatic when the evidence is clear, otherwise "ask them"
  merge(keep, drop) / unmerge(keep, drop)    every vector remembers the id it was first stored under,
                so a wrong merge can be undone
Similarity between two people = mean of the k best vector pairs (sets, not averages: an average of a front
and a side view matches neither).

Layout: PEOPLE_DIR/index.json = {people: {id: {name, created, last_seen, seen}}, links: {"a|b": n},
apart: ["a|b"], merges: [...]}; PEOPLE_DIR/<id>/vectors.npy (N x d), origins.json (N ids), face.jpg.
Files are written to a temp name and renamed, so a crash never leaves half a file.
Faces are biometric data (KVKK): they stay on the robot, and forget() deletes them for good.
"""
import json
import logging
import math
import os
import shutil
import threading
import time
import uuid

import numpy as np

from . import config

log = logging.getLogger(__name__)

def _write_atomic(path, write, backup=False):
    """Write to path.tmp, flush it to the SD card, then rename (a power cut leaves the old or the new file, never
    half of one). backup: keep the previous version as path.bak."""
    tmp = path + ".tmp"
    write(tmp)
    try:
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
    if backup and os.path.exists(path):
        try:
            shutil.copy2(path, path + ".bak")
        except OSError:
            pass
    os.replace(tmp, path)


def _load_index(path):
    """index.json, else its .bak, else {}: a broken index (power cut) must not stop the robot from starting -
    run_alive.sh would restart it every 30 s forever."""
    for p in (path, path + ".bak"):
        try:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict):
                if p != path:
                    log.warning("people: %s was broken, using %s", path, p)
                return data
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as e:
            log.warning("people: could not read %s (%s)", p, e)
    return {}


def _pair(a, b):
    return "|".join(sorted((a, b)))


class PeopleStore:
    def __init__(self, root=config.PEOPLE_DIR, clock=time.time, max_vectors=config.PEOPLE_MAX_VECTORS,
                 learn_min=config.FACE_LEARN_MIN_SIM, learn_max=config.FACE_LEARN_MAX_SIM,
                 match_sim=config.FACE_MATCH_SIM):
        self.root = os.path.expanduser(root)
        self._clock = clock
        self.max_vectors = max_vectors
        self.learn_min, self.learn_max = learn_min, learn_max
        self.match_sim = match_sim
        self._lock = threading.RLock()
        os.makedirs(self.root, exist_ok=True)
        data = _load_index(os.path.join(self.root, "index.json"))
        self.index = data.get("people", {})
        self.links = data.get("links", {})
        self.apart_pairs = set(data.get("apart", []))
        self.merges = data.get("merges", [])
        self.vectors, self.origins = {}, {}
        for pid in self.index:
            d = os.path.join(self.root, pid)
            vp, op = os.path.join(d, "vectors.npy"), os.path.join(d, "origins.json")
            try:
                self.vectors[pid] = np.load(vp) if os.path.exists(vp) else np.zeros((0, 0), np.float32)
            except (OSError, ValueError) as e:
                log.warning("people: %s vectors unreadable (%s) - starts empty", pid, e)
                self.vectors[pid] = np.zeros((0, 0), np.float32)
            try:
                with open(op) as f:
                    self.origins[pid] = json.load(f)
            except (OSError, ValueError):
                self.origins[pid] = [pid] * len(self.vectors[pid])
            if len(self.origins[pid]) != len(self.vectors[pid]):
                self.origins[pid] = [pid] * len(self.vectors[pid])

    # ---------- reading ----------

    def match(self, vec):
        """(person id, similarity) of the best stored vector, or (None, 0.0) when nobody is stored."""
        with self._lock:
            best, sim = None, 0.0
            for pid, m in self.vectors.items():
                if len(m):
                    s = float((m @ vec).max())
                    if best is None or s > sim:
                        best, sim = pid, s
            return best, sim

    def similarity(self, pid, vec):
        """Best similarity of vec to this one person's vectors (0.0 if unknown / empty)."""
        with self._lock:
            m = self.vectors.get(pid)
            return float((m @ vec).max()) if m is not None and len(m) else 0.0

    def name(self, pid):
        return self.index.get(pid, {}).get("name")

    def call(self, pid):
        """How the robot addresses them ("Boss"); their name when no other wish was given."""
        p = self.index.get(pid, {})
        return p.get("call") or p.get("name")

    def is_picture(self, pid):
        """A face that never moves - a photo on the wall (set by the evidence of note_track, or by hand)."""
        p = self.index.get(pid, {})
        if "picture_set" in p:                      # people_cli.py picture ID on/off wins
            return bool(p["picture_set"])
        return bool(p.get("picture"))

    def people(self):
        """[(id, name, number of vectors)], newest first."""
        with self._lock:
            return sorted(((pid, p.get("name"), len(self.vectors[pid])) for pid, p in self.index.items()),
                          key=lambda x: -self.index[x[0]]["created"])

    def set_similarity(self, a, b, k=3):
        """How alike two people's vector sets are: mean of the k most similar (a, b) vector pairs."""
        with self._lock:
            ma, mb = self.vectors[a], self.vectors[b]
            if not len(ma) or not len(mb):
                return 0.0
            s = np.sort((ma @ mb.T).ravel())[::-1]
            return float(s[:k].mean())

    def merge_candidates(self, ask_sim=config.PEOPLE_MERGE_ASK_SIM, auto_links=config.PEOPLE_MERGE_LINKS):
        """Pairs of ids that may be one person, most likely first: [dict(keep, drop, sim, links, same_name,
        auto)]. Never pairs seen together in one frame. auto = clear enough to merge without asking:
        a track matched both at least auto_links times and their sets look alike, or both carry the same
        name and look alike. The rest (similar sets only) are for asking the person."""
        with self._lock:
            ids = sorted(self.index, key=lambda p: self.index[p]["created"])
            out = []
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    key = _pair(a, b)
                    if key in self.apart_pairs:
                        continue
                    sim = self.set_similarity(a, b)
                    links = self.links.get(key, 0)
                    na, nb = self.name(a), self.name(b)
                    same_name = na is not None and na == nb
                    alike = sim >= self.match_sim
                    auto = (links >= auto_links and sim >= self.match_sim - 0.05) or (same_name and alike)
                    if auto or links or sim >= ask_sim:
                        # keep the named one, else the older one (ids are sorted oldest first)
                        keep, drop = (b, a) if (nb is not None and na is None) else (a, b)
                        out.append(dict(keep=keep, drop=drop, sim=sim, links=links, same_name=same_name, auto=auto))
            return sorted(out, key=lambda c: (not c["auto"], -c["links"], -c["sim"]))

    # ---------- writing ----------

    def enrol(self, vectors, photo=None):
        """Store a new person (no name yet) from one or more vectors; photo = HxWx3 uint8 RGB or None."""
        pid = uuid.uuid4().hex[:8]
        now = self._clock()
        with self._lock:
            os.makedirs(os.path.join(self.root, pid), exist_ok=True)
            self.vectors[pid] = np.array(vectors, np.float32).reshape(len(vectors), -1)
            self.origins[pid] = [pid] * len(vectors)
            self.index[pid] = dict(name=None, created=now, last_seen=now, seen=1)
            self._save_vectors(pid)
            if photo is not None:
                self._save_photo(pid, photo)
            self._save_index()
        return pid

    def set_name(self, pid, name):
        with self._lock:
            self.index[pid]["name"] = name
            self._save_index()

    def set_call(self, pid, call):
        """How they want to be addressed (None = by name)."""
        with self._lock:
            if call:
                self.index[pid]["call"] = call
            else:
                self.index[pid].pop("call", None)
            self._save_index()

    def seen(self, pid):
        """Count a sighting (one per track, not per frame)."""
        with self._lock:
            p = self.index.get(pid)
            if p is None:                           # merged away or forgotten meanwhile
                return
            p["last_seen"] = self._clock()
            p["seen"] = p.get("seen", 0) + 1
            self._save_index()

    def learn(self, pid, vec, sim=None):
        """Keep vec if the match was confident and it is not a near-copy of a stored vector.
        When full, drop the stored vector most similar to the others. Returns True if kept."""
        with self._lock:
            if pid not in self.vectors:
                return False
            m = self.vectors[pid]
            s = (m @ vec) if len(m) else np.zeros(0)
            sim = float(s.max()) if sim is None and len(m) else sim
            if sim is None or sim < self.learn_min or (len(m) and float(s.max()) >= self.learn_max):
                return False
            self.vectors[pid] = np.vstack([m, vec[np.newaxis].astype(np.float32)])
            self.origins[pid].append(pid)
            self._trim(pid)
            self._save_vectors(pid)
            return True

    def note_track(self, pid, still, at=None, mode="head", near_deg=5.0, min_still=config.PEOPLE_PICTURE_TRACKS):
        """Evidence from an ended track of this person, at = where it mostly was (head angles, in this mode:
        "head" / "nohead", vision.mode). still = it never moved for TRACK_STILL_S: its place is a still spot.
        Any other track more than near_deg from every still spot of the same mode was somewhere else: a picture on
        the wall never is. (Movement inside one place is not used: with a moving head the pose error alone made a
        photo jitter 1-1.6 deg, as much as a person sitting still.) Picture = at least min_still still tracks and
        hardly ever elsewhere (one glitch per four forgiven). Never deletes anyone; the flag only keeps a photo
        from counting as someone looking at the robot."""
        with self._lock:
            p = self.index.get(pid)
            if p is None:
                return
            spots = p.setdefault("still_spots", [])
            if still:
                p["still_tracks"] = p.get("still_tracks", 0) + 1
                if at is not None and not any(s[2] == mode and math.hypot(s[0] - at[0], s[1] - at[1]) <= near_deg
                                              for s in spots):
                    spots.append([round(at[0], 1), round(at[1], 1), mode])
                    del spots[:-10]
            elif at is not None:
                mine = [s for s in spots if s[2] == mode]
                if mine and all(math.hypot(s[0] - at[0], s[1] - at[1]) > near_deg for s in mine):
                    p["elsewhere"] = p.get("elsewhere", 0) + 1
                else:
                    return                          # same place (or no spot to compare in this mode): no news
            was = bool(p.get("picture"))
            p["picture"] = p.get("still_tracks", 0) >= min_still and p.get("elsewhere", 0) * 4 <= p["still_tracks"]
            if p["picture"] != was:
                log.info("people: %s (%s) %s", pid, p.get("name"), "looks like a picture - never moves"
                         if p["picture"] else "seen elsewhere - not a picture")
            self._save_index()

    def add_vector(self, pid, vec):
        """Selim said this face is pid (the dashboard): stored whatever the similarity (learn() would want a
        confident match) - unless a near-copy is there already. True if kept."""
        vec = np.asarray(vec, np.float32).reshape(-1)
        with self._lock:
            if pid not in self.vectors:
                raise KeyError(pid)
            m = self.vectors[pid]
            if len(m) and float((m @ vec).max()) >= 0.995:
                return False
            self.vectors[pid] = vec[None] if not len(m) else np.vstack([m, vec[None]])
            self.origins[pid] = self.origins.get(pid, []) + [pid]
            self._trim(pid)
            self._save_vectors(pid)
        return True

    def remove_like(self, vec, min_sim=0.98, keep=None):
        """Delete stored vectors that are this face (cosine >= min_sim) - from everyone except `keep`.
        Returns {pid: how many removed}."""
        vec = np.asarray(vec, np.float32).reshape(-1)
        out = {}
        with self._lock:
            for pid, m in list(self.vectors.items()):
                if pid == keep or not len(m):
                    continue
                drop = (m @ vec) >= min_sim
                if drop.any():
                    self.vectors[pid] = m[~drop]
                    self.origins[pid] = [o for o, d in zip(self.origins.get(pid, [pid] * len(m)), drop) if not d]
                    self._save_vectors(pid)
                    out[pid] = int(drop.sum())
        return out

    def set_picture(self, pid, on):
        """By hand: on (True / False) wins over the evidence; None = back to the evidence."""
        with self._lock:
            if on is None:
                self.index[pid].pop("picture_set", None)
            else:
                self.index[pid]["picture_set"] = bool(on)
            self._save_index()

    def link(self, a, b, n=1):
        """Evidence: one track matched both a and b."""
        if a == b or a not in self.index or b not in self.index:
            return
        with self._lock:
            key = _pair(a, b)
            self.links[key] = self.links.get(key, 0) + n
            self._save_index()

    def apart(self, a, b):
        """Evidence: a and b were seen at the same time, so they are two people. Returns True if new."""
        key = _pair(a, b)
        if a == b or key in self.apart_pairs:
            return False
        with self._lock:
            self.apart_pairs.add(key)
            self._save_index()
            return True

    def merge(self, keep, drop):
        """drop turned out to be keep: move its vectors (with their origins) into keep; undo with unmerge."""
        with self._lock:
            if keep == drop or keep not in self.index or drop not in self.index:
                raise KeyError(f"merge {keep} <- {drop}: one of them is gone (forgotten / merged meanwhile)")
            if _pair(keep, drop) in self.apart_pairs:
                raise ValueError(f"{keep} and {drop} were seen together - two people")
            self.merges.append(dict(keep=keep, drop=drop, t=self._clock(), entry=self.index.pop(drop)))
            self.vectors[keep] = np.vstack([self.vectors[keep], self.vectors.pop(drop)])
            self.origins[keep] += self.origins.pop(drop)
            k = self.index[keep]
            k["name"] = k.get("name") or self.merges[-1]["entry"].get("name")
            if not k.get("call") and self.merges[-1]["entry"].get("call"):
                k["call"] = self.merges[-1]["entry"]["call"]
            k["seen"] = k.get("seen", 0) + self.merges[-1]["entry"].get("seen", 0)
            self._move_evidence(drop, keep)
            self._trim(keep)
            self._save_vectors(keep)
            self._save_index()
            photo = os.path.join(self.root, drop, "face.jpg")
            if os.path.exists(photo):                   # keep the photo for an unmerge
                os.replace(photo, os.path.join(self.root, keep, f"face-{drop}.jpg"))
            shutil.rmtree(os.path.join(self.root, drop), ignore_errors=True)

    def unmerge(self, keep, drop):
        """Undo merge(keep, drop): drop comes back with the vectors that were first stored under it (or under
        ids merged into it before), and they are marked apart so they are not merged again."""
        with self._lock:
            rec = next(m for m in reversed(self.merges) if m["keep"] == keep and m["drop"] == drop)
            group = {drop} | self._merged_into(drop)
            mine = [i for i, o in enumerate(self.origins[keep]) if o in group]
            self.vectors[drop] = self.vectors[keep][mine]
            self.origins[drop] = [self.origins[keep][i] for i in mine]
            rest = [i for i in range(len(self.origins[keep])) if i not in set(mine)]
            self.vectors[keep] = self.vectors[keep][rest]
            self.origins[keep] = [self.origins[keep][i] for i in rest]
            self.index[drop] = rec["entry"]
            self.merges.remove(rec)
            self.apart_pairs.add(_pair(keep, drop))
            os.makedirs(os.path.join(self.root, drop), exist_ok=True)
            photo = os.path.join(self.root, keep, f"face-{drop}.jpg")
            if os.path.exists(photo):
                os.replace(photo, os.path.join(self.root, drop, "face.jpg"))
            self._save_vectors(keep)
            self._save_vectors(drop)
            self._save_index()

    def forget(self, pid):
        with self._lock:
            self.index.pop(pid, None)
            self.vectors.pop(pid, None)
            self.origins.pop(pid, None)
            self.links = {k: n for k, n in self.links.items() if pid not in k.split("|")}
            self.apart_pairs = {k for k in self.apart_pairs if pid not in k.split("|")}
            self._save_index()
            shutil.rmtree(os.path.join(self.root, pid), ignore_errors=True)

    # ---------- helpers (lock held) ----------

    def _merged_into(self, pid):
        """Ids merged into pid (and into those), from the merge history."""
        out = set()
        for m in self.merges:
            if m["keep"] == pid:
                out |= {m["drop"]} | self._merged_into(m["drop"])
        return out

    def _move_evidence(self, old, new):
        links = {}
        for key, n in self.links.items():
            a, b = key.split("|")
            a, b = (new if a == old else a), (new if b == old else b)
            if a != b:
                links[_pair(a, b)] = links.get(_pair(a, b), 0) + n
        self.links = links
        apart = set()
        for key in self.apart_pairs:
            a, b = key.split("|")
            a, b = (new if a == old else a), (new if b == old else b)
            if a != b:
                apart.add(_pair(a, b))
        self.apart_pairs = apart

    def _trim(self, pid):
        """Over max_vectors: drop the vector most similar to another one, until it fits."""
        m, o = self.vectors[pid], self.origins[pid]
        while len(m) > self.max_vectors:
            g = m @ m.T
            np.fill_diagonal(g, -1)
            i = int(np.argmax(g.max(axis=1)))
            m = np.delete(m, i, axis=0)
            o = o[:i] + o[i + 1:]
        self.vectors[pid], self.origins[pid] = m, o

    # ---------- files (lock held) ----------

    def _save_index(self):
        path = os.path.join(self.root, "index.json")
        data = dict(people=self.index, links=self.links, apart=sorted(self.apart_pairs), merges=self.merges)
        def write(tmp):
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1)
        try:
            _write_atomic(path, write, backup=True)
        except OSError as e:                # disk full / SD error: kept in memory, written with the next change
            log.warning("people: could not save the index (%s) - kept in memory", e)

    def _save_vectors(self, pid):
        d = os.path.join(self.root, pid)
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            log.warning("people: could not save %s's face vectors (%s) - kept in memory", pid, e)
            return
        def write_vectors(tmp):
            with open(tmp, "wb") as f:
                np.save(f, self.vectors[pid])
        def write_origins(tmp):
            with open(tmp, "w") as f:
                json.dump(self.origins[pid], f)
        try:
            _write_atomic(os.path.join(d, "vectors.npy"), write_vectors)
            _write_atomic(os.path.join(d, "origins.json"), write_origins)
        except OSError as e:                # disk full / SD error: kept in memory, written with the next change
            log.warning("people: could not save %s's face vectors (%s) - kept in memory", pid, e)

    def _save_photo(self, pid, photo):
        from PIL import Image
        path = os.path.join(self.root, pid, "face.jpg")
        try:
            _write_atomic(path, lambda tmp: Image.fromarray(photo).save(tmp, "JPEG", quality=90))
        except OSError as e:
            log.warning("people: could not save %s's photo (%s)", pid, e)
