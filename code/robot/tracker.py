"""
Face tracks: the same face from frame to frame, so the robot decides who someone is over several looks
(votes), never from a single frame, and knows when a person appears or leaves.

Positions are head angles (pan + = left, tilt + = up, degrees), i.e. where the head would have to look -
not pixels - so the head turning between frames does not break a track.

    tracker = Tracker()
    new, ended = tracker.update([Sighting(face, pan, tilt, size_deg), ...], t)
    for tr in tracker.tracks:
        if tracker.wants_embedding(tr, t):
            pid, sim = store.match(vec)
            if tracker.vote(tr, pid, sim, t):     # True when the track's identity was just decided
                ...
"""
import itertools
import math
from collections import Counter, deque
from dataclasses import dataclass, field

from . import config, gaze


@dataclass
class Sighting:
    face: object            # faces.Face in camera-frame pixels
    pan: float              # head angles that would centre this face
    tilt: float
    size_deg: float         # face width as an angle (bigger = closer)


@dataclass
class Track:
    id: int
    sighting: Sighting
    first_t: float
    last_t: float
    hits: int = 1
    person: str = None      # decided identity (people.py id), or None
    sim: float = 0.0        # best similarity behind the decision
    unknown: bool = False   # decided: nobody the robot knows
    votes: deque = field(default_factory=deque)
    seen_as: Counter = field(default_factory=Counter)   # every person this track matched, and how often
    last_vote: str = None   # result of the latest look (None = matched nobody)
    last_embed_t: float = -math.inf
    face_q: float = None    # best face quality of its looks (None = not checked yet); see is_face
    swap_looks: int = 0     # looks in a row that showed someone else (vision.py splits the track)
    engagement: float = 0.0     # 0..1: how much the face has been turned toward the robot lately (robot/gaze.py)
    looking_since: float = None  # frame time it started looking at the robot (None = not looking)
    pos0: tuple = None          # (pan, tilt) where it started
    max_dev: float = 0.0        # furthest it moved from there (deg)
    path: deque = field(default_factory=lambda: deque(maxlen=config.TRACK_PATH_N))   # (t, pan, tilt) while the
                                # head was still (a moving head's pose is less sure): how much the face moves
    picture: bool = False       # vision.py: a face that never moves (a photo on the wall), or a person known
                                # as a picture: never counted as looking at the robot
    _still_cache: tuple = None
    gaze_t: float = None        # frame time of the last gaze update
    big_gaze_t: float = None    # ...of the last one from big-stream landmarks (the good ones)
    gaze_raw: tuple = None      # (yaw, pitch, big) of the last gaze update - recorded, for calibrating

    @property
    def pan(self):
        return self.sighting.pan

    @property
    def tilt(self):
        return self.sighting.tilt

    def spread(self):
        """(seconds of still-head positions, 90th percentile distance from their median in deg)."""
        if len(self.path) < 2:
            return 0.0, 0.0
        pans = sorted(p[1] for p in self.path)
        tilts = sorted(p[2] for p in self.path)
        mp, mt = pans[len(pans) // 2], tilts[len(tilts) // 2]
        d = sorted(math.hypot(p[1] - mp, p[2] - mt) for p in self.path)
        return self.path[-1][0] - self.path[0][0], d[min(len(d) - 1, int(0.9 * len(d)))]

    @property
    def still(self):
        """Never moved for at least TRACK_STILL_S of still-head frames: a picture, not a person. Living people
        sway and turn (their tracks: 1-30 deg); a framed photo stayed within 0.2-0.6 deg for minutes
        (2026-11-01). Cached for a second (the camera loop asks every frame)."""
        t = self.last_t
        if self._still_cache is None or t - self._still_cache[0] >= 1.0:
            secs, p90 = self.spread()
            self._still_cache = (t, secs >= config.TRACK_STILL_S and p90 <= config.TRACK_STILL_DEG)
        return self._still_cache[1]

    @property
    def moved(self):
        """Clearly moved (more than a picture's jitter): evidence of a living person."""
        return self.spread()[1] > config.TRACK_STILL_DEG

    @property
    def looking(self):
        return self.looking_since is not None

    def looked_for(self, t):
        """Seconds it has been looking at the robot without a break (0 when not looking)."""
        return 0.0 if self.looking_since is None else t - self.looking_since

    def update_gaze(self, kps, t, big=False):
        """Smooth facing(kps) into engagement; looking on / off with hysteresis. big = landmarks from the big
        stream (vision.py); small-stream ones only count while there are no recent big ones."""
        if self.picture:
            self.engagement, self.looking_since = 0.0, None
            return
        if big:
            self.big_gaze_t = t
        elif self.big_gaze_t is not None and t - self.big_gaze_t < config.GAZE_SMALL_AFTER_S:
            return
        dt = 0.1 if self.gaze_t is None else min(1.0, max(0.0, t - self.gaze_t))
        self.gaze_t = t
        self.gaze_raw = tuple(round(float(v), 3) for v in gaze.head_turn(kps)) + (bool(big),)
        k = 1.0 - math.exp(-dt / config.GAZE_TAU_S)
        self.engagement += (gaze.facing(kps) - self.engagement) * k
        if self.looking_since is None and self.engagement >= config.GAZE_LOOKING_ON:
            self.looking_since = t
        elif self.looking_since is not None and self.engagement < config.GAZE_LOOKING_OFF:
            self.looking_since = None

    @property
    def is_face(self):
        """A real face (checked on the big stream) - not a lamp, a hand or a pattern the fast detector liked."""
        return self.face_q is not None and self.face_q >= config.FACE_CONFIRM_QUALITY

    @property
    def decided(self):
        return self.person is not None or self.unknown


class Tracker:
    def __init__(self, max_missing_s=config.TRACK_MAX_MISSING_S, n_votes=config.TRACK_ID_VOTES,
                 embed_every_s=config.TRACK_EMBED_EVERY_S, recheck_s=config.TRACK_RECHECK_S,
                 match_sim=config.FACE_MATCH_SIM, face_max_missing_s=config.TRACK_FACE_MAX_MISSING_S):
        self.max_missing_s, self.face_max_missing_s = max_missing_s, face_max_missing_s
        self.n_votes = n_votes
        self.embed_every_s, self.recheck_s = embed_every_s, recheck_s
        self.match_sim = match_sim
        self.tracks = []
        self._ids = itertools.count(1)

    def update(self, sightings, t, slack_deg=0.0, still=True, bg=None):
        """Match this frame's sightings to the tracks (nearest first, within about one face width of the TRACK's
        face, plus slack_deg when the head moved and its pose is less sure). still = the head was still for
        this frame (its positions count for Track.still); bg = (dpan, dtilt) the camera's true pose minus the
        commanded one (camera.Background): path positions are then relative to the background, so the servos'
        jitter does not make a photo look alive. Returns (new tracks, ended tracks)."""
        bp, bt = bg if bg is not None else (0.0, 0.0)
        pairs = []
        for i, s in enumerate(sightings):
            for tr in self.tracks:
                d = math.hypot(s.pan - tr.pan, s.tilt - tr.tilt)
                # the track's own size, not the larger one: a huge false box (40-60 deg) used to swallow a
                # small face 30 deg away; and a face does not change size by half between two frames
                a, b = s.size_deg, tr.sighting.size_deg
                if (d <= max(3.0, 1.2 * b) + slack_deg
                        and max(a, b) <= config.TRACK_SIZE_RATIO * max(1e-6, min(a, b))):
                    pairs.append((d, i, tr))
        used, moved = set(), set()
        for d, i, tr in sorted(pairs, key=lambda p: p[0]):
            if i in used or tr.id in moved:
                continue
            used.add(i)
            moved.add(tr.id)
            if sightings[i].face is not None:
                tr.update_gaze(sightings[i].face.kps, t)
            tr.sighting, tr.last_t, tr.hits = sightings[i], t, tr.hits + 1
            if tr.pos0 is not None:
                tr.max_dev = max(tr.max_dev, math.hypot(tr.pan - tr.pos0[0], tr.tilt - tr.pos0[1]))
            if still:
                tr.path.append((t, tr.pan + bp, tr.tilt + bt))
        new = []
        for i, s in enumerate(sightings):
            if i not in used:
                tr = Track(next(self._ids), s, t, t, votes=deque(maxlen=self.n_votes), pos0=(s.pan, s.tilt))
                if still:
                    tr.path.append((t, s.pan + bp, s.tilt + bt))
                self.tracks.append(tr)
                new.append(tr)
        ended = [tr for tr in self.tracks
                 if t - tr.last_t > (self.face_max_missing_s if tr.is_face else self.max_missing_s)]
        self.tracks = [tr for tr in self.tracks if tr not in ended]
        return new, ended

    def visible(self, t, max_age=0.15, faces_only=False):
        """Tracks seen in the latest frame(s); faces_only: just the ones confirmed as real faces."""
        return [tr for tr in self.tracks if t - tr.last_t <= max_age and (tr.is_face or not faces_only)]

    def wants_embedding(self, tr, t):
        """Embed this track now? Undecided: every embed_every_s; decided: every recheck_s (people swap places)
        - but soon again when the latest look disagreed with the decision (a duplicate id, or a swap)."""
        settled = tr.decided and tr.last_vote == tr.person
        wait = self.recheck_s if settled else self.embed_every_s
        return tr.last_t == t and t - tr.last_embed_t >= wait

    def vote(self, tr, pid, sim, t):
        """Add one recognition result (best person + similarity) to the track. A person is decided when most
        of the last n votes name them (a very confident look counts at once); 'unknown' only when ALL n votes
        matched nobody - a new person gets enrolled from that, and a known face that merely looked bad for a
        frame or two must not become a second person. Returns True when the decision changed."""
        tr.last_embed_t = t
        v = pid if pid is not None and sim >= self.match_sim else None
        tr.votes.append(v)
        tr.last_vote = v
        if v is not None:
            tr.seen_as[v] += 1
        before = (tr.person, tr.unknown)
        counts = Counter(tr.votes)
        named = [(c, p) for p, c in counts.items() if p is not None]
        if v is not None and sim >= self.match_sim + 0.2 and not before[0]:
            tr.person, tr.unknown = v, False        # one very clear look decides a new track...
        elif named and max(named)[0] >= self.n_votes // 2 + 1:
            tr.person, tr.unknown = max(named)[1], False   # ...changing a decision takes a majority
        elif not before[0] and len(tr.votes) == self.n_votes and counts[None] == self.n_votes:
            tr.person, tr.unknown = None, True      # never for a recognised track: it follows one face
                                                    # without a gap, so bad looks are bad light, not a stranger
        if tr.person is not None and tr.person == v:
            tr.sim = max(tr.sim, sim) if tr.person == before[0] else sim
        return (tr.person, tr.unknown) != before

    def split(self, tr, t, moved=None):
        """The track jumped to another face: end it (returned, for face_lost / evidence) and continue the face
        it follows now as a new, undecided track. moved = the person its last swap looks showed: those looks
        belong to the new face, so they are taken out of the old track's evidence."""
        if moved is not None and tr.seen_as.get(moved):
            tr.seen_as[moved] -= min(tr.seen_as[moved], tr.swap_looks - 1)   # the last one was never voted
            if tr.seen_as[moved] <= 0:
                del tr.seen_as[moved]
        new = Track(next(self._ids), tr.sighting, t, tr.last_t, votes=deque(maxlen=self.n_votes),
                    face_q=tr.face_q, engagement=tr.engagement, looking_since=tr.looking_since,
                    gaze_t=tr.gaze_t, big_gaze_t=tr.big_gaze_t)
        self.tracks[self.tracks.index(tr)] = new
        return new

    def rename(self, old, new):
        """People ids old and new were merged: every track now says new (decision, votes, evidence)."""
        for tr in self.tracks:
            if tr.person == old:
                tr.person = new
            if tr.last_vote == old:
                tr.last_vote = new
            tr.votes = deque((new if v == old else v for v in tr.votes), maxlen=tr.votes.maxlen)
            if old in tr.seen_as:
                tr.seen_as[new] += tr.seen_as.pop(old)

    @staticmethod
    def same_person_evidence(tr, min_each=2):
        """Pairs of person ids that this one track matched at least min_each times each: one face followed
        without a gap, so they are the same person (people.PeopleStore.link). Once is not enough - two
        people standing close together can swap a track for a frame."""
        ids = sorted(p for p, n in tr.seen_as.items() if n >= min_each)
        return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]

    def seen_together(self, t):
        """Pairs of decided persons visible in the same frame: two different people (PeopleStore.apart)."""
        ids = sorted({tr.person for tr in self.visible(t) if tr.person is not None})
        return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
