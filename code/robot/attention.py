"""
Attention: which face to look at, or where to look when there is none. Pure logic (no hardware); its
output goes to robot/head_life.HeadLife.set_focus(). The aim (Selim, 2026-11-01): it should feel alive, like an
animal - not a face tracker.

Interest = face size (closer = bigger) x one factor per cue in CUES: who it is (known > unknown > undecided),
whether they look at the robot (Track.engagement, robot/gaze.py) - x how fresh they are (boredom, below) - x
ATTN_SPOKE_GAIN for a few seconds after their voice was heard.

Calm, not twitchy (Selim 2026-11-01: "the head moves randomly"): few, slow idle looks that come back, long holds
on a person, no wide sweeps without a reason, pictures on the wall ignored, a face the tracker merely lost and
found again is not a newcomer.

Rules, in order:
  0. Pictures (Track.picture: a photo on the wall) are not looked at.
  1. Following someone: stay with them (at least ATTN_MIN_DWELL_S unless another is much more interesting); another
     face takes over only if clearly more interesting. A NEW face (there ATTN_NEW_MIN_S, away from the current
     one, not where a face was just lost) gets a short glance first; every few seconds another person who LOOKS
     at the robot gets one too (like looking round a group), the longest-unlooked-at first.
  1b. Listening: while the followed person (the partner, or someone looking at it) is speaking - hearing() - the
     mode is "listen": eyes stay on them (no breaks, glances or boredom); robot/head_life.py adds the nods.
  2. Boredom: someone who just sits there gets less interesting the longer they are watched (ATTN_BORED_S;
     slower while they look at the robot, never while talking with it) - then the robot looks away and around,
     with glances back, until they are fresh again or do something (look at it, speak).
  3. Eye contact is broken now and then (ATTN_BREAK_*): a short look aside, as people do - staring is creepy.
  4. A voice (hear()): the speaker's face if in view; else the person it is talking with (partner) or the one known
     face in view, if there; else where that person was last seen (voice recognition, robot/voices.py); else, if
     nobody in view is looking at it and no conversation is open, a look straight ahead at face height - and only
     its name makes it look round the whole room (3 stops, +-60 deg). Its name always turns it.
     No head (has_head=False, no battery): voices turn nothing and make no "turn" events.
  5. Lost them: keep looking where they were for a moment (a blink, a turned head), then look a little around
     that spot, then stay there idle (people come back where they were).
  6. Idle: rest looking ahead; every ATTN_IDLE_EVERY_S a slow look somewhere - often where people were seen - held
     a few seconds, then back. Slower and lower at night.
  7. Following a face far to the side for a while: ask the body to turn (body_turn), so the head can
     come back toward the middle.
  8. Bodies (robot/body.py, docs/vision.md "Bodies"): nobody's face to follow, but a person's body in view (legs up close,
     an arm at the edge, someone turned away): look where their face should be (BodyTrack.hint) - "seek" - for
     ATTN_SEEK_S, then give that body up (a chair the model takes for a person) until it moves. Head keypoints seen
     (surely a person: turned away, or above the face detector's reach) are followed longer (ATTN_SEEK_HEAD_S). Someone WAVING
     (wave()) is looked at like someone who spoke: their face if in view, else where their face should be.

    att = Attention()
    att.partner = conversation.partner()                   # who it is talking with (boredom stops for them)
    att.hear(t, person="p3", name=False)                    # someone spoke (person from the voice, or None)
    att.wave(t, track=7, person="p3", pan=20, tilt=30)      # someone waved (robot/body.py "waved" event)
    att.hearing = lambda: ears.pause_s                     # None = nobody speaking (else seconds of quiet)
    pan, tilt, mode = att.update(tracker.visible(t), t, bodies.seekable(t))   # follow / listen / glance / break /
                                                           # orient / scan / seek / hold / search / idle
    if att.body_turn is not None: ...                      # degrees the body could turn (+ = left)
"""
import math
import random
from datetime import datetime
from zoneinfo import ZoneInfo

from . import config


def key(tr):
    """Boredom is about a person, not a track (a track can end and a new one start on the same face)."""
    return tr.person if tr.person is not None else ("track", tr.id)


def who(tr):
    return 2.0 if tr.person is not None else (1.4 if tr.unknown else 1.0)


def engaged(tr):
    """x0.5 turned away .. x2.0 facing the robot."""
    return 0.5 + 1.5 * getattr(tr, "engagement", 0.0)


CUES = (("who", who), ("engaged", engaged))      # name, factor(track)


def cues(tr):
    """Each cue's factor, for logs: {"who": 2.0, "engaged": 1.7}."""
    return {name: f(tr) for name, f in CUES}


def interest(tr):
    """How interesting a face track is: its size times every cue."""
    v = tr.sighting.size_deg
    for _, f in CUES:
        v *= f(tr)
    return v


class Attention:
    def __init__(self, rng=None, hour=None, has_head=True, tz=config.TTS_TIMEZONE, hearing=None):
        self.rng = rng or random.Random()
        zone = ZoneInfo(tz)
        self._hour = hour or (lambda: datetime.now(zone).hour)     # local (Turkish) hour, whatever the Pi's clock zone
        self.has_head = has_head    # False: no battery - nothing to turn
        self.target = None          # track id followed
        self.glance = None          # (track id, until t) - a quick look at a newcomer
        self.last = (0.0, 0.0)      # where the followed face was last seen
        self.lost_t = None
        self.seen_t = 0.0
        self.hearing = hearing      # () -> None (nobody speaking) or seconds of quiet in their sentence
        self._listen_until = -math.inf
        self.idle = (0.0, config.ATTN_IDLE_HOME_TILT)
        self.next_idle_t = None     # (set on the first update: it starts resting ahead)
        self._idle_home = True      # idle: resting ahead (True) or on a look somewhere (False)
        self._just_bored = False    # just looked away from someone boring: not straight back at them
        self._idle_glance = False   # the glance under way is an idle glance back (its end is not a lost face)
        self.follow_t = 0.0         # when it started following the current target
        self._first = {}            # track id -> when it appeared (a newcomer glance waits ATTN_NEW_MIN_S)
        self._prev_pos = {}         # track id -> (pan, tilt) in the last update
        self._gone = []             # (pan, tilt, t) where tracks ended lately (a new track there = the same face)
        self.mode = "idle"
        self.side_since = None
        self.body_turn = None
        self._seen_ids = set()
        self.next_share_t = None    # when to glance at another person looking at the robot
        self.attended = {}          # track id -> last time it was followed or glanced at
        self.bored = {}             # person (or track) -> boredom, in seconds watched
        self.away = set()           # ...those it is bored of now (back when fresh again)
        self.partner = None         # the person it is talking with (set from robot/conversation.py)
        self.where = {}             # person -> (pan, tilt, t) where they were last seen
        self.orient = None          # (pan, tilt, until) looking toward a voice
        self.scan = None            # (start t, [(pan, tilt), ...]) looking round for a voice it could not place
        self.brk = None             # (pan offset, tilt offset, until) an eye-contact break
        self.next_break_t = None
        self._heard = []            # voices not handled yet: (t, person, name)
        self.events = []            # what it did about voices ({"kind": "turn", ...}), for the recorder / dashboard
        self._voice_t = {}          # person (None = unplaced) -> when it last turned to that voice
        self._spoke = {}            # person -> when their voice was last heard
        self._prev_t = None
        self._prev_looking = set()
        self._waves = []            # waves not handled yet: (t, face track id, person, pan, tilt)
        self._seeking = None        # the body whose face it is looking for
        self._given_up = []         # (pan, tilt, t) where it looked for a face on a body and found none

    # ---------- inputs ----------

    def hear(self, t, person=None, name=False):
        """Someone spoke at t: person = who the voice is (robot/voices.py) or None; name = it was called."""
        self._heard.append((t, person, name))

    def wave(self, t, track=None, person=None, pan=None, tilt=None):
        """Someone waved at t (robot/body.py): their face track if it has one, who they are, where their face is."""
        self._waves.append((t, track, person, pan, tilt))

    def fresh(self, tr):
        """1 = new / interesting again .. ATTN_BORED_MIN = watched long enough."""
        return max(config.ATTN_BORED_MIN, math.exp(-self.bored.get(key(tr), 0.0) / config.ATTN_BORED_S))

    def eff(self, tr):
        """interest x freshness x (just spoke: ATTN_SPOKE_GAIN)."""
        spoke = tr.person is not None and self._prev_t is not None and \
            self._prev_t - self._spoke.get(tr.person, -1e9) < config.ATTN_SPOKE_S
        return interest(tr) * self.fresh(tr) * (config.ATTN_SPOKE_GAIN if spoke else 1.0)

    # ---------- the decision ----------

    def update(self, tracks, t, bodies=None, shown=None):
        """tracks = the face tracks visible now, bodies = bodies to look for a face on (BodySense.seekable),
        shown = (pan, tilt, what) of something held up to it (robot/show.py ShowSense.target) or None.
        Returns (pan, tilt, mode)."""
        dt = 0.0 if self._prev_t is None else min(0.5, max(0.0, t - self._prev_t))
        self._prev_t = t
        tracks = [tr for tr in tracks if not getattr(tr, "picture", False)]     # a photo on the wall: never
        by_id = {tr.id: tr for tr in tracks}
        new = [tr for tr in tracks if tr.id not in self._seen_ids]
        self._seen_ids |= set(by_id)                # track ids are never reused
        newcomers = self._newcomers(tracks, new, by_id, t, watching=self.target in by_id)
        for tr in tracks:
            if tr.person is not None:
                self.where[tr.person] = (tr.pan, tr.tilt, t)
        self._wake_up(tracks, t)
        self._recover(tracks, dt)

        heard = self._voices(tracks, by_id, t)
        heard = self._waved(tracks, by_id, t) or heard
        if self.scan is not None and new:           # looking round for a voice: a face shows up - that is them
            self.scan = None
            heard = max(new, key=interest)
            self.bored[key(heard)] = 0.0
            self.away.discard(key(heard))
        if self.orient is not None:                 # turned toward a voice: a face there is the speaker
            near = [tr for tr in tracks if math.hypot(tr.pan - self.orient[0], tr.tilt - self.orient[1])
                    <= config.ATTN_ORIENT_NEAR_DEG]
            if near:
                self.orient, heard = None, max(near, key=interest)
                self.bored[key(heard)] = 0.0
                self.away.discard(key(heard))
        if heard is not None and heard.id != self.target:
            self.target, self.glance, self.brk = heard.id, None, None
        if heard is None and (self.orient is not None or self.scan is not None):
            self._side(None, t)
            out = self._toward_voice(t)
            if out is not None:
                return out

        # something held up to it (or asked "bu ne?"): look at the thing, not the face - below its name / a voice
        # (handled above), above everything else; the face it was following stays the target for afterwards
        if shown is not None:
            self._side(None, t)
            lo, hi = config.HEAD_TILT_SOFT
            return self._out(shown[0], max(lo, min(hi, shown[1])), "show")

        # who is worth looking at: not those it is bored of (unless fresh again)
        for tr in tracks:
            f, k = self.fresh(tr), key(tr)
            if f < config.ATTN_AWAY_BELOW and k != self.partner:
                self.away.add(k)
            elif f >= config.ATTN_BACK_ABOVE:
                self.away.discard(k)
        worth = [tr for tr in tracks if key(tr) not in self.away]

        cur = by_id.get(self.target)
        if cur is not None and key(cur) in self.away:
            cur = None                              # bored: look away (and around) - not a lost face to hold
            self.target, self.lost_t, self.mode = None, None, "idle"
            self._idle_home, self.next_idle_t, self._just_bored = True, t, True
        before = self.target
        # the followed face out of view a moment (the detector drops real faces for 0.8-3 s): hold on its spot,
        # do not jump to someone else and back - unless a face shows up right there (most likely them again)
        holding = self.target is not None and t - self.seen_t < config.ATTN_HOLD_S and self.mode in (
            "follow", "listen", "break", "hold")
        if cur is None and worth:                   # nobody followed (or they left): pick the best
            near = [tr for tr in worth if (holding or self.lost_t is not None) and
                    math.hypot(tr.pan - self.last[0], tr.tilt - self.last[1]) <= config.ATTN_REACQUIRE_DEG]
            if near or not holding:
                cur = max(near or worth, key=self.eff)
                self.target = cur.id
        elif cur is not None:
            best = max(worth, key=self.eff, default=cur)
            gain = self.eff(best) / max(1e-9, self.eff(cur))
            settled = t - self.follow_t >= config.ATTN_MIN_DWELL_S
            if best is not cur and gain > config.ATTN_SWITCH_GAIN and (settled or gain > config.ATTN_SWITCH_NOW_GAIN):
                cur, self.target = best, best.id
        listen = cur is not None and self._listening(cur, t)
        if listen:
            self.glance = None                      # eyes stay on whom it listens to
        elif cur is not None and self.target == before and newcomers and self.glance is None:
            other = max((tr for tr in newcomers if tr is not cur and math.hypot(
                tr.pan - cur.pan, tr.tilt - cur.tilt) > config.ATTN_GLANCE_MIN_DEG), key=interest, default=None)
            if other is not None:
                self.glance, self._idle_glance = (other.id, t + config.ATTN_GLANCE_S), False
        if self.target != before or self.next_share_t is None:
            self.follow_t = t
            self.next_share_t = t + self.rng.uniform(*config.ATTN_SHARE_EVERY_S)
            self.next_break_t, self.brk = None, None
        elif cur is not None and self.glance is None and t >= self.next_share_t and not listen:
            self._share(cur, tracks, t)

        if self.glance is not None:
            gid, until = self.glance
            g = by_id.get(gid)
            if g is None or t >= until:
                self.glance = None
            else:
                self._side(None, t)
                self.attended[gid] = t
                return self._out(g.pan, g.tilt, "glance")

        if cur is not None:
            self.attended[cur.id] = t
            self.last, self.lost_t, self.seen_t = (cur.pan, cur.tilt), None, t
            self._side(cur.pan, t)
            if listen:                              # they are talking to it: no boredom, no looking aside
                self.next_break_t, self.brk = None, None
                return self._out(cur.pan, cur.tilt, "listen")
            self._bore(cur, dt)
            b = self._eye_break(cur, t)
            if b is not None:
                return self._out(cur.pan + b[0], cur.tilt + b[1], "break")
            return self._out(cur.pan, cur.tilt, "follow")

        self._side(None, t)
        if self.has_head and bodies:
            out = self._seek(bodies, t, holding)
            if out is not None:
                return out
        else:
            self._seeking = None
        if not holding:
            self.target = None
        if self.lost_t is None and (self.mode in ("follow", "listen", "break")
                                    or (self.mode == "glance" and not self._idle_glance)):
            self.lost_t = self.seen_t               # counted from the last sighting, not from now
        if self.lost_t is not None:
            gone = t - self.lost_t
            if gone < config.ATTN_HOLD_S:
                return self._out(*self.last, "hold")
            if gone < config.ATTN_HOLD_S + config.ATTN_SEARCH_S:
                # a small, slow look to one side and the other around the last spot (where people reappear)
                phase = (gone - config.ATTN_HOLD_S) / config.ATTN_SEARCH_S
                swing = config.ATTN_SEARCH_DEG if phase < 0.5 else -config.ATTN_SEARCH_DEG
                return self._out(self.last[0] + swing, self.last[1], "search")
            self.lost_t = None                      # not back: stay looking there for a while (no jump)
            self.idle, self._idle_home = self.last, False
            self.next_idle_t = t + self._idle_wait(config.ATTN_IDLE_LOOK_S)
        if self.next_idle_t is None:
            self.next_idle_t = t + self._idle_wait(config.ATTN_IDLE_EVERY_S)
        if t >= self.next_idle_t:
            return self._next_idle(tracks, t)
        return self._out(*self.idle, "idle")

    def _idle_wait(self, span):
        return self.rng.uniform(*span) * (config.ATTN_NIGHT_SLOWER if self._night() else 1.0)

    def _next_idle(self, tracks, t):
        """Idle: rest ahead; now and then a slow look somewhere (often where people were seen), held, then back."""
        night = self._night()
        if not self._idle_home:                     # back to resting ahead
            self.idle, self._idle_home = (0.0, config.ATTN_NIGHT_HOME_TILT if night else config.ATTN_IDLE_HOME_TILT), True
            self.next_idle_t = t + self._idle_wait(config.ATTN_IDLE_EVERY_S)
            return self._out(*self.idle, "idle")
        self._idle_home = False
        self.next_idle_t = t + self._idle_wait(config.ATTN_IDLE_LOOK_S)
        bored_of = [tr for tr in tracks if key(tr) in self.away]
        if bored_of and not self._just_bored and self.rng.random() < config.ATTN_GLANCE_BACK_P:  # a glance back
            g = self.rng.choice(bored_of)
            self.glance, self._idle_glance = (g.id, t + config.ATTN_GLANCE_S), True
            self.attended[g.id] = t
            return self._out(g.pan, g.tilt, "glance")
        self._just_bored = False
        lim = config.ATTN_IDLE_PAN_MAX
        tilts = config.ATTN_NIGHT_TILT if night else config.ATTN_IDLE_TILT
        places = [(p, q) for p, q, seen in self.where.values()
                  if t - seen < config.ATTN_WHERE_KEEP_S and abs(p) <= lim and tilts[0] <= q <= tilts[1]]
        if places and self.rng.random() < config.ATTN_IDLE_PLACE_P:      # where someone was: are they back?
            self.idle = self.rng.choice(places)
        else:
            self.idle = (max(-lim, min(lim, self.rng.gauss(0, config.ATTN_IDLE_PAN_DEG))), self.rng.uniform(*tilts))
        return self._out(*self.idle, "idle")

    def _newcomers(self, tracks, new, by_id, t, watching=True):
        """Faces that really arrived while it watched someone: there for ATTN_NEW_MIN_S and not where a face was
        lost a moment ago (the tracker loses and restarts real faces - 6-13 new tracks a minute on 2026-11-01 -
        and each got a glance)."""
        for tr_id in set(self._prev_pos) - set(by_id):
            self._gone.append((*self._prev_pos[tr_id], t))
        self._gone = [g for g in self._gone if t - g[2] < config.ATTN_CHURN_S]
        self._prev_pos = {tr.id: (tr.pan, tr.tilt) for tr in tracks}
        for tr in new if watching else ():
            if not any(math.hypot(tr.pan - g[0], tr.tilt - g[1]) <= config.ATTN_CHURN_DEG for g in self._gone):
                self._first[tr.id] = t
        out = [tr for tr in tracks if tr.id in self._first and t - self._first[tr.id] >= config.ATTN_NEW_MIN_S]
        for tr in out:
            del self._first[tr.id]
        self._first = {k: v for k, v in self._first.items() if t - v < 5.0}
        return out

    def _listening(self, cur, t):
        """Is the followed person speaking to it? (hearing() says someone speaks, and cur is the partner or looks
        at the robot.) Held ATTN_LISTEN_HOLD_S over the gaps between their sentences."""
        if self.hearing is None or not self.has_head:
            return False
        try:
            quiet = self.hearing()
        except Exception:  # noqa: BLE001 - the ears are another thread's business
            quiet = None
        if quiet is not None:
            self._listen_until = t + config.ATTN_LISTEN_HOLD_S
        return t < self._listen_until and (key(cur) == self.partner or cur.looking)

    # ---------- helpers ----------

    def _toward_voice(self, t):
        """Looking where a voice came from (orient) or round the room for it (scan); None when that is over."""
        if self.orient is not None:                 # where that person was last seen
            pan, tilt, until = self.orient
            if t < until:
                return self._out(pan, tilt, "orient")
            # not there: search around that spot like for a face just lost
            self.orient, self.last, self.lost_t, self.mode = None, (pan, tilt), t - config.ATTN_HOLD_S, "orient"
        if self.scan is not None:                   # jump, stop, look: vision ignores a fast-moving head
            start, spots = self.scan
            i = int((t - start) / config.ATTN_SCAN_FIX_S)
            if i < len(spots):
                return self._out(*spots[i], "scan")
            self.scan = None
        return None

    def _scan_spots(self, t):
        """Where to stop and look for an unplaced voice (Selim, 2026-11-01): straight ahead and a little up, then
        far left and far right (random order) at that height. The wide camera sees ~+-50 deg around each stop,
        so 3 stops at +-60 cover ~+-110 deg - he stood 70 deg to the left and the old +-45 sweep missed him."""
        h = min(config.HEAD_TILT_SOFT[1], self._face_height(t) + config.ATTN_SCAN_UP_DEG)
        side = self.rng.choice((-1.0, 1.0)) * config.ATTN_SCAN_PAN_DEG
        return [(0.0, h), (side, h), (-side, h)]

    def _face_height(self, t):
        """Tilt where faces usually are here: the middle of where people were seen lately (people sit higher
        than the robot's head - Selim at +30..+39, 2026-11-01); ATTN_SCAN_TILT before anyone was seen."""
        tilts = sorted(w[1] for w in self.where.values() if t - w[2] < config.ATTN_WHERE_KEEP_S)
        return tilts[len(tilts) // 2] if tilts else config.ATTN_SCAN_TILT

    def _night(self):
        a, b = config.ATTN_NIGHT_HOURS
        h = self._hour()
        return h >= a or h < b

    def _bore(self, cur, dt):
        """Watching cur for dt more seconds: a little more bored of them (not of whom it is talking with)."""
        k = key(cur)
        if k == self.partner:
            return
        rate = config.ATTN_BORED_LOOKING_RATE if cur.looking else 1.0
        self.bored[k] = self.bored.get(k, 0.0) + rate * dt

    def _recover(self, tracks, dt):
        """Not watched: boredom fades (a person seen again later is interesting again)."""
        watched = next((key(tr) for tr in tracks if tr.id == self.target), None)
        for k in list(self.bored):
            if k != watched:
                self.bored[k] = max(0.0, self.bored[k] - config.ATTN_BORED_RECOVER * dt)
                if self.bored[k] == 0.0 and k not in self.away:
                    del self.bored[k]

    def _wake_up(self, tracks, t):
        """Doing something makes a person interesting again: starting to look at the robot."""
        looking = {tr.id for tr in tracks if tr.looking}
        for tr in tracks:
            if tr.id in looking and tr.id not in self._prev_looking:
                k = key(tr)
                self.bored[k] = self.bored.get(k, 0.0) * config.ATTN_WAKE_KEEP
        self._prev_looking = looking

    def _voices(self, tracks, by_id, t):
        """Turn to the voices heard since the last frame. Returns a track to follow now, or None."""
        heard, self._heard = self._heard, []
        follow = None
        for th, person, name in heard:
            if t - th > 5.0:
                continue                            # too old to turn to
            if person is not None:
                self._spoke[person] = th
                self.bored[person] = 0.0            # speaking makes them interesting
                self.away.discard(person)
            if not self.has_head:
                continue                            # no head to turn: nothing to do about where it came from
            cur = by_id.get(self.target)
            face = next((tr for tr in tracks if person is not None and tr.person == person), None)
            if face is None and person is None:     # unplaced: whom it talks with, or the one known face here
                face = next((tr for tr in tracks if self.partner is not None and tr.person == self.partner), None)
                known = [tr for tr in tracks if tr.person is not None]
                if face is None and len({tr.person for tr in known}) == 1:
                    face = max(known, key=interest)
            if face is not None:                    # in view: look at them
                follow = face
                self.events.append(dict(kind="turn", how="face", person=face.person, called=name))
                continue
            if cur is not None and cur.looking and not name:
                continue                            # it is facing someone who looks at it: most likely them
            last_turn = self._voice_t.get(person, -1e9)
            if not name and t - last_turn < config.ATTN_VOICE_AGAIN_S:
                continue                            # people talking on and on: not every sentence turns it
            w = self.where.get(person) if person is not None else None
            if w is not None and t - w[2] < config.ATTN_WHERE_KEEP_S:
                self.orient = (w[0], w[1], t + config.ATTN_ORIENT_S)
                self._voice_t[person] = t
                self.target = None
                self.events.append(dict(kind="turn", how="last_seen", person=person, called=name,
                                        pan=round(w[0]), tilt=round(w[1])))
            elif name and self.partner is None:
                # its name, nobody to place it with: look round the room (never during a conversation: the head
                # swinging away loses the face and drops the next sentence)
                self.scan = (t, self._scan_spots(t))
                self._voice_t[person] = t
                self.target = None
                self.events.append(dict(kind="turn", how="sweep", person=person, called=name))
            elif self.partner is None and not any(tr.looking for tr in tracks):
                # just a voice somewhere (11 of 12 sweeps on 2026-11-01 were not its name - TV, people talking
                # to each other): a calm look ahead at face height, no +-60 deg sweep; never mid-conversation
                self.orient = (0.0, self._face_height(t), t + config.ATTN_ORIENT_S)
                self._voice_t[person] = t
                self.target = None
                self.events.append(dict(kind="turn", how="ahead", person=person, called=name))
        return follow

    def _waved(self, tracks, by_id, t):
        """Turn to the people who waved since the last frame. Returns a face track to follow now, or None."""
        waves, self._waves = self._waves, []
        follow = None
        for tw, track, person, pan, tilt in waves:
            if t - tw > 3.0:
                continue
            if self.partner is not None and person != self.partner:
                continue                            # talking with someone: another's wave does not pull it away
            if person is not None:
                self._spoke[person] = tw            # as interesting as someone who just spoke
                self.bored[person] = 0.0
                self.away.discard(person)
            face = by_id.get(track) if track is not None else None
            if face is None and person is not None:
                face = next((tr for tr in tracks if tr.person == person), None)
            if face is not None:
                follow = face
                self.bored[key(face)] = 0.0
                self.away.discard(key(face))
                self.events.append(dict(kind="turn", how="wave", person=face.person, track=face.id))
            elif self.has_head and pan is not None:
                self.orient = (pan, self._face_height(t) if tilt is None else tilt, t + config.ATTN_ORIENT_S)
                self.target = None
                self.events.append(dict(kind="turn", how="wave", person=person, pan=round(pan), tilt=None if tilt
                                        is None else round(tilt)))
        return follow

    def _seek(self, bodies, t, holding):
        """Nobody's face to follow but a body in view: look where its face should be (pan, tilt, "seek"), or None.
        While a lost face is held, only a body there counts (they turned away / moved a little). A body looked for
        ATTN_SEEK_S without a face showing up is given up (until it moves - robot/body.py)."""
        self._given_up = [g for g in self._given_up if t - g[2] < config.ATTN_SEEK_FORGET_S]
        cands = []
        for b in bodies:
            if any(math.hypot(b.pan - g[0], b.tilt - g[1]) <= config.ATTN_SEEK_SPOT_DEG for g in self._given_up):
                continue                            # a new track on a thing it already gave up on
            if holding and math.hypot(b.hint[0] - self.last[0], (b.hint[1] if b.hint[1] is not None else
                                                                 self.last[1]) - self.last[1]) > config.ATTN_SEEK_NEAR_DEG:
                continue
            cands.append(b)
        if not cands:
            self._seeking = None
            return None
        b = next((x for x in cands if self._seeking is not None and x.id == self._seeking), None) or \
            max(cands, key=lambda x: x.height_deg)
        if b.seek_t is None:
            b.seek_t = t
            self.events.append(dict(kind="turn", how="body", body=b.id, hint=b.hint[2],
                                    pan=round(b.hint[0]), tilt=None if b.hint[1] is None else round(b.hint[1])))
        patience = config.ATTN_SEEK_HEAD_S if b.hint[2] == "head" else config.ATTN_SEEK_S
        if t - b.seek_t >= patience:               # no face there: a thing, or someone hidden - stop looking
            b.given_up, b.given_up_at = True, (b.pan, b.tilt)
            self._given_up.append((b.pan, b.tilt, t))
            self._seeking = None
            return None
        self._seeking = b.id
        pan = b.hint[0]
        tilt = b.hint[1] if b.hint[1] is not None else self._face_height(t)
        lo, hi = config.HEAD_TILT_SOFT
        self.lost_t = None                          # the person is here, just not their face
        self.last = (pan, max(lo, min(hi, tilt)))
        return self._out(pan, max(lo, min(hi, tilt)), "seek")

    def _eye_break(self, cur, t):
        """Mutual gaze: every few seconds a short look aside. (pan, tilt) offset now, or None."""
        if not cur.looking:
            self.next_break_t, self.brk = None, None
            return None
        if self.brk is not None:
            if t < self.brk[2]:
                return self.brk[:2]
            self.brk = None
        if self.next_break_t is None:
            self.next_break_t = t + self.rng.uniform(*config.ATTN_BREAK_EVERY_S)
        elif t >= self.next_break_t:
            side = self.rng.choice((-1.0, 1.0))
            self.brk = (side * self.rng.uniform(*config.ATTN_BREAK_DEG), -self.rng.uniform(2.0, 6.0),
                        t + self.rng.uniform(*config.ATTN_BREAK_S))
            self.next_break_t = None
            return self.brk[:2]
        return None

    def _share(self, cur, tracks, t):
        """Looking round a group: a glance at someone else who looks at the robot, the longest unlooked-at."""
        self.next_share_t = t + self.rng.uniform(*config.ATTN_SHARE_EVERY_S)
        others = [tr for tr in tracks if tr is not cur and getattr(tr, "looking", False)]
        if others:
            g = min(others, key=lambda tr: (self.attended.get(tr.id, -1e9), -interest(tr)))
            self.glance, self._idle_glance = (g.id, t + config.ATTN_SHARE_S), False
        if len(self.attended) > 200:                # tracks long gone
            self.attended = dict(sorted(self.attended.items(), key=lambda kv: kv[1])[-50:])

    def _out(self, pan, tilt, mode):
        self.mode = mode
        return pan, tilt, mode

    def _side(self, pan, t):
        """Following a face far to one side for a while -> suggest a body turn toward it."""
        if pan is None or abs(pan) < config.ATTN_BODY_TURN_DEG:
            self.side_since, self.body_turn = None, None
            return
        if self.side_since is None:
            self.side_since = t
        self.body_turn = pan if t - self.side_since >= config.ATTN_BODY_TURN_S else None
