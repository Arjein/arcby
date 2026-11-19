"""
Hand signs from the pose model's 17 body points (robot/body.py) - shoulders, elbows, wrists, no fingers - so the signs
are read from the forearm (docs/navigation.md part 3, docs/vision.md "Hand signs"):

  gel gel (Beckon)    "come here": the forearm up in front of the body, the wrist swinging toward the shoulder and
                      back (up-down in the picture, along the forearm) GESTURE_BECKON_TURNS times within
                      GESTURE_BECKON_WINDOW_S, over GESTURE_BECKON_AMP shoulder widths, the elbow staying put. The
                      sideways version of the same swing is a wave (body.WaveDetector): a swing that is not clearly
                      up-down (GESTURE_BECKON_RATIO) is not a beckon.
  dur (StopSign)      "stop": a hand raised to the shoulder or above, the person facing the robot (both shoulders
                      seen), held still GESTURE_STOP_HOLD_S - a wave swings, a stop sign does not.

All in shoulder widths of the person's own body, relative to their own shoulder / elbow: walking, the head turning or
the robot driving does not look like a sign. A false "gel gel" starts a drive nobody asked for, so it is strict; a
false stop sign only stops a drive, so it may be lenient. The thresholds are a first guess until Selim's own signs are
recorded by the robot (hwtest/signs.py replays recordings through these detectors).

    beckon, stop = Beckon(), StopSign()
    beckon.update(body, t)      # True while this body beckons
    stop.update(body, t)        # True while this body shows a stop sign
"""
from collections import deque

from . import body as B             # the keypoints and helpers (body.py imports this module: resolved at call time)
from . import config


def _spread(values):
    return max(values) - min(values) if values else 0.0


class Beckon:
    def __init__(self):
        self.hist = (deque(), deque())          # per arm: (t, sideways, up, elbow x, elbow y), shoulder widths

    def update(self, body, t):
        scale = B.WaveDetector._scale(body)
        beckons = False
        for (_, elbow, wrist), hist in zip(B.ARMS, self.hist):
            while hist and t - hist[0][0] > config.GESTURE_BECKON_WINDOW_S:
                hist.popleft()
            if scale is None or not (body.has(elbow) and body.has(wrist)):
                continue
            ex, ey = (float(v) for v in body.kps[elbow, :2])
            wx, wy = (float(v) for v in body.kps[wrist, :2])
            if wy >= ey + 0.2 * scale:          # the hand clearly down: the arm is down
                hist.clear()
                continue
            if not B.forearm_up(body, elbow, wrist):
                continue                        # a forearm swung toward the camera reads short: skip, keep the rest
            hist.append((t, (wx - ex) / scale, (ey - wy) / scale, ex / scale, ey / scale))
            beckons = beckons or self.beckons(hist)
        return beckons

    @staticmethod
    def beckons(hist):
        if len(hist) < config.GESTURE_BECKON_TURNS + 2:
            return False
        side, up = [h[1] for h in hist], [h[2] for h in hist]
        if _spread(up) < config.GESTURE_BECKON_AMP:
            return False
        if _spread(side) * config.GESTURE_BECKON_RATIO > _spread(up):
            return False                        # as much sideways as up-down: waving (or reaching about)
        if max(_spread([h[3] for h in hist]), _spread([h[4] for h in hist])) > config.GESTURE_BECKON_ELBOW:
            return False                        # the whole arm moves (walking, reaching, lifting): not the forearm alone
        return B.WaveDetector._swings(up, config.GESTURE_BECKON_TURNS, config.GESTURE_BECKON_AMP)


class StopSign:
    def __init__(self):
        self.hist = (deque(), deque())          # per arm, while the hand is up: (t, wrist x, wrist y) from the shoulder

    def update(self, body, t):
        both = body.has(B.L_SHOULDER) and body.has(B.R_SHOULDER)
        scale = B.WaveDetector._scale(body) if both else None
        shown = False
        for (shoulder, elbow, wrist), hist in zip(B.ARMS, self.hist):
            if scale is None or not (body.has(shoulder) and body.has(wrist)):
                hist.clear()
                continue
            sx, sy = (float(v) for v in body.kps[shoulder, :2])
            wx, wy = (float(v) for v in body.kps[wrist, :2])
            raised = wy <= sy + config.GESTURE_STOP_LEVEL * scale and \
                (not body.has(elbow) or float(body.kps[elbow, 1]) > wy)     # the hand above its own elbow
            if not raised:
                hist.clear()
                continue
            hist.append((t, (wx - sx) / scale, (wy - sy) / scale))
            while hist and t - hist[0][0] > config.GESTURE_STOP_HOLD_S + 1.0:
                hist.popleft()
            recent = [h for h in hist if t - h[0] <= config.GESTURE_STOP_HOLD_S]
            held = t - hist[0][0] >= config.GESTURE_STOP_HOLD_S and len(recent) >= 3
            still = max(_spread([h[1] for h in recent]), _spread([h[2] for h in recent])) <= config.GESTURE_STOP_STILL
            shown = shown or (held and still)
        return shown
