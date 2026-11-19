"""
Gaze (layer A): is a face turned toward the robot? From the 5 landmarks every detector gives (eyes, nose, mouth
corners) - no model, ~free, works as far as faces are found (3-4 m). It measures where the HEAD points, not the
eyes: someone facing the robot but reading their phone still counts (an eye-contact model for close range can
come on top, see docs/vision.md).

    yaw, pitch = head_turn(kps)      # 0, 0 = straight at the camera
    s = facing(kps)                  # 0 (turned away) .. 1 (facing the robot)

The tracker smooths facing() into Track.engagement (robot/tracker.py) and decides Track.looking.
"""
import numpy as np

from . import config

# on the ArcFace template the nose sits at 0.49 of the way from the eye line down to the mouth line
PITCH_STRAIGHT = 0.494


def head_turn(kps):
    """(yaw, pitch) proxies from 5 landmarks: yaw = nose offset sideways from the eyes' middle in eye distances
    (~0.2 = 20 deg), pitch = how far the nose sits off its straight-on height between eyes and mouth
    (+ = looking down / chin in, - = up)."""
    kps = np.asarray(kps, dtype=np.float64)
    eye_l, eye_r, nose, mouth_l, mouth_r = kps
    eye_mid, mouth_mid = (eye_l + eye_r) / 2, (mouth_l + mouth_r) / 2
    eye_d = float(np.hypot(*(eye_r - eye_l)))
    span = float(mouth_mid[1] - eye_mid[1])
    if eye_d < 1e-6 or span <= 1e-6:
        return 9.9, 9.9                       # degenerate (a profile, upside down): not facing
    yaw = abs(float(nose[0] - eye_mid[0])) / eye_d
    pitch = float(nose[1] - eye_mid[1]) / span - PITCH_STRAIGHT
    return yaw, pitch


def _ramp(x, full, zero):
    """1 up to full, falling linearly to 0 at zero."""
    return 1.0 if x <= full else max(0.0, (zero - x) / (zero - full))


def facing(kps, max_yaw=config.GAZE_MAX_YAW, max_pitch=config.GAZE_MAX_PITCH,
           full_yaw=config.GAZE_FULL_YAW, full_pitch=config.GAZE_FULL_PITCH):
    """0 .. 1: 1 = facing the robot (within full_yaw / full_pitch: people look at the robot, not into the lens, and
    sit off its axis), falling to 0 at max_yaw / max_pitch."""
    yaw, pitch = head_turn(kps)
    return _ramp(yaw, min(full_yaw, max_yaw), max_yaw) * _ramp(abs(pitch), min(full_pitch, max_pitch), max_pitch)
