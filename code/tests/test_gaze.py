"""robot/gaze.py + Track gaze: facing the robot from 5 landmarks, smoothed, looking with hysteresis."""
import unittest

import numpy as np

from robot import config, gaze
from robot.faces import ARCFACE_KPS
from robot.tracker import Sighting, Track


def kps(yaw=0.0, pitch=0.0):
    """Template landmarks with the nose pushed sideways (yaw, eye distances) / down (pitch, eye-mouth spans)."""
    k = ARCFACE_KPS.copy()
    eye_d = k[1, 0] - k[0, 0]
    span = (k[3, 1] + k[4, 1]) / 2 - (k[0, 1] + k[1, 1]) / 2
    k[2, 0] += yaw * eye_d
    k[2, 1] += pitch * span
    return k


class FacingTest(unittest.TestCase):
    def test_straight_face_faces_the_robot(self):
        yaw, pitch = gaze.head_turn(kps())
        self.assertLess(abs(yaw), 0.01)
        self.assertLess(abs(pitch), 0.01)
        self.assertGreater(gaze.facing(kps()), 0.95)

    def test_turned_away_does_not(self):
        mid = (config.GAZE_FULL_YAW + config.GAZE_MAX_YAW) / 2
        self.assertAlmostEqual(gaze.facing(kps(yaw=mid)), 0.5, delta=0.03)   # template nose ~0.003 off
        self.assertEqual(gaze.facing(kps(yaw=0.2)), 1.0)     # Selim looking at the robot from 15 deg off its axis
        self.assertEqual(gaze.facing(kps(yaw=0.6)), 0.0)                 # side view
        self.assertEqual(gaze.facing(kps(pitch=0.3)), 0.0)               # head down (the phone)
        self.assertEqual(gaze.facing(kps(pitch=-0.3)), 0.0)              # looking over the robot
        self.assertEqual(gaze.facing(np.zeros((5, 2))), 0.0)             # degenerate


class TrackGazeTest(unittest.TestCase):
    def test_a_glance_does_not_count_a_steady_look_does(self):
        tr = Track(1, Sighting(None, 0, 0, 5), 0.0, 0.0)
        t = 0.0
        for _ in range(3):                                              # 0.2 s facing the robot
            t += 0.066
            tr.update_gaze(kps(), t)
        self.assertFalse(tr.looking)
        for _ in range(30):                                             # 2 s
            t += 0.066
            tr.update_gaze(kps(), t)
        self.assertTrue(tr.looking)
        self.assertGreater(tr.looked_for(t), 0.5)
        t_on = tr.looking_since
        for _ in range(2):                                              # a quick look away: still looking
            t += 0.066
            tr.update_gaze(kps(yaw=0.6), t)
        self.assertEqual(tr.looking_since, t_on)
        for _ in range(30):                                             # turned away for 2 s
            t += 0.066
            tr.update_gaze(kps(yaw=0.6), t)
        self.assertFalse(tr.looking)
        self.assertEqual(tr.looked_for(t), 0.0)

    def test_big_stream_landmarks_win_over_the_small_stream(self):
        tr = Track(1, Sighting(None, 0, 0, 5), 0.0, 0.0)
        t = 0.0
        for i in range(60):                                             # 4 s: big crop says "phone"...
            t += 0.066
            if i % 4 == 0:
                tr.update_gaze(kps(pitch=0.3), t, big=True)
            tr.update_gaze(kps(), t)                                    # ...noisy small stream says "facing"
        self.assertLess(tr.engagement, 0.1)
        self.assertFalse(tr.looking)
        for _ in range(40):                                             # no big crops for 2.6 s: small counts
            t += 0.066
            tr.update_gaze(kps(), t)
        self.assertTrue(tr.looking)


if __name__ == "__main__":
    unittest.main()
