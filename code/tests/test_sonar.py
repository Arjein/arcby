"""robot/sonar.py: the HC-SR04 echo timed from kernel edge timestamps (fake lgpio) - a busy program must not
change the distance."""
import threading
import time
import unittest

from robot import config
from robot.sonar import Sonar, pulse_cm


class FakeLgpio:
    """Edges arrive on another thread (like lgpio's notify thread) with exact kernel times, while the caller may be
    slow. echo_s None = the sensor never answers; 'long' = no echo in range."""
    BOTH_EDGES, SET_PULL_DOWN = 3, 32

    class error(Exception):
        pass

    def __init__(self, echo_s=0.01, deliver_late_s=0.0, busy=False):
        self.echo_s, self.late, self.busy = echo_s, deliver_late_s, busy
        self.claimed, self.cb = set(), None

    def gpiochip_open(self, n):
        return 7

    def gpiochip_close(self, h):
        pass

    def gpio_claim_output(self, h, pin, level):
        if self.busy:
            raise self.error("GPIO busy")
        self.claimed.add(pin)

    def gpio_claim_alert(self, h, pin, edges, flags):
        self.claimed.add(pin)

    def gpio_free(self, h, pin):
        self.claimed.discard(pin)

    def callback(self, h, pin, edges, func):
        fake = self

        class Cb:
            def cancel(self):
                fake.cb = None
        self.cb = func
        return Cb()

    def gpio_write(self, h, pin, level):
        if pin != config.SONAR_TRIG or level != 0 or self.echo_s is None:
            return
        rise = time.monotonic_ns() + 400_000                  # echo starts ~0.4 ms after the trigger
        fall = rise + int((0.038 if self.echo_s == "long" else self.echo_s) * 1e9)
        func = self.cb

        def deliver():                                        # the program is busy: edges reach us late,
            time.sleep(self.late)                             # but with their true kernel times
            func(7, config.SONAR_ECHO, 1, rise)
            func(7, config.SONAR_ECHO, 0, fall)
        threading.Thread(target=deliver, daemon=True).start()


class SonarTest(unittest.TestCase):
    def test_distance_from_kernel_times_even_when_late(self):
        for late in (0.0, 0.02):                              # 20 ms late delivery = 343 cm with the old loop
            lg = FakeLgpio(echo_s=0.01, deliver_late_s=late)
            self.assertAlmostEqual(Sonar(lg).ping(), 171.5, delta=0.2)
            self.assertEqual(lg.claimed, set())               # pins freed after every ping (shared sensor)

    def test_no_answer_nothing_in_range_and_busy(self):
        self.assertEqual(Sonar(FakeLgpio(echo_s=None)).ping(), "noanswer")
        self.assertEqual(Sonar(FakeLgpio(echo_s="long")).ping(), "nothing")
        self.assertEqual(Sonar(FakeLgpio(busy=True)).ping(), "busy")

    def test_pulse_maths(self):
        self.assertEqual(pulse_cm([]), "noanswer")
        self.assertEqual(pulse_cm([(1, 1000)]), "nothing")
        self.assertEqual(pulse_cm([(0, 500), (1, 1000), (0, 1000 + 5_830_000)]), 100.0)   # a stale fall ignored


if __name__ == "__main__":
    unittest.main()
