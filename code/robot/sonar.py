"""
HC-SR04 distance sensor (front, ~7 cm above the floor), on 3.3 V with ECHO direct.

The pins are claimed only for each ping and freed again (like the dashboard),
so two programs can share the sensor: whoever pings second gets "busy".

The echo is timed from the kernel's own edge timestamps (lgpio alerts), not by watching the pin in a Python loop:
in a busy program (alive.py: vision, ears, speaker...) a loop can lose the GIL for 5 ms at the wrong moment, and
5 ms = 86 cm. If alerts are not available, the old loop is the fallback.

    sonar = Sonar()
    sonar.ping()        # cm (float) or 'nothing' / 'noanswer' / 'busy' / 'error: ...'
    sonar.measure()     # median of a few pings in cm, or None
"""
import statistics
import threading
import time

from . import config

SOUND_M_S = 343.0


MAX_ECHO_S = 0.030                  # a longer pulse = the sensor's own "nothing in range" (~38 ms) - over ~5 m


def pulse_cm(edges):
    """Distance from the echo pin's edges [(level 1/0, kernel time ns)]: the first rising edge and the falling edge
    after it. 'noanswer' = the pin never went high, 'nothing' = no fall / too long a pulse (no echo in range)."""
    rise = next((t for lv, t in edges if lv == 1), None)
    if rise is None:
        return "noanswer"
    fall = next((t for lv, t in edges if lv == 0 and t > rise), None)
    if fall is None:
        return "nothing"
    s = (fall - rise) / 1e9
    if s > MAX_ECHO_S:
        return "nothing"
    return round(s * SOUND_M_S / 2 * 100, 1)


class Sonar:
    ECHO_WAIT_S = 0.07

    def __init__(self, lgpio_module=None):
        if lgpio_module is None:
            import lgpio as lgpio_module
        self.lgpio = lgpio_module
        self.chip = lgpio_module.gpiochip_open(0)
        self.alerts = hasattr(lgpio_module, "callback") and hasattr(lgpio_module, "gpio_claim_alert")

    def ping(self):
        """cm (float), or 'nothing' (no echo in range) / 'noanswer' (sensor silent) / 'busy' / 'error: ...'."""
        lg, h = self.lgpio, self.chip
        try:
            lg.gpio_claim_output(h, config.SONAR_TRIG, 0)
        except lg.error as e:
            return "busy" if "busy" in str(e).lower() else f"error: {e}"
        try:
            if self.alerts:
                try:
                    lg.gpio_claim_alert(h, config.SONAR_ECHO, lg.BOTH_EDGES, lg.SET_PULL_DOWN)
                except lg.error as e:
                    if "busy" in str(e).lower():
                        return "busy"
                    self.alerts = False             # no alerts here: time it the old way from now on
            if not self.alerts:
                try:
                    lg.gpio_claim_input(h, config.SONAR_ECHO, lg.SET_PULL_DOWN)
                except lg.error as e:
                    return "busy" if "busy" in str(e).lower() else f"error: {e}"
                return self._ping_polled()
            return self._ping_alerts()
        finally:
            lg.gpio_free(h, config.SONAR_TRIG)
            try:
                lg.gpio_free(h, config.SONAR_ECHO)
            except lg.error:
                pass

    def _trigger(self):
        lg, h = self.lgpio, self.chip
        lg.gpio_write(h, config.SONAR_TRIG, 1)
        time.sleep(0.00001)                         # >= 10 us trigger pulse
        lg.gpio_write(h, config.SONAR_TRIG, 0)

    def _ping_alerts(self):
        """Echo pulse length from kernel timestamps (ns) of its rising and falling edge."""
        lg, h = self.lgpio, self.chip
        edges = []                                  # (level, kernel time ns), filled by lgpio's own thread
        got = threading.Event()

        def on_edge(chip, gpio, level, tick):
            edges.append((level, tick))
            if level == 0 and any(lv == 1 for lv, _ in edges):
                got.set()
        cb = lg.callback(h, config.SONAR_ECHO, lg.BOTH_EDGES, on_edge)
        try:
            self._trigger()
            # the echo starts within ~0.5 ms and lasts at most ~38 ms (no echo); wait a little longer
            got.wait(self.ECHO_WAIT_S)
        finally:
            cb.cancel()
        return pulse_cm(list(edges))

    def _ping_polled(self):
        """Fallback without alerts: watch the pin (inaccurate when the program is busy)."""
        lg, h = self.lgpio, self.chip
        self._trigger()
        t0 = time.perf_counter()
        while not lg.gpio_read(h, config.SONAR_ECHO):
            if time.perf_counter() - t0 > 0.03:
                return "noanswer"
        start = time.perf_counter()
        while lg.gpio_read(h, config.SONAR_ECHO):
            if time.perf_counter() - start > 0.04:
                return "nothing"                    # no echo within ~6.8 m
        return round((time.perf_counter() - start) * SOUND_M_S / 2 * 100, 1)

    def measure(self, n=3):
        """Median of n pings (>= 60 ms apart) in cm; None when nothing usable came back."""
        got = []
        for i in range(n):
            if i:
                time.sleep(0.06)
            r = self.ping()
            if isinstance(r, float):
                got.append(r)
        return statistics.median(got) if got else None

    def close(self):
        self.lgpio.gpiochip_close(self.chip)
