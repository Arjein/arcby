"""
Shared state: the latest value of every reading, with the time it was taken.

Each module publishes only its own keys (e.g. "imu"); everyone may read.
A reading older than the caller's max_age counts as missing (None), which is
how safety.py will notice a dead sensor.
"""
import threading
import time


class State:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._data = {}   # key -> (timestamp, value)

    def publish(self, key, value, t=None):
        with self._lock:
            self._data[key] = (self._clock() if t is None else t, value)

    def get(self, key, max_age=None):
        """Latest value, or None if never published or older than max_age seconds."""
        with self._lock:
            item = self._data.get(key)
        if item is None:
            return None
        t, value = item
        if max_age is not None and self._clock() - t > max_age:
            return None
        return value

    def age(self, key):
        """Seconds since key was last published, or None if never."""
        with self._lock:
            item = self._data.get(key)
        return None if item is None else self._clock() - item[0]
