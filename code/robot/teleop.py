"""
Keyboard driving logic (no terminal code here, so it can be tested).

Hold-to-drive: a drive key gives a "lease" of KEY_LEASE_S; the terminal's
auto-repeat renews it while the key is held. Let go -> the lease runs out ->
stop. No keys at all (SSH dropped) -> stop as well.

  w forward   s backward   a spin left   d spin right
  q forward-left curve     e forward-right curve
  space stop   + / - speed level   r reset safety   x quit
"""
from . import config

DRIVE_KEYS = {
    "w": ("forward", lambda s, t: (s, s)),
    "s": ("backward", lambda s, t: (-s, -s)),
    "a": ("spin left", lambda s, t: (-t, t)),
    "d": ("spin right", lambda s, t: (t, -t)),
    "q": ("curve left", lambda s, t: (s * 0.5, s)),
    "e": ("curve right", lambda s, t: (s, s * 0.5)),
}


class Teleop:
    def __init__(self, speed=config.KEY_SPEED, turn=config.KEY_TURN, lease_s=config.KEY_LEASE_S):
        self.speed = speed
        self.turn = turn
        self.lease_s = lease_s
        self._key = None
        self._until = 0.0
        self.want_reset = False
        self.want_quit = False

    def press(self, key, now):
        """Handle one key. Returns a short text for the status line, or None."""
        key = key.lower()
        if key in DRIVE_KEYS:
            self._key, self._until = key, now + self.lease_s
        elif key == " ":
            self._key, self._until = None, 0.0
        elif key in "+=":
            self.speed = round(min(1.0, self.speed + 0.1), 1)
            return f"speed {self.speed:.1f}"
        elif key in "-_":
            self.speed = round(max(0.1, self.speed - 0.1), 1)
            return f"speed {self.speed:.1f}"
        elif key == "r":
            self.want_reset = True
            return "safety reset"
        elif key in ("x", "\x03", "\x1b"):
            self.want_quit = True
        return None

    def command(self, now):
        """(left, right, name) to send right now."""
        if self._key is None or now > self._until:
            self._key = None
            return 0.0, 0.0, "stop"
        name, fn = DRIVE_KEYS[self._key]
        left, right = fn(self.speed, self.turn)
        return left, right, name


def car_velocity(throttle, steer, speed, turn_dps):
    """Car-style keys -> (speed -1..1, turn rate deg/s, + = left) for motors.request_velocity.

    throttle  -1..1   + = gas (W), - = reverse (S)
    steer     -1..1   + = right (D), - = left (A)
    speed     0..1    speed level (slider)
    turn_dps          turn rate for a full steer (slider)

    Standing still, A / D turn on the spot at that rate. Reversing steers like a
    car: back + right swings the nose to the LEFT.
    """
    throttle = max(-1.0, min(1.0, throttle))
    steer = max(-1.0, min(1.0, steer))
    turn = -steer * turn_dps                  # steer right = negative yaw rate
    if throttle < 0:
        turn = -turn
    return throttle * speed + 0.0, turn + 0.0     # + 0.0 turns -0.0 into a plain 0.0
