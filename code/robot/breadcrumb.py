"""
Power breadcrumb: what the robot was doing in its last second, kept through a power cut.

On 2026-11-08 the Pi lost power right after a spin - no under-voltage warning, no black box, and nav.log ended in
garbage (the SD card loses the page cache on a cut). So once a second the programs that run the robot write one line
into a small ring file and fsync it:

    {"seq": 812, "wall": 1762613002.4, "boot": "3f2a...", "prog": "nav.py", "throttled": "0x0", "core_v": 0.86,
     "pwm": [0.6, -0.6], "doing": "spinning", "state": "explore/TURN", "cpu_c": 58.1}

The file has BREADCRUMB_SLOTS fixed-size slots (it never grows; line seq goes into slot seq % SLOTS). A clean stop
writes a last line with "exit": true. The next start reads the newest line first:

    crumb = Breadcrumb("alive.py", pwm=lambda: motors.applied, state=lambda: "follow")
    prev = crumb.start(stop_event)          # the previous run's last line (or None) - before this run's first write
    log.info(describe(prev))                # "last seen 15:43:22 (nav.py, other boot, NO clean stop), spinning, ..."
    ...
    crumb.close()                           # the "exit" line

Never stops the robot: a write that fails is logged once and the breadcrumb goes quiet.
"""
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def _vcgencmd(*args):
    try:
        out = subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=1).stdout
        return out.strip().split("=", 1)[1]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def pi_power():
    """(throttled hex string, core volts) from vcgencmd - (None, None) when it is not there (not a Pi)."""
    thr = _vcgencmd("get_throttled")
    v = _vcgencmd("measure_volts", "core")
    try:
        volts = float(v.rstrip("V")) if v else None
    except ValueError:
        volts = None
    return thr, volts


def doing(pwm):
    """Applied PWM (left, right) -> a word: standing / forward / backward / spinning / turning."""
    if pwm is None:
        return None
    l, r = pwm
    if abs(l) < 0.01 and abs(r) < 0.01:
        return "standing"
    if l * r < 0:
        return "spinning"
    if abs(l - r) > 0.1:
        return "turning"
    return "forward" if l + r > 0 else "backward"


def read_last(path=None, slot=None):
    """The newest line in the ring file (dict) or None (no file / nothing readable)."""
    path = Path(path or config.BREADCRUMB_FILE).expanduser()
    slot = slot or config.BREADCRUMB_SLOT
    try:
        data = path.read_bytes()
    except OSError:
        return None
    best = None
    for i in range(0, len(data), slot):
        try:
            rec = json.loads(data[i:i + slot].decode("utf-8", "replace").strip() or "null")
        except ValueError:
            continue                # a slot half written when the power went
        if isinstance(rec, dict) and isinstance(rec.get("seq"), int) and (best is None or rec["seq"] > best["seq"]):
            best = rec
    return best


def describe(rec, now_boot=None):
    """One plain sentence about the previous run's last line."""
    if rec is None:
        return "breadcrumb: no previous run recorded"
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec.get("wall", 0)))
    same = now_boot is not None and rec.get("boot") == now_boot[:8]       # lines keep the boot id's first 8
    if rec.get("exit"):
        how = "clean stop"
    elif same:
        how = "no clean stop (killed or crashed)"
    else:
        how = "NO clean stop before a reboot - power cut?"
    parts = [f"last seen {when} ({rec.get('prog', '?')}, {'same' if same else 'other'} boot, {how})"]
    if rec.get("doing"):
        parts.append(rec["doing"])
    if rec.get("pwm") is not None:
        parts.append("PWM %.2f/%.2f" % tuple(rec["pwm"]))
    if rec.get("state"):
        parts.append(f"state {rec['state']}")
    if rec.get("throttled") is not None:
        parts.append(f"throttled {rec['throttled']}")
    if rec.get("core_v") is not None:
        parts.append(f"core {rec['core_v']:.2f} V")
    if rec.get("cpu_c") is not None:
        parts.append(f"{rec['cpu_c']:.0f} C")
    return "breadcrumb: " + ", ".join(parts)


class Breadcrumb:
    def __init__(self, prog, pwm=None, state=None, path=None, power=pi_power, clock=time.time,
                 slots=None, slot=None, fsync=True):
        """pwm: () -> (left, right) applied PWM or None; state: () -> short text (what the program is doing)."""
        self.prog = prog
        self.pwm, self.state = pwm, state
        self.path = Path(path or config.BREADCRUMB_FILE).expanduser()
        self.power, self.clock = power, clock
        self.slots = slots or config.BREADCRUMB_SLOTS
        self.slot = slot or config.BREADCRUMB_SLOT
        self.fsync = fsync
        self.boot = boot_id()
        self.previous = None
        self.seq = 0
        self.fd = None
        self.failed = None
        self.thread = None
        self._closing = threading.Event()
        self._lock = threading.Lock()

    def open(self):
        """Read the previous run's last line, then open the file for this run. Returns the previous line."""
        self.previous = read_last(self.path, self.slot)
        self.seq = (self.previous or {}).get("seq", -1) + 1
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            size = self.slots * self.slot
            if os.fstat(self.fd).st_size != size:
                os.ftruncate(self.fd, size)         # a new file, or BREADCRUMB_SLOTS changed
        except OSError as e:
            self._fail(e)
        return self.previous

    def _fail(self, e):
        if self.failed is None:
            log.warning("breadcrumb: off (%s)", e)
        self.failed = str(e)
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = None

    def sample(self, exit_=False):
        rec = {"seq": self.seq, "wall": round(self.clock(), 2), "boot": (self.boot or "")[:8], "prog": self.prog}
        thr, volts = self.power() if self.power else (None, None)
        rec["throttled"], rec["core_v"] = thr, volts
        pwm = None
        if self.pwm is not None:
            try:
                p = self.pwm()
                pwm = None if p is None else [round(float(p[0]), 2), round(float(p[1]), 2)]
            except Exception:  # noqa: BLE001 - a breadcrumb never stops the robot
                pwm = None
        rec["pwm"], rec["doing"] = pwm, doing(pwm)
        if self.state is not None:
            try:
                st = self.state()
                rec["state"] = None if st is None else str(st)[:40]
            except Exception:  # noqa: BLE001
                rec["state"] = None
        try:
            rec["cpu_c"] = round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
        except (OSError, ValueError):
            pass
        if exit_:
            rec["exit"] = True
        return rec

    def write(self, exit_=False):
        with self._lock:
            if self.fd is None:
                return None
            rec = self.sample(exit_)
            line = json.dumps(rec, separators=(",", ":")).encode()
            if len(line) > self.slot - 1:           # never spill into the next slot: drop the extras
                for k in ("cpu_c", "state", "core_v"):
                    rec.pop(k, None)
                line = json.dumps(rec, separators=(",", ":")).encode()[:self.slot - 1]
            try:
                os.pwrite(self.fd, line.ljust(self.slot - 1) + b"\n", (self.seq % self.slots) * self.slot)
                if self.fsync:
                    os.fsync(self.fd)
            except OSError as e:
                self._fail(e)
                return None
            self.seq += 1
            return rec

    def start(self, stop):
        """open() + a thread writing every BREADCRUMB_S until stop is set. Returns the previous run's last line."""
        prev = self.open()
        self.thread = threading.Thread(target=self._run, args=(stop,), name="breadcrumb", daemon=True)
        self.thread.start()
        return prev

    def _run(self, stop):
        while not stop.is_set() and not self._closing.is_set() and self.fd is not None:
            self.write()
            self._closing.wait(config.BREADCRUMB_S)

    def close(self):
        """The clean-stop line, then close."""
        self._closing.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.fd is not None:
            self.write(exit_=True)
            with self._lock:
                try:
                    os.close(self.fd)
                except OSError:
                    pass
                self.fd = None
