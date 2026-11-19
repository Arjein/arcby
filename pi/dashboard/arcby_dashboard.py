#!/usr/bin/env python3
"""
Arcby status dashboard - read-only web page for the robot's Raspberry Pi.

Standard library only (plus python3-lgpio, already on Raspberry Pi OS, for the
distance sensor). Serves:
  /             the dashboard page (polls /api/status every 2 s, /api/sensors + /api/live every 0.5 s)
  /api/status   JSON with health, GPIO header, devices, network, processes, logs, conversation (what
                code/alive.py heard and said: text from its events.jsonl - the robot keeps no audio)
  /api/sensors  JSON with the live sensor readings (HC-SR04 distance) + camera status
  /api/live     the robot's live state from code/alive.py (~/scratch/alive/state.json: faces by name,
                attention, ears, voice, mind, conversation, sensors) + its age; polled every 0.5 s
  /camera.mjpg  live camera picture (MJPEG), only while someone watches

Read-only by design: no endpoint runs commands from the request or changes
anything. Pin states come from `pinctrl get` (never claims pins, so the motor
scripts can run at the same time). One exception: while a page is open, the
HC-SR04 is pinged 4 times a second - GPIO5/6 are claimed only for each ping
(a few ms), so code/hwtest/distance.py can still run; then the card shows
'busy'. Set ARCBY_DASHBOARD_DISTANCE=0 to turn pinging off.
Second exception: the Camera card's Start button runs rpicam-vid while the
stream is watched (stops 3 s after the last viewer). ARCBY_DASHBOARD_CAMERA=0
turns it off.
Third exception: the Speaker card's slider writes ~/.config/arcby/speaker_level.json
({"level": 0..1}, POST /api/speaker-level) - the robot's voice reads it before every
sentence. Nothing else is ever written.

Run:  python3 arcby_dashboard.py          (listens on 0.0.0.0:8080)
Docs: docs/dashboard.md
"""
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arcby_archive as archive                     # noqa: E402 - the People / Heard tabs (next to this file)

try:
    import lgpio
except ImportError:  # not on a Pi / package missing: the distance card says so
    lgpio = None

HOST = os.environ.get("ARCBY_DASHBOARD_HOST", "0.0.0.0")
PORT = int(os.environ.get("ARCBY_DASHBOARD_PORT", "8080"))
DISTANCE_ON = os.environ.get("ARCBY_DASHBOARD_DISTANCE", "1") != "0"
PID_FILE = Path(__file__).resolve().with_name("dashboard.pid")

CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")

# --------------------------------------------------------------------------
# Arcby pin map (copied from docs/pinout.md - keep the two in sync)
# --------------------------------------------------------------------------

# Physical pin -> (BCM GPIO number or None, header name)
HEADER = {
    1: (None, "3V3"), 2: (None, "5V"),
    3: (2, "GPIO2 SDA1"), 4: (None, "5V"),
    5: (3, "GPIO3 SCL1"), 6: (None, "GND"),
    7: (4, "GPIO4"), 8: (14, "GPIO14 TXD"),
    9: (None, "GND"), 10: (15, "GPIO15 RXD"),
    11: (17, "GPIO17"), 12: (18, "GPIO18"),
    13: (27, "GPIO27"), 14: (None, "GND"),
    15: (22, "GPIO22"), 16: (23, "GPIO23"),
    17: (None, "3V3"), 18: (24, "GPIO24"),
    19: (10, "GPIO10"), 20: (None, "GND"),
    21: (9, "GPIO9"), 22: (25, "GPIO25"),
    23: (11, "GPIO11"), 24: (8, "GPIO8"),
    25: (None, "GND"), 26: (7, "GPIO7"),
    27: (0, "GPIO0 ID_SD"), 28: (1, "GPIO1 ID_SC"),
    29: (5, "GPIO5"), 30: (None, "GND"),
    31: (6, "GPIO6"), 32: (12, "GPIO12"),
    33: (13, "GPIO13"), 34: (None, "GND"),
    35: (19, "GPIO19"), 36: (16, "GPIO16"),
    37: (26, "GPIO26"), 38: (20, "GPIO20"),
    39: (None, "GND"), 40: (21, "GPIO21"),
}

# Every wire between a module and the Pi (copied from docs/pinout.md + docs/wiring.md -
# keep them in sync). One entry per module wire: (wire, Pi pin or None, via).
# "bb N" = mini breadboard column N (top = 3.3 V side, bottom = 4.7 V side).
WIRING = [
    {"module": "Motors (L298N)", "short": "L298N", "group": "motors", "wires": [
        ("ENA", 11, "direct (cap removed)"), ("IN1", 13, "direct"), ("IN2", 15, "direct"),
        ("IN3", 16, "direct"), ("IN4", 18, "direct"), ("ENB", 22, "direct (cap removed)"),
        ("GND screw terminal", 14, "direct, same terminal as the - Wago wire")]},
    {"module": "IMU (MPU6050)", "short": "MPU", "group": "imu", "wires": [
        ("VCC", 1, "bb 4 top (3.3 V)"), ("GND", 25, "direct"),
        ("SDA", 3, "bb 2 top"), ("SCL", 5, "bb 3 top")]},
    {"module": "Distance (HC-SR04)", "short": "HC-SR04", "group": "distance", "wires": [
        ("VCC", 1, "bb 4 top (3.3 V) - never 5 V"), ("TRIG", 29, "direct"),
        ("ECHO", 31, "direct (safe on 3.3 V)"), ("GND", 30, "direct")]},
    {"module": "Fan (Armor Lite)", "short": "fan", "group": "fan", "wires": [
        ("red +", 4, "direct"), ("black -", 6, "direct"), ("blue control", 7, "direct (gpio-fan overlay)")]},
    {"module": "Pan-tilt (PTZ board)", "short": "PTZ", "group": "pantilt", "wires": [
        ("SDA", 3, "bb 2 top"), ("SCL", 5, "bb 3 top"),
        ("VCC", None, "bb 1 bottom = 4.7 V servo buck OUT+ (never Pi 3.3 V)"),
        ("GND", None, "bb 2 bottom = servo buck OUT-")]},
    {"module": "Amp (MAX98357A)", "short": "amp", "group": "audio", "wires": [
        ("VIN", 2, "direct"), ("GND", 34, "direct"), ("DIN", 40, "direct"),
        ("BCLK", 12, "bb 5 top"), ("LRC", 35, "bb 6 top"),
        ("GAIN", None, "not connected"), ("SD", None, "not connected")]},
    {"module": "Mic (INMP441)", "short": "mic", "group": "audio", "wires": [
        ("VDD", 17, "direct"), ("GND", 39, "direct"), ("SD", 38, "direct"),
        ("SCK", 12, "bb 5 top"), ("WS", 35, "bb 6 top"), ("L/R", None, "to the mic's own GND pin")]},
    {"module": "Camera (Module 3 Wide)", "short": "camera", "group": "camera", "wires": [
        ("ribbon", None, "CSI camera connector on the Pi, not the header")]},
]

# Pi pin -> breadboard column, for the Pi -> breadboard jumpers.
BREADBOARD = [
    (1, "bottom", "4.7 V", "servo buck OUT+", "PTZ VCC"),
    (2, "bottom", "GND (servo)", "servo buck OUT-", "PTZ GND"),
    (2, "top", "SDA", "Pi pin 3", "MPU SDA, PTZ SDA"),
    (3, "top", "SCL", "Pi pin 5", "MPU SCL, PTZ SCL"),
    (4, "top", "3.3 V", "Pi pin 1", "MPU VCC, HC-SR04 VCC"),
    (5, "top", "I2S BCLK", "Pi pin 12", "amp BCLK, mic SCK"),
    (6, "top", "I2S LRCLK", "Pi pin 35", "amp LRC, mic WS"),
    ("14-17", "both", "reserved", "", "BSS138 level shifter, only if the PTZ board pulls SDA/SCL to 4.7 V"),
]

# Pins shared by several modules get the bus colour instead of the first module's.
PIN_GROUP = {3: "pantilt", 5: "pantilt"}
RESERVED = {27: "HAT EEPROM - avoid", 28: "HAT EEPROM - avoid", 36: "audio overlay (GPIO16) - avoid"}


def _pin_map():
    """Physical pin -> (group, label), built from WIRING so the header and the list agree."""
    users = {}
    for m in WIRING:
        for wire, pin, via in m["wires"]:
            if pin is not None:
                users.setdefault(pin, []).append((m, wire, via))
    pins = {pin: ("reserved", text) for pin, text in RESERVED.items()}
    for pin, us in users.items():
        label = " + ".join(f"{m['short']} {wire}" for m, wire, _ in us)
        via = us[0][2]
        if via.startswith("bb "):
            label += f" ({via.split(' (')[0].replace('bb', 'breadboard col')})"
        pins[pin] = (PIN_GROUP.get(pin, us[0][0]["group"]), label)
    return pins


ARCBY_PINS = _pin_map()

THROTTLE_BITS = [
    (0, 16, "under-voltage"),
    (1, 17, "ARM freq capped"),
    (2, 18, "throttled"),
    (3, 19, "soft temp limit"),
]

PINCTRL_FUNCS = {"ip": "input", "op": "output", "no": "none"}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def run(cmd, timeout=3):
    """Run a fixed command list; return stdout. Raises RuntimeError on failure."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError(f"{cmd[0]} not installed")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{cmd[0]} timed out after {timeout} s")
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).strip().splitlines()
        raise RuntimeError(f"{cmd[0]} exit {p.returncode}: {msg[-1] if msg else ''}")
    return p.stdout


def read(path):
    return Path(path).read_text()


def safe(fn, *args):
    """Call fn; turn any exception into {'error': ...} so one source never breaks the page."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - dashboard must never crash
        return {"error": f"{type(e).__name__}: {e}"}


_cache = {}
_cache_lock = threading.Lock()


def cached(key, ttl, fn):
    """Return fn() cached for ttl seconds (errors are cached too, so a broken tool is not hammered)."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = safe(fn)
    with _cache_lock:
        _cache[key] = (now, value)
    return value


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

_prev_cpu = None


def read_cpu_times():
    out = {}
    for line in read("/proc/stat").splitlines():
        if not line.startswith("cpu"):
            break
        parts = line.split()
        vals = [int(v) for v in parts[1:9]]
        idle = vals[3] + vals[4]  # idle + iowait
        out[parts[0]] = (sum(vals), idle)
    return out


def cpu_usage():
    global _prev_cpu
    cur = read_cpu_times()
    prev, _prev_cpu = _prev_cpu, cur
    if not prev:
        return None
    usage = {}
    for name, (total, idle) in cur.items():
        if name not in prev:
            continue
        dt = total - prev[name][0]
        di = idle - prev[name][1]
        usage[name] = round(100.0 * (dt - di) / dt, 1) if dt > 0 else 0.0
    return usage


def throttled():
    out = run(["vcgencmd", "get_throttled"]).strip()
    value = int(out.split("=", 1)[1], 16)
    flags = [{"flag": name, "now": bool(value >> now & 1), "since_boot": bool(value >> past & 1)}
             for now, past, name in THROTTLE_BITS]
    return {"raw": hex(value), "flags": flags}


def meminfo():
    info = {}
    for line in read("/proc/meminfo").splitlines():
        key, rest = line.split(":", 1)
        info[key] = int(rest.split()[0]) * 1024
    total, avail = info["MemTotal"], info["MemAvailable"]
    return {"total": total, "available": avail, "used": total - avail,
            "swap_total": info.get("SwapTotal", 0),
            "swap_used": info.get("SwapTotal", 0) - info.get("SwapFree", 0)}


def core_freqs():
    freqs = []
    for d in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        f = d / "cpufreq" / "scaling_cur_freq"
        if f.exists():
            freqs.append({"cpu": d.name, "mhz": int(f.read_text()) // 1000})
    return freqs


def health():
    usage = cpu_usage() or {}
    st = os.statvfs("/")
    volts = safe(lambda: float(run(["vcgencmd", "measure_volts", "core"]).strip()
                               .split("=")[1].rstrip("V")))
    return {
        "temp_c": safe(lambda: int(read("/sys/class/thermal/thermal_zone0/temp")) / 1000),
        "throttled": safe(throttled),
        "cpu_percent": usage.get("cpu"),
        "core_percent": [usage[k] for k in sorted(usage) if k != "cpu"],
        "core_freq": safe(core_freqs),
        "core_volts": volts,
        "load": os.getloadavg(),
        "uptime_s": float(read("/proc/uptime").split()[0]),
        "memory": safe(meminfo),
        "disk": {"total": st.f_blocks * st.f_frsize,
                 "used": (st.f_blocks - st.f_bfree) * st.f_frsize,
                 "free": st.f_bavail * st.f_frsize},
    }


# --------------------------------------------------------------------------
# GPIO header
# --------------------------------------------------------------------------

PINCTRL_RE = re.compile(r"^\s*(\d+):\s*([^|]*)\|\s*(\w+)\s*//\s*(.*)$")


def pinctrl_states():
    states = {}
    for line in run(["pinctrl", "get", "0-27"]).splitlines():
        m = PINCTRL_RE.match(line)
        if not m:
            continue
        gpio = int(m.group(1))
        tokens = m.group(2).split()
        func = tokens[0] if tokens else "?"
        pull = next((t for t in tokens if t in ("pu", "pd", "pn")), "")
        desc = m.group(4).split("=", 1)[-1].strip()
        states[gpio] = {
            "func": PINCTRL_FUNCS.get(func, "alt" if func.startswith("a") else func),
            "alt": desc if func.startswith("a") else "",
            "pull": {"pu": "up", "pd": "down", "pn": "none"}.get(pull, pull),
            "level": m.group(3),
        }
    return states


def gpio():
    states = safe(pinctrl_states)
    error = states.pop("error", None)
    pins, warnings = [], []
    for pin in range(1, 41):
        num, name = HEADER[pin]
        group, label = ARCBY_PINS.get(pin, ("power" if num is None else "free",
                                            "" if num is None else "free"))
        if num is None and pin not in ARCBY_PINS:
            label = "spare " + name
        boot_high = num is not None and num <= 8
        pins.append({"pin": pin, "gpio": num, "name": name, "group": group, "label": label,
                     "power": num is None, "boot_high": boot_high,
                     "state": states.get(num) if num is not None else None})
        if group == "motors" and boot_high:
            warnings.append(f"Motor pin {pin} is on GPIO{num}, which boots HIGH - wheels may spin at power-up")
        if num == 16 and group not in ("free", "reserved"):
            warnings.append("Pin 36 (GPIO16) is used by Arcby but claimed by the audio overlay")
    return {"pins": pins, "warnings": warnings, "error": error}


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


def camera():
    out = run(["rpicam-hello", "--list-cameras"], timeout=5)
    cams = [l.strip() for l in out.splitlines() if re.match(r"^\s*\d+\s*:", l)]
    return {"cameras": cams, "text": out.strip()}


def sound_cards():
    lines = read("/proc/asound/cards").splitlines()
    cards = [l.strip() for l in lines if re.match(r"^\s*\d+\s*\[", l)]
    return {"cards": cards}


def i2c_scan():
    if not shutil.which("i2cdetect"):
        return {"error": "i2c-tools not installed"}
    out = run(["i2cdetect", "-y", "1"], timeout=5)
    found = []
    for line in out.splitlines()[1:]:
        row, _, cells = line.partition(":")
        base = int(row, 16)
        for i, cell in enumerate(_cells(cells)):
            if cell not in ("--", ""):
                found.append({"addr": f"0x{base + i:02x}", "in_use_by_driver": cell == "UU"})
    return {"bus": 1, "devices": found}


def _cells(cells):
    # Each cell is 3 chars wide (" xx"); empty cells are spaces.
    return [cells[i:i + 3].strip() for i in range(0, 48, 3)]


def usb():
    return {"devices": [l.strip() for l in run(["lsusb"]).splitlines() if l.strip()]}


def devices():
    return {
        "camera": cached("camera", 60, camera),
        "sound": cached("sound", 30, sound_cards),
        "i2c": cached("i2c", 60, i2c_scan),
        "usb": cached("usb", 60, usb),
    }


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def interfaces():
    data = json.loads(run(["ip", "-j", "addr", "show"]))
    return [{"name": i.get("ifname"), "state": i.get("operstate"),
             "addrs": [f"{a['local']}/{a['prefixlen']}" for a in i.get("addr_info", [])]}
            for i in data]


def wifi():
    out = []
    for line in read("/proc/net/wireless").splitlines()[2:]:
        name, _, rest = line.partition(":")
        f = rest.split()
        out.append({"iface": name.strip(), "link": float(f[1]), "level_dbm": float(f[2])})
    return out


def ssid():
    if not shutil.which("iwgetid"):
        return {"error": "iwgetid not installed"}
    try:
        return {"ssid": run(["iwgetid", "-r"]).strip()}
    except RuntimeError:
        return {"ssid": ""}  # exit 255 = not associated


def network():
    return {
        "hostname": socket.gethostname(),
        "interfaces": cached("ifaces", 30, interfaces),
        "wifi": safe(wifi),
        "ssid": cached("ssid", 30, ssid),
    }


# --------------------------------------------------------------------------
# Processes
# --------------------------------------------------------------------------

_prev_proc = ({}, 0.0)


def processes():
    global _prev_proc
    now = time.monotonic()
    prev_ticks, prev_t = _prev_proc
    dt = now - prev_t if prev_t else 0
    procs, ticks = [], {}
    for d in os.scandir("/proc"):
        if not d.name.isdigit():
            continue
        try:
            raw = read(f"/proc/{d.name}/stat")
        except OSError:
            continue  # process exited
        pid = int(d.name)
        comm = raw[raw.index("(") + 1:raw.rindex(")")]
        f = raw[raw.rindex(")") + 2:].split()
        t = int(f[11]) + int(f[12])
        ticks[pid] = t
        cpu = None
        if dt > 0 and pid in prev_ticks:
            cpu = round(100.0 * (t - prev_ticks[pid]) / CLK_TCK / dt, 1)
        procs.append({"pid": pid, "name": comm, "rss": int(f[21]) * PAGE_SIZE, "cpu": cpu})
    _prev_proc = (ticks, now)

    def with_cmd(p):
        try:
            cmd = read(f"/proc/{p['pid']}/cmdline").replace("\0", " ").strip()
        except OSError:
            cmd = ""
        return dict(p, cmd=(cmd or p["name"])[:100])

    by_mem = [with_cmd(p) for p in sorted(procs, key=lambda p: p["rss"], reverse=True)[:8]]
    by_cpu = [with_cmd(p) for p in sorted(procs, key=lambda p: p["cpu"] or 0, reverse=True)[:8]]
    return {"count": len(procs), "by_mem": by_mem, "by_cpu": by_cpu,
            "self": {"pid": os.getpid(), "rss": self_rss()}}


def self_rss():
    for line in read("/proc/self/status").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


# --------------------------------------------------------------------------
# Sensors (live, own endpoint)
# --------------------------------------------------------------------------

HCSR04_TRIG, HCSR04_ECHO = 5, 6     # GPIO (pins 29 / 31), docs/pinout.md
SOUND_M_S = 343.0                   # speed of sound at ~20 C
PING_PERIOD_S = 0.25                # 4 pings/s (the sensor needs >= 60 ms between pings)
VIEW_TIMEOUT_S = 10.0               # stop pinging when no page asked for this long
HISTORY_S = 10.0


def robot_running(proc="/proc"):
    """Is code/alive.py (or another robot program that pings the sonar) running? Then the dashboard must not ping:
    two programs pinging one HC-SR04 hear each other's echoes (ghost readings in the robot's recordings)."""
    try:
        pids = [d for d in os.listdir(proc) if d.isdigit()]
    except OSError:
        return False
    for pid in pids:
        try:
            with open(f"{proc}/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ")
        except OSError:
            continue
        if any(name in cmd for name in (b"code/alive.py", b"code/body_mcp.py", b"code/drive_web.py")):
            return True
    return False


class DistanceSampler:
    """Pings the HC-SR04 in a background thread, only while a page is open and no robot program uses it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque(maxlen=int(HISTORY_S / PING_PERIOD_S))   # (t, cm or None)
        self.status = "idle"
        self.last_view = 0.0
        self.chip = None
        self._robot, self._next_robot_check = False, 0.0

    def viewed(self):
        self.last_view = time.monotonic()

    def run(self):
        while True:
            if time.monotonic() - self.last_view > VIEW_TIMEOUT_S:
                with self.lock:
                    self.status = "idle"
                    self.history.clear()
                time.sleep(0.5)
                continue
            t = time.monotonic()
            if t >= self._next_robot_check:
                self._robot = robot_running()
                self._next_robot_check = t + 2.0
            if self._robot:                         # alive.py pings it itself: stay off the pins
                with self.lock:
                    self.status = "robot"
                    self.history.clear()
                time.sleep(0.5)
                continue
            result = self.ping()
            with self.lock:
                self.status = result if isinstance(result, str) else "ok"
                if result in ("ok", "nothing") or isinstance(result, float):
                    self.history.append((t, result if isinstance(result, float) else None))
            time.sleep(max(0.0, PING_PERIOD_S - (time.monotonic() - t)))

    def ping(self):
        """cm (float), or a status: 'nothing' / 'noanswer' / 'busy' / 'error: ...'."""
        try:
            if self.chip is None:
                self.chip = lgpio.gpiochip_open(0)
            h = self.chip
            lgpio.gpio_claim_output(h, HCSR04_TRIG, 0)
        except lgpio.error as e:
            return "busy" if "busy" in str(e).lower() else f"error: {e}"
        try:
            lgpio.gpio_claim_input(h, HCSR04_ECHO, lgpio.SET_PULL_DOWN)
        except lgpio.error as e:
            lgpio.gpio_free(h, HCSR04_TRIG)
            return "busy" if "busy" in str(e).lower() else f"error: {e}"
        try:
            lgpio.gpio_write(h, HCSR04_TRIG, 1)
            time.sleep(0.00001)                     # >= 10 us trigger pulse
            lgpio.gpio_write(h, HCSR04_TRIG, 0)
            t0 = time.perf_counter()
            while not lgpio.gpio_read(h, HCSR04_ECHO):
                if time.perf_counter() - t0 > 0.03:
                    return "noanswer"
            start = time.perf_counter()
            while lgpio.gpio_read(h, HCSR04_ECHO):
                if time.perf_counter() - start > 0.04:
                    return "nothing"                # no echo within ~6.8 m
            return round((time.perf_counter() - start) * SOUND_M_S / 2 * 100, 1)
        finally:
            lgpio.gpio_free(h, HCSR04_TRIG)         # TRIG stays an output, LOW
            lgpio.gpio_free(h, HCSR04_ECHO)

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            hist = list(self.history)
            status = self.status
        vals = [cm for _, cm in hist if cm is not None]
        recent = vals[-8:]
        return {
            "status": status,
            "cm": hist[-1][1] if hist else None,
            "age_s": round(now - hist[-1][0], 2) if hist else None,
            "median_cm": round(sorted(recent)[len(recent) // 2], 1) if recent else None,
            "spread_cm": round(max(recent) - min(recent), 1) if recent else None,
            "history": [cm for _, cm in hist],
            "period_s": PING_PERIOD_S,
            "pins": "TRIG GPIO5 (pin 29) · ECHO GPIO6 (pin 31)",
        }


_distance = DistanceSampler() if (DISTANCE_ON and lgpio is not None) else None


def sensors_json():
    if _distance is None:
        why = "turned off (ARCBY_DASHBOARD_DISTANCE=0)" if not DISTANCE_ON else "python3-lgpio not installed"
        dist = {"status": f"error: {why}"}
    else:
        _distance.viewed()
        dist = _distance.snapshot()
    cam = _camera.snapshot() if _camera else {"status": "error: turned off (ARCBY_DASHBOARD_CAMERA=0)"}
    return json.dumps({"time": time.time(), "distance": dist, "camera": cam}, separators=(",", ":")).encode()


# --------------------------------------------------------------------------
# Camera (live view)
# --------------------------------------------------------------------------

CAM_W, CAM_H, CAM_FPS = 640, 480, 10
CAM_IDLE_S = 3.0                    # camera stops this long after the last viewer left
CAM_FIRST_FRAME_S = 8.0             # rpicam-vid needs ~2 s to start; give up after this
CAM_MAX_BUFFER = 4 * 1024 * 1024    # bytes without a complete JPEG = garbage, drop it


def camera_cmd():
    prog = shutil.which("rpicam-vid") or shutil.which("libcamera-vid") or "rpicam-vid"
    return [prog, "-t", "0", "-n", "--codec", "mjpeg", "--width", str(CAM_W), "--height", str(CAM_H),
            "--framerate", str(CAM_FPS), "--flush", "-o", "-"]


def split_jpegs(buf):
    """Remove complete JPEG frames (FFD8 ... FFD9) from the front of the bytearray, return them.
    Inside a JPEG's compressed data every 0xFF is followed by 0x00, so FFD9 only marks the end."""
    frames = []
    while True:
        start = buf.find(b"\xff\xd8")
        if start < 0:
            del buf[:-1 if buf.endswith(b"\xff") else len(buf)]    # keep a half marker
            return frames
        end = buf.find(b"\xff\xd9", start + 2)
        if end < 0:
            del buf[:start]
            return frames
        frames.append(bytes(buf[start:end + 2]))
        del buf[:end + 2]


class CameraStreamer:
    """Runs rpicam-vid (MJPEG to stdout) only while a browser watches /camera.mjpg, and shares the
    latest frame between all viewers. The camera is free for other programs (code/hwtest/bringup.py,
    later the robot) a few seconds after the last viewer closed the stream."""

    def __init__(self, cmd=None, idle_s=CAM_IDLE_S):
        self.cmd = cmd or camera_cmd()
        self.idle_s = idle_s
        self.cond = threading.Condition()
        self.frame, self.seq = None, 0
        self.viewers = 0
        self.proc = None
        self.status = "off"
        self.error = ""
        self.frame_times = deque(maxlen=20)

    def open_viewer(self):
        with self.cond:
            self.viewers += 1
            if self.proc is None:
                self._start()

    def close_viewer(self):
        with self.cond:
            self.viewers -= 1
            if self.viewers == 0:
                t = threading.Timer(self.idle_s, self._stop_if_unwatched)
                t.daemon = True
                t.start()

    def _start(self):
        self.frame, self.status, self.error = None, "starting", ""
        self.frame_times.clear()
        try:
            proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    stdin=subprocess.DEVNULL)
        except OSError as e:
            self.status, self.error = "error", f"{self.cmd[0]}: {e}"
            return
        self.proc = proc
        err = threading.Thread(target=self._read_stderr, args=(proc,), name="camera-err", daemon=True)
        err.start()
        threading.Thread(target=self._read, args=(proc, err), name="camera", daemon=True).start()

    def _stop_if_unwatched(self):
        with self.cond:
            if self.viewers == 0 and self.proc is not None:
                self.proc.terminate()
                self.proc, self.status, self.error = None, "off", ""
                self.cond.notify_all()

    def _read(self, proc, err_thread):
        buf = bytearray()
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            buf += chunk
            frames = split_jpegs(buf)
            if len(buf) > CAM_MAX_BUFFER:
                buf.clear()
            if frames:
                with self.cond:
                    self.frame = frames[-1]
                    self.seq += 1
                    self.status = "live"
                    self.frame_times.append(time.monotonic())
                    self.cond.notify_all()
        code = proc.wait()
        proc.stdout.close()
        err_thread.join(1.0)                    # collect the last error lines first
        with self.cond:
            if self.proc is proc:
                self.proc = None
                if self.viewers == 0:           # stopped because nobody watches
                    self.status, self.error = "off", ""
                else:                           # died while watched: camera busy / missing
                    self.status = "error"
                    self.error = self._why(proc) or f"camera program ended (exit code {code})"
                self.cond.notify_all()

    def _read_stderr(self, proc):
        proc.err_lines = deque(maxlen=8)
        for raw in proc.stderr:
            line = raw.decode(errors="replace").strip()
            if line:
                proc.err_lines.append(line)
        proc.stderr.close()

    @staticmethod
    def _why(proc):
        lines = list(getattr(proc, "err_lines", []))
        errors = [l for l in lines if re.search(r"error|failed|busy|no cameras", l, re.I)]
        return (errors or lines or [""])[-1]

    def wait_frame(self, after_seq, timeout):
        """(seq, jpeg) of the next frame newer than after_seq; None on timeout or camera stop."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after_seq or self.proc is None, timeout)
            if self.seq > after_seq and self.frame is not None:
                return self.seq, self.frame
            return None

    def snapshot(self):
        with self.cond:
            ft = list(self.frame_times)
            fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 1 and ft[-1] > ft[0] else None
            return {"status": self.status, "error": self.error, "viewers": self.viewers,
                    "fps": round(fps, 1) if fps else None,
                    "size": f"{CAM_W}x{CAM_H}", "kib": len(self.frame) // 1024 if self.frame else None}


CAMERA_ON = os.environ.get("ARCBY_DASHBOARD_CAMERA", "1") != "0"
_camera = CameraStreamer() if CAMERA_ON else None


# --------------------------------------------------------------------------
# Logs + full status
# --------------------------------------------------------------------------


def logs():
    return {"lines": run(["journalctl", "-n", "30", "--no-pager"], timeout=5).splitlines()}


LEVEL_MAX_PCT = 70      # the robot never plays louder (robot/config.py TTS_LEVEL_MAX): full scale overdrove the
                        # small speaker (muffled after hours at 100 %, 2026-11-01)
LEVEL_FILE = Path(os.environ.get("ARCBY_LEVEL_FILE", "~/.config/arcby/speaker_level.json")).expanduser()


def speaker_level():
    """{"level": 0..1 or None (not set: the robot's config decides)}"""
    try:
        return {"level": min(1.0, max(0.0, float(json.loads(LEVEL_FILE.read_text())["level"])))}
    except (OSError, ValueError, KeyError, TypeError):
        return {"level": None}


def set_speaker_level(body):
    """POST body {"level": 0..100 (percent)} -> the file (atomic). Returns the new state."""
    pct = float(json.loads(body or b"{}")["level"])
    if not 0 <= pct <= 100:
        raise ValueError("level must be 0-100")
    pct = min(pct, LEVEL_MAX_PCT)           # protects the speaker
    LEVEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEVEL_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"level": round(pct / 100, 3)}))
    os.replace(tmp, LEVEL_FILE)
    return speaker_level()


ALIVE_DIR = Path(os.environ.get("ARCBY_ALIVE_DIR", "~/scratch/alive")).expanduser()
TALK_KINDS = {"heard", "said", "dropped", "voice", "called", "not_for_me", "session_end", "met", "forgot", "turn",
              "interrupted", "own_echo", "showing", "shown"}


def _tail(path, size):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - size))
        return f.read().decode("utf-8", "replace").splitlines()[1:]   # the first line may be cut


def conversation(n=30):
    """The robot's last heard / said / dropped sentences and voices (alive.py's events.jsonl), and the ears' state
    from its latest status line ("ears 3 heard, gain x80")."""
    items = []
    for line in _tail(ALIVE_DIR / "events.jsonl", 262144):
        if '"kind"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") in TALK_KINDS:
            items.append({k: e.get(k) for k in ("kind", "t", "text", "name", "by", "sim", "seconds", "error", "how",
                                                "called", "pan", "tilt")})
    ears = None
    for line in reversed(_tail(ALIVE_DIR / "alive.log", 65536)):
        i = line.find("| ears ")
        if i >= 0:
            ears = {"at": line[11:19], "state": line[i + 7:].strip()}
            break
    return {"items": items[-n:], "ears": ears}


LIVE_STALE_S = 3.0                  # alive.py writes state.json twice a second; older = not running / hung


def live_state(path=None, now=None):
    """alive.py's live state (~/scratch/alive/state.json, written twice a second) + its age. Never raises:
    a missing / broken / half-written file gives {"state": None, "why": ...}."""
    path = ALIVE_DIR / "state.json" if path is None else Path(path)
    now = time.time() if now is None else now
    out = {"time": now, "running": cached("robot_running", 2, robot_running) is True, "age_s": None,
           "stale": True, "state": None, "why": None}
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        out["why"] = "no state file (alive.py has not run since boot)"
        return out
    except OSError as e:
        out["why"] = f"cannot read state file: {e}"
        return out
    try:
        state = json.loads(raw.decode("utf-8", "replace"))
        if not isinstance(state, dict):
            raise ValueError("not a JSON object")
    except ValueError as e:
        out["why"] = f"broken state file: {e}"
        return out
    wall = state.get("wall")
    if not isinstance(wall, (int, float)):
        try:
            wall = path.stat().st_mtime
        except OSError:
            wall = None
    if wall is not None:
        out["age_s"] = round(max(0.0, now - wall), 1)
        out["stale"] = out["age_s"] > LIVE_STALE_S
    if out["stale"]:
        out["why"] = "state is old - alive.py stopped or hangs" if wall is not None else "state has no time"
    out["state"] = state
    return out


def live_json():
    return json.dumps(live_state(), separators=(",", ":")).encode()


# ---- learning mode (code/learn.py on wall power -> ~/scratch/learn/state.json) ----
LEARN_STATE = Path(os.environ.get("ARCBY_LEARN_STATE", "~/scratch/learn/state.json")).expanduser()
LEARN_CLI = Path(os.environ.get("ARCBY_LEARN_CLI", "~/projects/arcby/code/learn.py")).expanduser()
LEARN_ID = re.compile(r"^p\d{1,20}$")


def learn_state(path=None, now=None):
    """learn.py's progress (state.json) + its age. Never raises."""
    path = LEARN_STATE if path is None else Path(path)
    now = time.time() if now is None else now
    out = {"time": now, "state": None, "age_s": None, "why": None}
    try:
        state = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(state, dict):
            raise ValueError("not a JSON object")
    except FileNotFoundError:
        out["why"] = "it has not learnt anything yet (starts by itself on the wall adapter)"
        return out
    except (OSError, ValueError) as e:
        out["why"] = f"cannot read the learning state: {e}"
        return out
    upd = state.get("updated")
    if isinstance(upd, (int, float)):
        out["age_s"] = round(max(0.0, now - upd), 1)
    out["state"] = state
    return out


def learn_decide(raw, run=subprocess.run, cli=None):
    """{"id": "p123", "accept": true} from the page -> code/learn.py apply|reject ID (the rules live there)."""
    d = json.loads(raw.decode("utf-8"))
    if not isinstance(d, dict):
        raise ValueError("send a JSON object")
    pid, accept = d.get("id"), d.get("accept")
    if not isinstance(pid, str) or not LEARN_ID.match(pid) or not isinstance(accept, bool):
        raise ValueError("send {\"id\": \"p...\", \"accept\": true|false}")
    r = run([sys.executable, str(cli or LEARN_CLI), "apply" if accept else "reject", pid],
            capture_output=True, text=True, timeout=30)
    return {"ok": r.returncode == 0, "id": pid, "accept": accept,
            "message": (r.stderr or r.stdout or "").strip()[-300:]}


def wiring():
    """Static wiring list for the page: every module wire with its Pi pin and where that pin is."""
    modules = []
    for m in WIRING:
        rows = []
        for wire, pin, via in m["wires"]:
            row = {"wire": wire, "pin": pin, "via": via}
            if pin is not None:
                row["name"] = HEADER[pin][1]
                row["where"] = f"{'inner' if pin % 2 else 'outer'} row, position {(pin + 1) // 2} of 20 from the SD-card end"
            rows.append(row)
        modules.append({"module": m["module"], "group": m["group"], "wires": rows})
    bb = [{"col": c, "half": h, "net": n, "src": f, "dst": t} for c, h, n, f, t in BREADBOARD]
    return {"modules": modules, "breadboard": bb}


_status_lock = threading.Lock()
_status = (0.0, b"")


DATA_DISK = Path(os.environ.get("ARCBY_DATA_DISK", "/mnt/arcby-data"))   # the USB stick (robot/config.py DATA_DISK)
GB = 1_000_000_000


def disk_free(path):
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return None


def alerts(h, stick_mounted, stick_free, alive, head_on):
    """What Selim must see on every tab, worst first: [{level: danger / warn / info, text, detail}]."""
    out = []
    flags = {f["flag"]: f for f in ((h or {}).get("throttled") or {}).get("flags", [])}
    uv = flags.get("under-voltage", {})
    if uv.get("now"):
        out.append(dict(level="danger", text="UNDER-VOLTAGE NOW - the battery is nearly empty or the power supply is "
                        "too weak", detail="Charge the battery, or use the 5 V 3 A wall adapter. The robot stops its "
                        "motors by itself; the Pi may switch off any moment."))
    elif uv.get("since_boot"):
        out.append(dict(level="warn", text="Under-voltage happened since the last boot",
                        detail="The battery dipped (or the cable / supply is weak). Charge it before a long run."))
    t = (h or {}).get("temp_c")
    if isinstance(t, (int, float)) and t >= 80:
        out.append(dict(level="danger", text=f"The Pi is HOT: {t:.0f} °C - it is slowing itself down",
                        detail="Is the fan running? Free the air around the upper deck."))
    elif isinstance(t, (int, float)) and t >= 75:
        out.append(dict(level="warn", text=f"The Pi is warm: {t:.0f} °C", detail="Learning pauses above 75 °C."))
    elif flags.get("throttled", {}).get("now"):
        out.append(dict(level="warn", text="The Pi is throttled right now", detail=""))
    free = ((h or {}).get("disk") or {}).get("free")
    if isinstance(free, (int, float)) and free < 1 * GB:
        out.append(dict(level="danger", text=f"The SD card is almost full ({free / GB:.1f} GB free)",
                        detail="Programs and logs fail when it is full."))
    elif isinstance(free, (int, float)) and free < 3 * GB:
        out.append(dict(level="warn", text=f"The SD card is filling up ({free / GB:.1f} GB free)", detail=""))
    if not stick_mounted:
        out.append(dict(level="danger", text="The USB stick is not mounted - nothing is being recorded",
                        detail="Plug it in (the SanDisk, label arcby-data) and reboot, or check `findmnt /mnt/arcby-data`."))
    elif isinstance(stick_free, (int, float)) and stick_free < 5 * GB:
        out.append(dict(level="warn", text=f"The USB stick is filling up ({stick_free / GB:.0f} GB free)",
                        detail="The oldest recordings are deleted by themselves at the limit."))
    if not alive:
        out.append(dict(level="warn", text="The robot program (alive.py) is not running",
                        detail="It starts at boot; after a clean stop it stays off until the next boot."))
    elif head_on is False:
        out.append(dict(level="info", text="On wall power / battery off: no head, no motors", detail=""))
    return out


def alerts_now(h=None):
    h = safe(health) if h is None else h
    st = live_state()
    head = ((st.get("state") or {}).get("head") or {}).get("on") if not st.get("stale") else None
    return alerts(h if isinstance(h, dict) else {}, os.path.ismount(DATA_DISK), disk_free(DATA_DISK),
                  archive.alive_running(), head)


def status_json():
    """Full status, built at most once per second however many browsers poll."""
    global _status
    with _status_lock:
        if time.monotonic() - _status[0] < 1.0:
            return _status[1]
        h = safe(health)
        data = {
            "time": time.time(),
            "health": h,
            "gpio": safe(gpio),
            "devices": safe(devices),
            "network": safe(network),
            "processes": safe(processes),
            "logs": cached("logs", 10, logs),
            "talk": safe(conversation),
            "speaker": safe(speaker_level),
            "wiring": cached("wiring", 3600, wiring),
            "alerts": safe(alerts_now, h),
        }
        body = json.dumps(data, separators=(",", ":")).encode()
        _status = (time.monotonic(), body)
        return body


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arcby status</title>
<style>
:root{--bg:#0b0e11;--card:#12171c;--line:#222b33;--fg:#c9d4dc;--dim:#6b7a86;--ok:#4ade80;--warn:#facc15;--bad:#f87171;
--motors:#f97316;--pantilt:#38bdf8;--imu:#a3e635;--camera:#94a3b8;--fan:#a78bfa;--distance:#f472b6;--audio:#34d399;--power:#ef4444;--free:#3b4650;--reserved:#64748b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:10px 12px 24px}
h1{font-size:15px;margin:0 0 8px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
h1 small{color:var(--dim);font-weight:normal}
h2{font-size:12px;margin:0 0 6px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;min-width:0}
.wide{grid-column:1/-1}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px}
.kv b{color:var(--dim);font-weight:normal}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.dim{color:var(--dim)}
.err{color:var(--bad)}
.tag{display:inline-block;padding:0 5px;border-radius:3px;border:1px solid var(--line);margin:1px 3px 1px 0}
.tag.on{border-color:var(--bad);color:var(--bad)}.tag.past{border-color:var(--warn);color:var(--warn)}
table{border-collapse:collapse;width:100%}
td,th{padding:1px 6px 1px 0;text-align:left;vertical-align:top;white-space:nowrap}
th{color:var(--dim);font-weight:normal}
td.cmd{white-space:normal;word-break:break-all}
.scroll{overflow-x:auto}
pre{margin:0;white-space:pre-wrap;word-break:break-all;font:inherit;font-size:11px;max-height:420px;overflow:auto}
.hdr{display:grid;grid-template-columns:1fr 1fr;gap:2px 6px}
.pin{display:flex;align-items:center;gap:6px;border-left:3px solid var(--g);padding:1px 5px;background:#0e1317;min-width:0}
.pin.l{flex-direction:row-reverse;border-left:0;border-right:3px solid var(--g);text-align:right}
.pin .n{width:22px;height:18px;flex:none;border-radius:50%;display:grid;place-items:center;background:var(--g);color:#000;font-weight:bold;font-size:10px}
.pin .t{min-width:0;flex:1;overflow:hidden}
.pin .t div{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pin .s{flex:none;font-size:10px}
.lv{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 2px;vertical-align:middle}
.lv.hi{background:var(--ok)}.lv.lo{background:#334}
.legend span{margin-right:10px;white-space:nowrap}
.legend i{display:inline-block;width:10px;height:10px;background:var(--g);margin-right:4px;vertical-align:middle}
.bar{height:6px;background:#1d252c;border-radius:3px;overflow:hidden;margin-top:2px}
.bar i{display:block;height:100%;background:var(--ok)}
#conn.bad{font-weight:bold}
.sensor{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;align-items:center}
.big{font-size:30px;font-weight:bold;color:var(--distance);line-height:1}
.big small{font-size:13px;color:var(--dim);font-weight:normal}
.big.stale{color:var(--dim)}
.spark{width:100%;height:64px;display:block;background:#0e1317;border-radius:4px}
.spark polyline{fill:none;stroke:var(--distance);stroke-width:1.5;vector-effect:non-scaling-stroke}
.spark line{stroke:var(--line);stroke-width:1;vector-effect:non-scaling-stroke}
.spark text{fill:var(--dim);font-size:9px}
.wires{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:8px}
.mod{border-left:3px solid var(--g);padding:2px 0 2px 8px;min-width:0}
.mod h3{margin:0 0 2px;font-size:12px;color:var(--g)}
.mod td.p{font-weight:bold;color:var(--fg)}
.mod td{white-space:normal}
.camwrap{background:#0e1317;border-radius:4px;aspect-ratio:4/3;display:grid;place-items:center;overflow:hidden}
#camimg{width:100%;height:100%;object-fit:contain;display:none}
#camimg.rot{transform:rotate(180deg)}
.camctl{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:6px 0 4px}
button{font:inherit;background:#1d252c;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:3px 10px;cursor:pointer}
button:hover{border-color:var(--camera)}
.banner{padding:8px 10px;margin-bottom:10px;border:1px solid var(--bad);border-radius:6px;color:var(--bad);font-weight:bold;background:#1a1012}
.banner small{display:block;color:var(--dim);font-weight:normal}
#live{margin-bottom:10px}
#live .card{border-color:#2a3a46}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{border:1px solid var(--line);border-radius:6px;padding:6px 9px;min-width:150px;flex:1 1 150px;max-width:260px;background:#0e1317}
.chip.target{border-color:var(--pantilt);box-shadow:0 0 0 1px var(--pantilt) inset}
.chip .nm{font-size:17px;font-weight:bold;color:var(--c);display:flex;justify-content:space-between;align-items:center;gap:6px}
.badge{display:inline-flex;align-items:center;gap:3px;padding:0 5px;border-radius:3px;font-size:11px;font-weight:normal;border:1px solid currentColor;white-space:nowrap}
.badge svg{width:13px;height:13px}
.state{font-size:16px;font-weight:bold;margin-bottom:4px}
.empty{font-size:14px;color:var(--dim);padding:6px 0}
.people{display:flex;flex-wrap:wrap;gap:6px}
.person{border:1px solid var(--line);border-radius:4px;padding:3px 7px;background:#0e1317}
.person.seen{border-color:var(--ok)}
.sub{margin:8px 0 3px;font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
.sub:first-child{margin-top:0}
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin:0 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.tabs button{border-radius:4px 4px 0 0}
.tabs button.on{border-color:var(--pantilt);color:var(--pantilt)}
.tab{display:none}.tab.on{display:block}
.pgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.pcard{border:1px solid var(--line);border-radius:6px;padding:6px;background:#0e1317;cursor:pointer;min-width:0}
.pcard:hover,.pcard.sel{border-color:var(--pantilt)}
.pcard img,.pcard .noimg{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;background:#1d252c;display:block}
.pcard .noimg{display:grid;place-items:center;color:var(--dim)}
.pcard .nm{font-size:14px;font-weight:bold;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-top:8px}
.face{border:1px solid var(--line);border-radius:6px;padding:4px;background:#0e1317;min-width:0}
.face img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;display:block}
.face select,.face button{width:100%;margin-top:3px;font-size:11px;padding:2px 4px}
.face .cap{font-size:10px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.acts{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
.acts input,.acts select{font:inherit;background:#0e1317;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:3px 6px}
button.danger{border-color:var(--bad);color:var(--bad)}
button.armed{background:var(--bad);color:#000}
#msg{min-height:16px;margin:4px 0}
.clip{display:grid;grid-template-columns:auto 1fr auto;gap:4px 10px;align-items:center;border-bottom:1px solid var(--line);padding:4px 0}
.clip audio{height:28px;max-width:220px}
@media(max-width:480px){.pin .s .f{display:none}.pin .lab{font-size:10px}.chip{max-width:none}}
.toast{position:fixed;left:50%;bottom:16px;transform:translateX(-50%);max-width:min(92vw,560px);background:#1d252c;border:1px solid var(--line);border-radius:6px;padding:8px 12px;z-index:20;display:none;box-shadow:0 4px 18px #0008}
.toast.ok{border-color:var(--ok);color:var(--ok)}.toast.bad{border-color:var(--bad);color:var(--bad)}.toast.work{border-color:var(--pantilt)}
.spin{display:inline-block;width:10px;height:10px;border:2px solid var(--dim);border-top-color:var(--pantilt);border-radius:50%;animation:sp .8s linear infinite;vertical-align:-1px;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
.face.busy{opacity:.45;pointer-events:none}
button:disabled,select:disabled{opacity:.5;cursor:wait}
#alerts{position:sticky;top:0;z-index:15;margin:-10px -12px 10px;display:none}
.alert{padding:10px 14px;border-bottom:1px solid #0006;display:flex;gap:12px;align-items:center}
.alert .ic{font-size:30px;line-height:1;flex:none}
.alert b{font-size:17px;display:block}
.alert small{display:block;font-size:12px;opacity:.9}
.alert.danger{background:#b91c1c;color:#fff;animation:pulse 1.6s ease-in-out infinite}
.alert.warn{background:#a16207;color:#fff}
.alert.warn b{font-size:14px}
.alert.info{background:#1d252c;color:var(--dim);padding:4px 14px}
.alert.info b{font-size:12px;font-weight:normal}
.alert.info .ic{font-size:14px}
@keyframes pulse{50%{background:#7f1d1d}}
</style></head><body><div id="toast" class="toast"></div><div id="alerts"></div>
<h1>Arcby <small id="host"></small> <small id="conn">connecting...</small></h1>
<nav class="tabs" id="tabs"><button data-t="live">Live</button><button data-t="people">People</button><button data-t="heard">Heard</button><button data-t="learning">Learning</button><button data-t="system">System</button></nav>
<div class="tab" data-tab="live">
<div id="livebanner" class="banner" style="display:none"></div>
<div class="grid" id="live">
<section class="card wide live" id="lfaces"><h2>Who it sees</h2><div class="dim">loading</div></section>
<section class="card live" id="lattn"><h2>Attention</h2><div class="dim">loading</div></section>
<section class="card live" id="ldrive"><h2>Drive</h2><div class="dim">loading</div></section>
<section class="card live" id="lconv"><h2>Conversation now</h2><div class="dim">loading</div></section>
<section class="card live" id="lears"><h2>Ears</h2><div class="dim">loading</div></section>
<section class="card live" id="lvoice"><h2>Voice + mind</h2><div class="dim">loading</div></section>
<section class="card live" id="lsens"><h2>Robot sensors</h2><div class="dim">loading</div></section>
<section class="card live" id="lpeople"><h2>People it knows</h2><div class="dim">loading</div></section>
<section class="card wide" id="talk"><h2>Conversation</h2><div class="dim">loading</div></section>
</div>
</div>
<div class="tab" data-tab="people"><div class="grid">
<section class="card wide" id="people"><h2>People it knows</h2><div class="dim">loading</div></section>
<section class="card wide" id="pdetail" style="display:none"></section>
</div></div>
<div class="tab" data-tab="heard"><div class="grid">
<section class="card wide" id="heard"><h2>What it heard</h2><div class="dim">loading</div></section>
</div></div>
<div class="tab" data-tab="learning"><div class="grid">
<section class="card wide" id="learn"><h2>Learning</h2><div class="dim">loading</div></section>
</div></div>
<div class="tab" data-tab="system">
<div class="grid">
<section class="card" id="health"><h2>Health</h2><div class="dim">loading</div></section>
<section class="card" id="sensors"><h2>Sensors</h2><div class="dim">loading</div></section>
<section class="card" id="camera"><h2>Camera</h2>
<div class="camwrap"><img id="camimg" alt="live camera picture"><div id="camoff" class="dim">camera off</div></div>
<div class="camctl"><button id="cambtn" type="button">Start camera</button><button id="camrot" type="button">Rotate 180°</button><span id="camstat" class="dim"></span></div>
<div class="dim">640×480, ~10 fps. Runs only while shown here; stops ~3 s after Stop or closing the tab, so other programs (bringup.py, the robot) can use the camera.</div>
</section>
<section class="card" id="speaker"><h2>Speaker</h2>
<div class="kv"><b>level</b><span><input id="lvl" type="range" min="0" max="70" step="5" style="width:180px;vertical-align:middle"> <span id="lvlval">…</span></span></div>
<div class="dim">The robot's voice loudness (peak of the sound; the card has no other volume control). Used from the next sentence on. The family sleeps at night - keep it low then.</div>
</section>
<section class="card" id="network"><h2>Network</h2><div class="dim">loading</div></section>
<section class="card wide" id="gpio"><h2>GPIO header</h2><div class="dim">loading</div></section>
<section class="card wide" id="wiring"><h2>Wiring: what goes where</h2><div class="dim">loading</div></section>
<section class="card" id="devices"><h2>Devices</h2><div class="dim">loading</div></section>
<section class="card" id="processes"><h2>Processes</h2><div class="dim">loading</div></section>
<section class="card wide" id="logs"><h2>Logs <small class="dim">journalctl -n 30</small></h2><div class="dim">loading</div></section>
</div>
</div>
<script>
"use strict";
const $=id=>document.getElementById(id);
const ESC={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"};
function esc(v){return String(v==null?"":v).replace(/[&<>"']/g,c=>ESC[c]);}
function mib(b){return b==null?"?":(b/1048576).toFixed(b<104857600?1:0)+" MiB";}
function gib(b){return (b/1073741824).toFixed(1)+" GiB";}
function pct(v){return v==null?"…":v.toFixed(1)+"%";}
function dur(s){s=Math.floor(s);const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return (d?d+"d ":"")+h+"h "+m+"m";}
function errBox(x){return x&&x.error?'<div class="err">'+esc(x.error)+"</div>":"";}
function bar(p,cls){p=Math.max(0,Math.min(100,p||0));const c=p>90?"var(--bad)":p>70?"var(--warn)":"var(--ok)";return '<div class="bar"><i style="width:'+p.toFixed(0)+"%;background:"+c+'"></i></div>';}
function set(id,title,html){$(id).innerHTML="<h2>"+title+"</h2>"+html;}
function section(id,title,data,fn){try{set(id,title,data&&data.error?errBox(data):fn(data));}catch(e){set(id,title,'<div class="err">render error: '+esc(e.message)+"</div>");}}

function rHealth(h){
  const t=h.temp_c,tc=typeof t!=="number"?"err":t>=75?"bad":t>=65?"warn":"ok";
  let thr="";
  if(h.throttled&&h.throttled.flags){
    const any=h.throttled.flags.some(f=>f.now||f.since_boot);
    thr=any?h.throttled.flags.filter(f=>f.now||f.since_boot).map(f=>'<span class="tag '+(f.now?"on":"past")+'">'+esc(f.flag)+(f.now?" NOW":" since boot")+"</span>").join(""):'<span class="ok">none</span>';
    thr+=' <span class="dim">'+esc(h.throttled.raw)+"</span>";
  }else thr=errBox(h.throttled);
  const m=h.memory||{},d=h.disk||{};
  const freqs=Array.isArray(h.core_freq)?h.core_freq.map((f,i)=>esc(f.mhz)+" MHz"+(h.core_percent[i]!=null?' <span class="dim">'+pct(h.core_percent[i])+"</span>":"")).join(" · "):errBox(h.core_freq);
  const v=typeof h.core_volts==="number"?h.core_volts.toFixed(3)+" V":errBox(h.core_volts);
  return '<div class="kv">'+
    "<b>temp</b><span class=\""+tc+'">'+(typeof t==="number"?t.toFixed(1)+" °C":errBox(t))+"</span>"+
    "<b>power</b><span>"+thr+"</span>"+
    "<b>cpu</b><span>"+pct(h.cpu_percent)+bar(h.cpu_percent)+"</span>"+
    "<b>cores</b><span>"+freqs+"</span>"+
    "<b>core V</b><span>"+v+"</span>"+
    "<b>load</b><span>"+h.load.map(x=>x.toFixed(2)).join(" ")+"</span>"+
    "<b>uptime</b><span>"+dur(h.uptime_s)+"</span>"+
    "<b>memory</b><span>"+(m.error?errBox(m):mib(m.used)+" / "+mib(m.total)+(m.swap_used?' <span class="dim">swap '+mib(m.swap_used)+"</span>":"")+bar(100*m.used/m.total))+"</span>"+
    "<b>disk /</b><span>"+gib(d.used)+" / "+gib(d.total)+bar(100*d.used/d.total)+"</span>"+
    "</div>";
}

const GROUPS={motors:"motors",imu:"IMU power",pantilt:"I2C bus: IMU + pan-tilt",fan:"fan",distance:"distance",audio:"audio",power:"power/GND",free:"free",reserved:"avoid"};
function pinCell(p,left){
  const s=p.state;let st="";
  if(p.power)st='<span class="dim">'+esc(p.name)+"</span>";
  else if(s){
    const f=s.func==="alt"?(s.alt||"alt"):s.func==="input"?"in":s.func==="output"?"OUT":s.func;
    st='<span class="f">'+esc(f)+" "+esc(s.pull==="none"?"":s.pull)+'</span><span class="lv '+(s.level==="hi"?"hi":"lo")+'" title="'+esc(s.level)+'"></span>';
  }else st='<span class="dim">?</span>';
  const name=p.power?"":esc(p.name)+(p.boot_high?' <span class="warn" title="boots HIGH">▲boot</span>':"");
  return '<div class="pin '+(left?"l":"r")+'" style="--g:var(--'+esc(p.group)+')" title="pin '+esc(p.pin)+" · "+esc(p.name)+" · "+esc(p.label)+'">'+
    '<span class="n">'+esc(p.pin)+'</span><span class="t"><div>'+name+'</div><div class="lab dim">'+esc(p.label)+'</div></span><span class="s">'+st+"</span></div>";
}
function rGpio(g){
  let h="";
  if(g.error)h+='<div class="err">pinctrl: '+esc(g.error)+"</div>";
  for(const w of g.warnings)h+='<div class="bad">⚠ '+esc(w)+"</div>";
  h+='<div class="legend dim" style="margin-bottom:6px">'+Object.entries(GROUPS).map(([k,v])=>'<span style="--g:var(--'+k+')"><i></i>'+esc(v)+"</span>").join("")+
    ' <span><span class="lv hi"></span>high <span class="lv lo"></span>low</span> <span class="warn">▲boot = GPIO0–8 boot HIGH</span></div>';
  h+='<div class="hdr">';
  for(let i=0;i<40;i+=2)h+=pinCell(g.pins[i],true)+pinCell(g.pins[i+1],false);
  return h+'</div><div class="dim" style="margin-top:4px">Pin 1 top-left (3.3 V, nearest SD card). Left = odd pins (inner row), right = even pins (outer row).</div>';
}

function list(x,key,empty){
  if(!x)return '<span class="dim">…</span>';
  if(x.error)return errBox(x);
  const a=x[key]||[];
  return a.length?a.map(l=>"<div>"+esc(l)+"</div>").join(""):'<span class="warn">'+esc(empty)+"</span>";
}
function rDevices(d){
  const i=d.i2c;
  let i2c=i&&i.error?errBox(i):i?(i.devices.length?i.devices.map(x=>'<span class="tag">'+esc(x.addr)+(x.in_use_by_driver?" UU":"")+"</span>").join(""):'<span class="warn">no devices on bus 1</span>'):"…";
  return '<div class="kv">'+
    "<b>camera</b><span>"+list(d.camera,"cameras","no cameras detected")+"</span>"+
    "<b>sound</b><span>"+list(d.sound,"cards","no sound cards")+"</span>"+
    "<b>i2c-1</b><span>"+i2c+"</span>"+
    "<b>usb</b><span>"+list(d.usb,"devices","none")+"</span></div>";
}

function rNetwork(n){
  const ifs=Array.isArray(n.interfaces)?n.interfaces.filter(i=>i.addrs.length).map(i=>"<div>"+esc(i.name)+' <span class="dim">'+esc(i.state)+"</span> "+i.addrs.map(esc).join(" ")+"</div>").join(""):errBox(n.interfaces);
  let wifi;
  if(Array.isArray(n.wifi)){
    const s=n.ssid&&n.ssid.ssid!=null?esc(n.ssid.ssid||"(not connected)"):errBox(n.ssid);
    wifi=n.wifi.length?n.wifi.map(w=>{const c=w.level_dbm>=-60?"ok":w.level_dbm>=-72?"warn":"bad";return esc(w.iface)+" "+s+' <span class="'+c+'">'+esc(w.level_dbm)+" dBm</span> link "+esc(w.link)+"/70";}).join("<br>"):'<span class="dim">no wireless interface</span>';
  }else wifi=errBox(n.wifi);
  return '<div class="kv"><b>host</b><span>'+esc(n.hostname)+"</span><b>ip</b><span>"+ifs+"</span><b>wifi</b><span>"+wifi+"</span></div>";
}

function ptable(rows,key){
  return '<div class="scroll"><table><tr><th>pid</th><th>rss</th><th>cpu</th><th>command</th></tr>'+rows.map(p=>
    "<tr><td>"+esc(p.pid)+"</td><td"+(key==="rss"?' class="ok"':"")+">"+mib(p.rss)+"</td><td"+(key==="cpu"?' class="ok"':"")+">"+(p.cpu==null?"…":esc(p.cpu.toFixed(1)))+'</td><td class="cmd">'+esc(p.cmd)+"</td></tr>").join("")+"</table></div>";
}
function rProcesses(p){
  return '<div class="dim">'+esc(p.count)+" processes · dashboard pid "+esc(p.self.pid)+' RSS <span class="ok">'+mib(p.self.rss)+"</span></div>"+
    '<h2 style="margin-top:6px">by memory</h2>'+ptable(p.by_mem,"rss")+'<h2 style="margin-top:6px">by cpu (% of one core)</h2>'+ptable(p.by_cpu,"cpu");
}
let lvlBusy=false;
document.addEventListener("DOMContentLoaded",()=>{const r=$("lvl");if(!r)return;
  r.addEventListener("input",()=>{lvlBusy=true;$("lvlval").textContent=r.value+"%";});
  r.addEventListener("change",()=>{fetch("/api/speaker-level",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({level:+r.value})})
    .then(x=>x.json()).then(d=>{$("lvlval").textContent=d.error?"error: "+d.error:Math.round(d.level*100)+"% (saved)";}).finally(()=>{lvlBusy=false;});});});
function rLogs(l){return "<pre>"+esc(l.lines.join("\n"))+"</pre>";}
const TALK={heard:["heard","var(--audio)"],said:["said","var(--pantilt)"],dropped:["not sent - nobody looking","var(--dim)"],
voice:["voice","var(--fan)"],called:["called by name","var(--warn)"],not_for_me:["not for me - quiet","var(--dim)"],
session_end:["conversation over","var(--dim)"],met:["met","var(--ok)"],forgot:["forgot","var(--bad)"],
turn:["head turned to the voice","var(--motors)"],interrupted:["stopped - talked over","var(--warn)"],
own_echo:["its own words heard back - not answered","var(--dim)"],
showing:["shown something","var(--warn)"],shown:["looked at what was shown","var(--pantilt)"]};
const HOW={face:"looked at the face it saw",last_seen:"looked where it last saw them",sweep:"swept the room (ahead, left, right)"};
function rTalk(c){
  let h=c.ears?'<div class="kv"><b>ears</b><span>'+esc(c.ears.state)+' <span class="dim">('+esc(c.ears.at)+')</span></span></div>':
    '<div class="dim">alive.py is not running (no ears status)</div>';
  if(!c.items.length)return h+'<div class="dim">nothing heard yet</div>';
  h+='<div class="scroll"><table><tr><th>time</th><th>what</th><th>who</th><th>text</th></tr>';
  for(const e of c.items.slice().reverse()){
    const k=TALK[e.kind]||[e.kind,"var(--fg)"],tm=new Date(e.t*1000).toTimeString().slice(0,8);
    let who=e.kind==="said"?"Arcby"+(e.by?' <span class="dim">('+esc(e.by)+")</span>":""):esc(e.name||(e.kind==="voice"?"unknown":""));
    if(e.kind==="voice"&&e.sim!=null)who+=' <span class="dim">'+Number(e.sim).toFixed(2)+"</span>";
    let text=e.text!=null?esc(e.text):(e.seconds!=null?'<span class="dim">'+Number(e.seconds).toFixed(1)+" s of speech</span>":"");
    if(e.kind==="showing"||e.kind==="shown")text=esc(e.what||"something")+(e.asked?' <span class="dim">(asked)</span>':"")+
      (e.kind==="shown"?' <span class="dim">'+(e.picture?"picture sent":"no picture")+(e.answered?", answered":"")+"</span>":"");
        if(e.kind==="turn")text=esc(HOW[e.how]||e.how)+(e.pan!=null?' <span class="dim">(pan '+e.pan+", tilt "+e.tilt+")</span>":"")+(e.called?" - called by name":"");
    if(e.kind==="dropped")text+=' <span class="dim">- heard, not sent (nobody was looking)</span>';
    if(e.error)text+=' <span class="err">'+esc(e.error)+"</span>";
    h+="<tr><td>"+tm+'</td><td style="color:'+k[1]+'">'+k[0]+"</td><td>"+who+"</td><td>"+text+"</td></tr>";
  }
  return h+"</table></div>";
}

const DIST_STATUS={
  ok:["ok","measuring"],
  nothing:["warn","no echo: nothing in range, or the object is soft / at an angle"],
  noanswer:["bad","sensor not answering: check VCC, GND and TRIG (pin 29)"],
  busy:["warn","pins busy: another program uses GPIO5/6 (hwtest/distance.py?)"],
  idle:["dim","starting..."],
  robot:["dim","in use by the robot (alive.py pings it itself) - the dashboard does not ping meanwhile"]};
function spark(hist,period){
  const W=hist.length?Math.max(hist.length-1,1):1,vals=hist.filter(v=>v!=null);
  const top=Math.max(50,Math.ceil(Math.max(0,...vals)*1.2/50)*50);
  let lines="",run=[];
  const flush=()=>{if(run.length>1)lines+='<polyline points="'+run.join(" ")+'"/>';run=[];};
  hist.forEach((v,i)=>{if(v==null)flush();else run.push((i/W*100).toFixed(2)+","+(100-v/top*100).toFixed(2));});
  flush();
  return '<svg class="spark" viewBox="0 0 100 100" preserveAspectRatio="none">'+
    '<line x1="0" y1="50" x2="100" y2="50"/>'+lines+"</svg>"+
    '<div class="dim" style="display:flex;justify-content:space-between"><span>last '+Math.round(hist.length*period)+" s</span><span>scale 0–"+top+" cm (mid line "+top/2+" cm)</span></div>";
}
function rSensors(s){
  const d=s.distance||{};
  const known=DIST_STATUS[d.status];
  const st=known?known:[d.status&&d.status.startsWith("error")?"bad":"dim",d.status||"?"];
  const stale=d.status!=="ok"||d.age_s==null||d.age_s>1;
  const big=d.cm!=null?esc(d.cm.toFixed(1))+" <small>cm</small>":"–– <small>cm</small>";
  let h='<h2 style="color:var(--distance)">HC-SR04 distance</h2><div class="sensor">'+
    '<div class="big'+(stale?" stale":"")+'">'+big+"</div>"+
    '<div class="kv"><b>status</b><span class="'+st[0]+'">'+esc(st[1])+"</span>"+
    "<b>median</b><span>"+(d.median_cm!=null?esc(d.median_cm)+" cm":"–")+' <span class="dim">last 8</span></span>'+
    "<b>spread</b><span>"+(d.spread_cm!=null?esc(d.spread_cm)+" cm":"–")+"</span></div></div>";
  if(d.history)h+='<div style="margin-top:6px">'+spark(d.history,d.period_s)+"</div>";
  if(d.pins)h+='<div class="dim" style="margin-top:4px">'+esc(d.pins)+" · 4 pings/s while this page is open</div>";
  h+='<h2 style="margin-top:10px">MPU6050 IMU</h2><div class="dim">shown here once the robot software runs (not wired yet)</div>';
  return h;
}
const BLANK="data:image/gif;base64,R0lGODlhAQABAAAAACw=";
let camOn=false;
function camSet(on,why){
  camOn=on;const img=$("camimg");
  img.src=on?"camera.mjpg?t="+Date.now():BLANK;   // a new src closes the old stream
  img.style.display=on?"block":"none";
  $("camoff").style.display=on?"none":"block";
  $("camoff").textContent=why||"camera off";
  $("cambtn").textContent=on?"Stop camera":"Start camera";
}
$("cambtn").onclick=()=>camSet(!camOn);
$("camimg").onerror=()=>{if(camOn)camSet(false,"no picture - see the status next to the buttons");};
function camRot(on){$("camimg").classList.toggle("rot",on);try{localStorage.setItem("camrot",on?"1":"0");}catch(e){}}
$("camrot").onclick=()=>camRot(!$("camimg").classList.contains("rot"));
try{camRot(localStorage.getItem("camrot")==="1");}catch(e){}
function camStatus(c){
  if(!c)return;
  const cls=c.status==="live"?"ok":c.status==="error"||(c.status||"").startsWith("error")?"bad":"dim";
  let t=c.status;
  if(c.status==="live")t+=(c.fps?" · "+c.fps+" fps":"")+" · "+c.viewers+" viewer"+(c.viewers===1?"":"s")+(c.kib?" · "+c.kib+" KiB/frame":"");
  if(c.error)t+=": "+c.error;
  $("camstat").className=cls;$("camstat").textContent=t;
}
let stimer=null;
async function pollSensors(){
  clearTimeout(stimer);
  try{
    const r=await fetch("api/sensors",{cache:"no-store"});
    if(!r.ok)throw new Error("HTTP "+r.status);
    const s=await r.json();
    section("sensors","Sensors",s,rSensors);
    camStatus(s.camera);
  }catch(e){set("sensors","Sensors",'<div class="err">'+esc(e.message)+"</div>");}
  if(!document.hidden)stimer=setTimeout(pollSensors,500);
}

function rWiring(w){
  let h='<div class="dim" style="margin-bottom:6px">Module wire → Pi pin. Pin numbers are the physical header pins (see the GPIO card above). "breadboard col N" = the mini breadboard under the upper deck.</div><div class="wires">';
  for(const m of w.modules){
    h+='<div class="mod" style="--g:var(--'+esc(m.group)+')"><h3>'+esc(m.module)+'</h3><table><tr><th>wire</th><th>Pi pin</th><th>via</th></tr>';
    for(const r of m.wires){
      const pin=r.pin==null?'<td class="dim">–</td>':'<td class="p" title="'+esc(r.where)+'">'+esc(r.pin)+' <span class="dim">'+esc(r.name)+"</span></td>";
      h+="<tr><td>"+esc(r.wire)+"</td>"+pin+'<td class="dim">'+esc(r.via.replace(/^bb /,"breadboard col "))+"</td></tr>";
    }
    h+="</table></div>";
  }
  h+='</div><h2 style="margin-top:10px">Breadboard columns</h2><div class="scroll"><table><tr><th>col</th><th>half</th><th>net</th><th>from</th><th>to</th></tr>'+
    w.breadboard.map(b=>"<tr><td>"+esc(b.col)+"</td><td>"+esc(b.half)+"</td><td>"+esc(b.net)+"</td><td>"+esc(b.src)+'</td><td class="cmd">'+esc(b.dst)+"</td></tr>").join("")+
    '</table></div><div class="dim" style="margin-top:4px">Top half = 3.3 V side, bottom half = 4.7 V servo side - never bridge them.</div>';
  return h;
}

// ---- live state of the robot (alive.py -> state.json -> /api/live), every 0.5 s ----
const EYE='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>';
function num(v,d,u){return typeof v==="number"?v.toFixed(d)+(u||""):"–";}
function meter(p,col){p=Math.max(0,Math.min(100,p||0));return '<div class="bar"><i style="width:'+p.toFixed(0)+"%;background:"+col+'"></i></div>';}
function quota(today,limit){if(today==null)return '<span class="dim">–</span>';return esc(today)+(limit?' <span class="dim">/ '+esc(limit)+" today</span>"+bar(100*today/limit):' <span class="dim">today</span>');}
function side(pan){return typeof pan!=="number"?"":pan>8?"left":pan<-8?"right":"ahead";}
function pose(pan,tilt){return "pan "+num(pan,0,"°")+" · tilt "+num(tilt,0,"°");}
function faceLabel(f){
  if(f.picture)return ["photo","var(--dim)","a picture (it never moves)"];
  if(f.name)return [f.name,"var(--ok)",f.sim!=null?"match "+num(f.sim,2):""];
  if(f.unknown)return ["stranger","var(--warn)","someone it does not know"];
  return ["?","var(--fg)","not recognised yet"];
}
function rFaces(st){
  const fs=(st.faces||[]).slice().sort((a,b)=>(b.pan||0)-(a.pan||0));   // robot's left first, like looking through its eyes
  if(!fs.length)return '<div class="empty">nobody in view</div>';
  return '<div class="chips">'+fs.map(f=>{
    const [nm,col,hint]=faceLabel(f),eng=typeof f.engagement==="number"?f.engagement*100:0;
    const look=f.looking?'<span class="badge ok">'+EYE+"looking</span>":'<span class="dim" style="font-size:11px">not looking</span>';
    return '<div class="chip'+(f.target?" target":"")+'" style="--c:'+col+'">'+
      '<div class="nm"><span>'+esc(nm)+"</span>"+(f.target?'<span class="badge" style="color:var(--pantilt)">attention</span>':"")+"</div>"+
      '<div style="margin:2px 0">'+look+' <span class="dim">'+esc(hint)+"</span></div>"+
      '<div class="dim" style="font-size:11px">engagement '+eng.toFixed(0)+"%</div>"+meter(eng,"var(--pantilt)")+
      '<div class="dim" style="margin-top:3px">'+esc(side(f.pan))+" · "+pose(f.pan,f.tilt)+" · "+num(f.size_deg,1,"°")+" wide</div></div>";
  }).join("")+"</div>";
}
const MODES={idle:"looking around calmly",hold:"holding where the face was",search:"searching for a lost face",follow:"following a face",
  glance:"glancing at someone",listen:"listening",break:"short eye-contact break",orient:"turning to a voice",scan:"looking round for a voice"};
function rAttn(st){
  const a=st.attention||{},hd=st.head;
  let h='<div class="state" style="color:var(--pantilt)">'+esc(a.mode||"?")+' <small class="dim" style="font-weight:normal">'+esc(MODES[a.mode]||"")+"</small></div>";
  h+='<div class="kv"><b>on whom</b><span>'+(a.target_name?"<b style=\"color:var(--fg);font-weight:bold\">"+esc(a.target_name)+"</b>"+(a.target!=null?' <span class="dim">track '+esc(a.target)+"</span>":""):'<span class="dim">nobody</span>')+"</span>"+
    "<b>talk partner</b><span>"+(a.partner?esc(a.partner):'<span class="dim">–</span>')+"</span>"+
    "<b>gaze to</b><span>"+pose(a.pan,a.tilt)+"</span>"+
    "<b>head</b><span>"+(!hd?'<span class="dim">–</span>':hd.on?'<span class="ok">on</span> · '+pose(hd.pan,hd.tilt)+' <span class="dim">'+num(hd.dps,0," °/s")+"</span>":'<span class="warn">off</span> <span class="dim">(no battery / PTZ board not answering)</span>')+"</span></div>";
  return h;
}
function rConv(st){
  const c=st.conversation;
  if(!c)return '<div class="dim">no conversation module (alive.py runs without ears / voice)</div>';
  let h=c.with_?'<div class="state ok">talking with '+esc(c.with_)+' <small class="dim" style="font-weight:normal">'+num(c.open_s,0," s left")+"</small></div>":
    '<div class="state dim">no open conversation</div>';
  h+='<div class="kv"><b>waiting for</b><span>'+(c.waiting_for?'<span class="warn">'+esc(c.waiting_for)+"</span>":'<span class="dim">nothing</span>')+"</span>"+
    "<b>dropped</b><span>"+esc(c.dropped==null?"–":c.dropped)+' <span class="dim">sentences heard while nobody looked</span></span></div>';
  return h;
}
function rEars(st){
  const e=st.ears;
  if(!e)return '<div class="dim">ears off (alive.py runs without the microphone)</div>';
  const now=!e.running?["bad","stopped"]:e.muted?["warn","muted - the robot speaks"]:e.hearing?["ok","hearing someone speak"]:["dim","listening (quiet)"];
  const lv=typeof e.level_db==="number"?Math.max(0,Math.min(100,(e.level_db+90)/70*100)):0;
  return '<div class="state '+now[0]+'">'+now[1]+"</div>"+
    '<div class="kv"><b>level</b><span>'+num(e.level_db,1," dBFS")+' <span class="dim">gain ×'+num(e.gain,0)+"</span>"+meter(lv,"var(--audio)")+"</span>"+
    "<b>heard</b><span>"+esc(e.heard==null?"–":e.heard)+' <span class="dim">sentences · '+esc(e.barged||0)+" barge-ins</span></span>"+
    "<b>speech-to-text</b><span>"+esc(e.stt||"–")+"</span>"+
    "<b>live model</b><span>"+esc(e.live||"–")+"</span>"+
    "<b>STT requests</b><span>"+quota(e.stt_today)+"</span></div>";
}
function rVoice(st){
  const v=st.voice,m=st.mind;
  let h='<div class="sub">voice</div>';
  if(!v)h+='<div class="dim">voice off (--quiet or no key)</div>';
  else h+='<div class="state '+(v.speaking?"ok":v.busy?"warn":"dim")+'">'+(v.speaking?"speaking":v.busy?"busy (preparing a sentence)":"quiet")+"</div>"+
    '<div class="kv"><b>model</b><span>'+esc(v.model||"–")+' <span class="dim">voice '+esc(v.voice||"–")+"</span></span>"+
    "<b>requests</b><span>"+quota(v.today,v.limit)+"</span></div>";
  h+='<div class="sub" style="margin-top:8px">mind</div>';
  if(!m)h+='<div class="dim">mind off (rules only)</div>';
  else h+='<div class="kv"><b>model</b><span>'+esc(m.model||"–")+"</span><b>requests</b><span>"+quota(m.today,m.limit)+"</span></div>";
  return h;
}
function stCls(s){return !s?"dim":s==="ok"?"ok":/err|fail|lost|dead/.test(s)?"bad":"warn";}
function rSens(st){
  const S=st.sensors||{},status=S.status||{},sy=S.system,so=S.sonar,im=S.imu,cam=st.camera,rec=st.recording;
  const tag=k=>status[k]?' <span class="'+stCls(status[k])+'">'+esc(status[k])+"</span>":"";
  let h="";
  h+='<div class="sub">IMU'+tag("imu")+"</div>";
  h+=im?'<div class="kv"><b>roll / pitch</b><span>'+num(im.roll,1,"°")+" / "+num(im.pitch,1,"°")+"</span>"+
    "<b>yaw · tilt</b><span>"+num(im.yaw,1,"°")+" · "+num(im.tilt,1,"°")+"</span>"+
    "<b>tipped</b><span>"+(im.tipped?'<span class="bad">YES</span>':'<span class="ok">no</span>')+' <span class="dim">bump '+num(im.bump_g,3," g")+"</span></span>"+
    "<b>chip temp</b><span>"+num(im.temp_c,1," °C")+"</span></div>":'<div class="dim">no reading</div>';
  h+='<div class="sub">sonar'+tag("sonar")+"</div>";
  h+=so?'<div class="kv"><b>distance</b><span>'+(typeof so.cm==="number"?'<b style="color:var(--distance);font-weight:bold">'+so.cm.toFixed(1)+" cm</b>":'<span class="dim">'+esc(so.why||"no echo")+"</span>")+"</span></div>":'<div class="dim">no reading</div>';
  h+='<div class="sub">system'+tag("system")+"</div>";
  if(sy){const t=sy.cpu_c,tc=typeof t!=="number"?"dim":t>=75?"bad":t>=65?"warn":"ok";
    h+='<div class="kv"><b>CPU</b><span><span class="'+tc+'">'+num(t,1," °C")+'</span> <span class="dim">load '+num(sy.load,2)+"</span></span>"+
    "<b>free</b><span>"+num(sy.mem_free_mb,0," MB")+' RAM · '+num(sy.disk_free_gb,1," GB")+" disk</span>"+
    "<b>fan</b><span>"+(sy.fan==null?"–":sy.fan?'<span class="ok">on</span>':'<span class="dim">off</span>')+"</span>"+
    "<b>undervoltage</b><span>"+(sy.undervolt_now?'<span class="bad">NOW</span>':sy.undervolt_since_boot?'<span class="warn">since boot</span>':'<span class="ok">none</span>')+"</span></div>";
  }else h+='<div class="dim">no reading</div>';
  h+='<div class="sub">camera</div>';
  h+=cam?'<div class="kv"><b>speed</b><span>'+num(cam.fps,1," fps")+' <span class="dim">EV '+num(cam.ev,1)+"</span></span>"+
    "<b>face models</b><span>detect "+num(cam.detect_ms,0," ms")+" · embed "+num(cam.embed_ms,0," ms")+"</span></div>":'<div class="dim">–</div>';
  h+='<div class="sub">recording</div>';
  h+=rec?'<div class="kv"><b>state</b><span>'+(rec.failed?'<span class="bad">failed</span>':'<span class="ok">recording</span>')+"</span>"+
    '<b>to</b><span class="cmd" style="word-break:break-all">'+esc(String(rec.dir||"").split("/").slice(-2).join("/"))+"</span></div>":'<div class="dim">not recording</div>';
  return h;
}
function rPeople(st){
  const ps=st.people||[];
  if(!ps.length)return '<div class="dim">nobody yet</div>';
  const inView=new Set((st.faces||[]).map(f=>f.person).filter(Boolean));
  return '<div class="people">'+ps.map(p=>{
    const n=p.voice_prints,few=typeof n==="number"&&n<10;
    return '<div class="person'+(inView.has(p.id)?" seen":"")+'" title="'+esc(p.id)+'"><b>'+esc(p.name)+"</b>"+
      (p.call&&p.call!==p.name?' <span class="dim">"'+esc(p.call)+'"</span>':"")+
      '<div class="dim" style="font-size:11px">'+(n==null?"voice –":'voice prints <span class="'+(few?"warn":"ok")+'">'+esc(n)+"</span>"+(few?" - needs more":""))+
      (inView.has(p.id)?' · <span class="ok">in view</span>':"")+"</div></div>";
  }).join("")+'</div><div class="dim" style="margin-top:6px">Green border = in view now. Voice prints are learnt while the person is seen speaking (~10+ for a reliable voice ID).</div>';
}
const DRIVE_STATES={START:"learning the floor in front",SCAN:"looking around",CHOOSE:"choosing a way",TURN:"turning",
  DRIVE:"driving",CROSS:"crossing a floor edge",BACK:"backing up",APPROACH:"going to a person",GREET:"greeting",
  FOLLOW:"following",LOST:"lost them - looking around",DONE:"done"};
function rDrive(st){
  const d=st.drive;
  if(!d)return '<div class="dim">no drive state (older alive.py)</div>';
  if(d.mode==="off")return '<div class="state dim">no drive</div><div class="dim">'+esc(d.why||"")+"</div>";
  const on=d.mode!=="stopped";
  let h=on?'<div class="state ok">'+esc(d.mode)+' <small class="dim" style="font-weight:normal">'+esc(DRIVE_STATES[d.state]||d.state||"")+
      " · "+num(d.for_s,0," s")+(d.max_s?" of "+num(d.max_s,0," s"):"")+"</small></div>":
    '<div class="state dim">stopped</div>';
  h+='<div class="kv"><b>'+(on?"started by":"why")+"</b><span>"+esc(d.why||"–")+"</span>"+
    (on&&d.note?"<b>note</b><span>"+esc(d.note)+"</span>":"")+
    "<b>sonar</b><span>"+(d.sonar==null?"–":esc(d.sonar)+(d.sonar==="nothing"?"":" cm"))+"</span>"+
    "<b>safety</b><span>"+((d.safety||[]).length?'<span class="warn">'+esc(d.safety.join(", "))+"</span>":'<span class="ok">ok</span>')+"</span>"+
    (d.fault?'<b>fault</b><span class="bad">'+esc(d.fault)+"</span>":"")+"</div>";
  h+='<div style="display:flex;gap:8px;margin-top:8px"><button style="background:var(--bad);color:#fff;font-weight:bold;flex:1;padding:10px" '+
    'onclick="cmd({op:\'drive_stop\'},this)">STOP</button>'+
    '<button style="padding:10px" '+(on?"disabled ":"")+'onclick="cmd({op:\'drive_start\',mode:\'explore\'},this)">Explore '+
    '<span class="dim">(2 min)</span></button></div>';
  return h;
}
const LIVE=[["lfaces","Who it sees",rFaces],["lattn","Attention",rAttn],["ldrive","Drive",rDrive],["lconv","Conversation now",rConv],["lears","Ears",rEars],
  ["lvoice","Voice + mind",rVoice],["lsens","Robot sensors",rSens],["lpeople","People it knows",rPeople]];
function liveShow(on,msg,sub){
  for(const [id] of LIVE)$(id).style.display=on?"":"none";
  const b=$("livebanner");b.style.display=on?"none":"";
  if(!on)b.innerHTML=esc(msg)+(sub?"<small>"+esc(sub)+"</small>":"");
}
function rLive(d){
  if(d.error&&!d.state){liveShow(false,"Live state: error",d.error);return;}
  if(!d.state||d.stale){
    const age=d.age_s!=null?(d.age_s<120?Math.round(d.age_s)+" s":d.age_s<7200?Math.round(d.age_s/60)+" min":Math.round(d.age_s/3600)+" h"):null;
    const msg=!d.running?"Robot not running (alive.py)"+(age?" - last state "+age+" old":""):
      age?"State stale ("+age+" old) - alive.py runs but does not update":"alive.py runs - no live state yet (starting?)";
    liveShow(false,msg,d.why||"");return;
  }
  liveShow(true);
  for(const [id,title,fn] of LIVE)section(id,title,d.state,fn);
}
let ltimer=null;
async function pollLive(){
  clearTimeout(ltimer);
  try{
    const r=await fetch("api/live",{cache:"no-store"});
    if(!r.ok)throw new Error("HTTP "+r.status);
    rLive(await r.json());
  }catch(e){liveShow(false,"Live state unavailable","dashboard offline: "+e.message);}
  if(!document.hidden)ltimer=setTimeout(pollLive,500);
}

// ---- learning mode (code/learn.py on the wall adapter -> /api/learn), every 2 s ----
function dur(sec){if(sec==null)return "–";sec=Math.round(sec);if(sec<60)return sec+" s";const m=Math.floor(sec/60);
  if(m<60)return m+" min "+(sec%60)+" s";return Math.floor(m/60)+" h "+(m%60)+" min";}
function rLearn(d){
  const st=d.state;if(!st)return '<div class="dim">'+esc(d.why||"no learning state")+"</div>";
  let h="";
  if(st.running&&st.job){
    const p=st.total?100*st.done/st.total:0;
    h+='<div class="kv"><b>now</b><span>'+esc(st.title||st.job)+(st.step?' <span class="dim">('+esc(st.step)+")</span>":"")+"</span></div>"+
      meter(p,"var(--ok)")+'<div class="kv"><b>progress</b><span>'+esc(st.done)+" / "+esc(st.total)+" ("+p.toFixed(0)+'%)</span></div>'+
      '<div class="kv"><b>time left</b><span>'+(st.eta_s!=null?"~"+dur(st.eta_s):'<span class="dim">measuring</span>')+"</span></div>";
    if(st.queue&&st.queue.length)h+='<div class="kv"><b>then</b><span>'+st.queue.map(esc).join(", ")+"</span></div>";
  }else h+='<div class="kv"><b>now</b><span class="dim">not learning'+(st.why?" - "+esc(st.why):"")+"</span></div>";
  if(d.age_s!=null)h+='<div class="dim" style="font-size:11px">updated '+dur(d.age_s)+" ago</div>";
  const ps=st.proposals||[];
  if(ps.length){h+='<h2 style="margin-top:10px">Waiting for you</h2>';
    for(const p of ps)h+='<div class="kv"><b>'+esc(p.key)+"</b><span>"+esc(p.old)+" → <b>"+esc(p.new)+"</b> "+
      '<button onclick="learnDecide(\''+esc(p.id)+'\',true)">apply</button> <button onclick="learnDecide(\''+esc(p.id)+'\',false)">reject</button>'+
      '<br><span class="dim">'+esc(p.reason)+"</span></span></div>";}
  const rs=(st.results||[]).slice().reverse();
  if(rs.length){h+='<h2 style="margin-top:10px">Last results</h2>';
    for(const r of rs.slice(0,4)){
      h+='<div class="kv"><b>'+esc(r.title||r.job)+'</b><span class="dim">'+esc(r.how)+(r.t?" · "+new Date(r.t*1000).toLocaleString():"")+"</span></div>"+
        (r.findings||[]).map(f=>'<div class="dim" style="margin-left:12px">'+esc(f)+"</div>").join("")+
        (r.changes||[]).map(c=>'<div style="margin-left:12px">'+esc(c.key)+" → "+esc(c.new)+' <span class="dim">('+esc(c.how)+")</span></div>").join("");}
  }
  return h;
}
async function learnDecide(id,accept){
  try{const r=await fetch("api/learn-decide",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:id,accept:accept})});
    const d=await r.json();if(!r.ok)alertless("learning: "+(d.error||d.message||r.status));}catch(e){alertless("learning: "+e.message);}
  pollLearn();
}
function alertless(msg){const el=$("learn");if(el)el.insertAdjacentHTML("beforeend",'<div class="dim" style="color:var(--bad)">'+esc(msg)+"</div>");}
let lntimer=null;
async function pollLearn(){
  clearTimeout(lntimer);
  try{const r=await fetch("api/learn",{cache:"no-store"});if(!r.ok)throw new Error("HTTP "+r.status);
    section("learn","Learning",await r.json(),rLearn);}
  catch(e){section("learn","Learning",{why:"unavailable: "+e.message},rLearn);}
  if(!document.hidden)lntimer=setTimeout(pollLearn,2000);
}

// ---- tabs ----
let TAB="live";
function showTab(t){
  TAB=t;
  document.querySelectorAll(".tab").forEach(d=>d.classList.toggle("on",d.dataset.tab===t));
  document.querySelectorAll("#tabs button").forEach(b=>b.classList.toggle("on",b.dataset.t===t));
  try{localStorage.setItem("arcbyTab",t);}catch(e){}
  if(t==="people")loadPeople(true);
  if(t==="heard")loadHeard();
}
document.addEventListener("DOMContentLoaded",()=>{
  document.querySelectorAll("#tabs button").forEach(b=>b.addEventListener("click",()=>showTab(b.dataset.t)));
  let t="live";try{t=localStorage.getItem("arcbyTab")||"live";}catch(e){}
  showTab(["live","people","heard","learning","system"].includes(t)?t:"live");
});

// ---- People: who it knows, their recorded faces, corrections (robot/commands.py does them) ----
let PEOPLE=[],SEL=null,LOOKS=[],LOFF=0,OPEN_SEQ=0,TOAST_T=null;
function ago(t){if(!t)return "never";const s=Date.now()/1000-t;return s<90?Math.round(s)+" s ago":s<5400?Math.round(s/60)+" min ago":s<172800?Math.round(s/3600)+" h ago":Math.round(s/86400)+" days ago";}
function pname(p){return p.name||("no name · "+p.id);}
const SPIN='<span class="spin"></span>';
function toast(text,state){                       // state: "work" (spinner, stays) / "ok" / "bad"
  const el=$("toast");clearTimeout(TOAST_T);
  el.className="toast "+(state||"");el.innerHTML=(state==="work"?SPIN:state==="ok"?"✓ ":state==="bad"?"✗ ":"")+esc(text);
  el.style.display="block";
  if(state!=="work")TOAST_T=setTimeout(()=>{el.style.display="none";},state==="bad"?8000:3500);
}
async function cmd(c,el){                         // el: the button / select / picture tile that was used
  if(el){el.disabled=true;el.classList&&el.classList.add("busy");}
  toast("sending to the robot…","work");
  let ok=false;
  try{const r=await fetch("api/command",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(c)});
    const d=await r.json();ok=r.ok&&d.ok;toast(d.message||d.error||("HTTP "+r.status),ok?"ok":"bad");}
  catch(e){toast("failed: "+e.message,"bad");}
  if(el){el.disabled=false;el.classList&&el.classList.remove("busy");}
  return ok;
}
function arm(btn,label,fn){                      // a dangerous button needs a second click within 4 s
  if(btn.dataset.armed){delete btn.dataset.armed;btn.classList.remove("armed");fn();return;}
  btn.dataset.armed="1";btn.dataset.label=btn.textContent;btn.textContent=label;btn.classList.add("armed");
  setTimeout(()=>{if(btn.dataset.armed){delete btn.dataset.armed;btn.textContent=btn.dataset.label;btn.classList.remove("armed");}},4000);
}
async function loadPeople(reopen){
  try{const r=await fetch("api/people",{cache:"no-store"});const d=await r.json();
    PEOPLE=d.people||[];section("people","People it knows",d,rPeople);
    if(SEL&&reopen)openPerson(SEL,true);}catch(e){section("people","People it knows",{error:e.message},rPeople);}
}
function rPeople(d){
  let h='<div class="pgrid">';
  for(const p of PEOPLE){
    h+='<div class="pcard'+(SEL===p.id?" sel":"")+'" data-id="'+p.id+'" onclick="openPerson(\''+p.id+'\')">'+
      (p.photo?'<img loading="lazy" src="'+esc(p.photo)+'" alt="">':'<div class="noimg">no photo</div>')+
      '<div class="nm">'+esc(pname(p))+"</div>"+
      '<div class="dim">'+esc(p.looks)+" pictures · "+esc(p.vectors)+" face prints · "+esc(p.voices)+" voice</div>"+
      '<div class="dim">seen '+esc(ago(p.last_seen))+(p.picture?' · <span class="warn">photo on the wall</span>':"")+"</div></div>";
  }
  h+='<div class="pcard'+(SEL==="unknown"?" sel":"")+'" data-id="unknown" onclick="openPerson(\'unknown\')"><div class="noimg">?</div>'+
    '<div class="nm">Not recognised</div><div class="dim">'+esc(d.unknown_looks||0)+" pictures</div>"+
    '<div class="dim">faces it did not know - label them</div></div></div>';
  return h+'<div class="dim" style="margin-top:6px">Tap a person to see their pictures. A picture is theirs when the robot decided so, its best match was at least '+
    esc(d.match_sim)+", or you said so. Changes go to the robot and are saved there.</div>";
}
function peopleOptions(except){
  return PEOPLE.filter(p=>p.id!==except).map(p=>'<option value="'+p.id+'">'+esc(pname(p))+"</option>").join("");
}
async function openPerson(id,keep){
  const seq=++OPEN_SEQ;
  SEL=id;if(!keep)LOFF=0;
  document.querySelectorAll(".pcard").forEach(c=>c.classList.toggle("sel",c.dataset.id===id));
  const p=PEOPLE.find(x=>x.id===id),el=$("pdetail");
  el.style.display="block";
  if(!keep){                                     // at once: whose pictures are coming, and where (phones: below the cards)
    el.innerHTML="<h2>"+(p?esc(pname(p)):"Not recognised")+"</h2><div class=\"dim\">"+SPIN+" loading the pictures…</div>";
    el.scrollIntoView({behavior:"smooth",block:"start"});
  }
  try{const r=await fetch("api/people/looks?id="+encodeURIComponent(id)+"&offset=0&n="+(LOFF+60),{cache:"no-store"});
    const d=await r.json();if(seq!==OPEN_SEQ)return;     // tapped someone else meanwhile
    LOOKS=d.looks||[];el.innerHTML=rDetail(id,d);}
  catch(e){if(seq===OPEN_SEQ)el.innerHTML='<div class="err">'+esc(e.message)+"</div>";}
}
async function faceDone(tile,ok){                // a picture moved / deleted: it leaves this list, the counts follow
  if(!ok)return;
  tile.style.transition="opacity .3s";tile.style.opacity="0";setTimeout(()=>tile.remove(),300);
  loadPeople(false);
}
function rDetail(id,d){
  const p=PEOPLE.find(x=>x.id===id);
  let h="<h2>"+(p?esc(pname(p)):"Not recognised")+' <small class="dim">'+esc(d.total)+" pictures, newest first</small></h2>";
  if(p){
    h+='<div class="acts"><input id="pn" value="'+esc(p.name||"")+'" placeholder="name" maxlength="40"><button onclick="cmd({op:\'name\',id:\''+id+'\',name:$(\'pn\').value||null},this).then(ok=>ok&&loadPeople(false))">Save name</button>'+
      '<input id="pc" value="'+esc(p.call||"")+'" placeholder="how it calls them" maxlength="40"><button onclick="cmd({op:\'call\',id:\''+id+'\',call:$(\'pc\').value||null},this).then(ok=>ok&&loadPeople(false))">Save</button></div>'+
      '<div class="acts"><select id="pm"><option value="">merge into…</option>'+peopleOptions(id)+'</select>'+
      '<button onclick="if(!$(\'pm\').value){toast(\'choose who to merge into first\',\'bad\');return;}const k=$(\'pm\').value;arm(this,\'Click again: merge\',()=>cmd({op:\'merge\',keep:k,drop:\''+id+'\'},this).then(ok=>{if(ok){SEL=k;loadPeople(true);}}))">Merge</button>'+
      '<select id="pp"><option value="">photo on the wall…</option><option value="on">yes, a photo</option><option value="off">no, a real person</option><option value="auto">let the robot decide</option></select>'+
      '<button onclick="if($(\'pp\').value)cmd({op:\'picture\',id:\''+id+'\',state:$(\'pp\').value},this).then(ok=>ok&&loadPeople(false))">Set</button>'+
      '<button class="danger" onclick="arm(this,\'Click again: forget + delete '+esc(p.looks)+' pictures\',()=>cmd({op:\'forget\',id:\''+id+'\'},this).then(ok=>{if(ok){SEL=null;$(\'pdetail\').style.display=\'none\';loadPeople(false);}}))">Forget completely</button></div>';
  }
  h+='<div class="fgrid">';
  for(const lk of LOOKS){
    const big=lk.crop_url||lk.face_url,f=esc(JSON.stringify(lk.face));
    h+='<div class="face"><a href="'+esc(big)+'" target="_blank"><img loading="lazy" src="'+esc(lk.face_url)+'" alt=""></a>'+
      '<div class="cap">'+esc(new Date((lk.time||0)*1000).toLocaleString())+"</div>"+
      '<div class="cap">'+(lk.sim!=null?"match "+Number(lk.sim).toFixed(2):"")+(lk.width_px?" · "+Math.round(lk.width_px)+" px":"")+"</div>"+
      '<select onchange="const t=this.closest(\'.face\'),v=this.value;if(v)cmd({op:\'relabel_look\',face:'+f+',to:v},t).then(ok=>faceDone(t,ok))">'+
      '<option value="">this is…</option>'+peopleOptions(id)+'<option value="new">a new person</option></select>'+
      '<button class="danger" onclick="const t=this.closest(\'.face\');arm(this,\'Click again\',()=>cmd({op:\'delete_look\',face:'+f+'},t).then(ok=>faceDone(t,ok)))">Delete</button></div>';
  }
  h+="</div>";
  if(LOOKS.length<d.total)h+='<div class="acts"><button onclick="LOFF+=60;this.innerHTML=SPIN+\' loading…\';openPerson(\''+id+'\',true)">More ('+esc(d.total-LOOKS.length)+' left)</button></div>';
  return h;
}

// ---- Heard: the kept sentences ----
async function loadHeard(){
  try{const r=await fetch("api/heard",{cache:"no-store"});section("heard","What it heard",await r.json(),rHeard);}
  catch(e){section("heard","What it heard",{error:e.message},rHeard);}
}
function rHeard(d){
  const cs=d.clips||[];if(!cs.length)return '<div class="dim">'+esc(d.why||"nothing kept")+"</div>";
  let h='<div id="hmsg" class="dim"></div>';
  for(const c of cs)
    h+='<div class="clip"><span class="dim">'+esc(new Date(c.time*1000).toLocaleString())+"<br>"+esc(c.kind)+"</span>"+
      "<span>"+esc(c.text||"(no words)")+'<br><audio controls preload="none" src="'+esc(c.url)+'"></audio></span>'+
      '<button class="danger" onclick="arm(this,\'Click again\',()=>delClip(\''+esc(c.name)+'\'))">Delete</button></div>';
  return h+'<div class="dim" style="margin-top:6px">The newest 40 sentences are kept (local only). "dropped" = heard but not for the robot (nobody looking).</div>';
}
async function delClip(name){
  try{const r=await fetch("api/heard-delete",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:name})});
    const d=await r.json();loadHeard();}catch(e){}
}

// ---- alarms: on top of every tab (power, heat, disk, the USB stick, the robot program) ----
let lastOk=0;
function rAlerts(list){
  const el=$("alerts");
  if(!list.length){el.style.display="none";el.innerHTML="";return;}
  const ic={danger:"⚠",warn:"⚠",info:"ⓘ"};
  el.innerHTML=list.map(a=>'<div class="alert '+esc(a.level)+'"><span class="ic">'+(ic[a.level]||"⚠")+"</span><span><b>"+esc(a.text)+"</b>"+
    (a.detail?"<small>"+esc(a.detail)+"</small>":"")+"</span></div>").join("");
  el.style.display="block";
}
let timer=null;
async function poll(){
  clearTimeout(timer);
  try{
    const r=await fetch("api/status",{cache:"no-store"});
    if(!r.ok)throw new Error("HTTP "+r.status);
    const s=await r.json();
    $("conn").className="ok";$("conn").textContent="live · "+new Date(s.time*1000).toLocaleTimeString();
    if(s.network&&s.network.hostname)$("host").textContent=s.network.hostname;
    section("health","Health",s.health,rHealth);
    section("gpio","GPIO header",s.gpio,rGpio);
    section("wiring","Wiring: what goes where",s.wiring,rWiring);
    section("devices","Devices",s.devices,rDevices);
    section("network","Network",s.network,rNetwork);
    section("processes","Processes",s.processes,rProcesses);
    section("talk","Conversation",s.talk,rTalk);
    if(s.speaker&&!s.speaker.error&&!lvlBusy){const v=s.speaker.level;$("lvlval").textContent=v==null?"not set (robot default 50%)":Math.round(v*100)+"%";if(v!=null)$("lvl").value=Math.round(v*100);}
    section("logs","Logs",s.logs,rLogs);
    lastOk=Date.now();rAlerts(Array.isArray(s.alerts)?s.alerts:[]);
  }catch(e){$("conn").className="bad";$("conn").textContent="offline: "+e.message;
    if(Date.now()-lastOk>6000)rAlerts([{level:"danger",text:"THE PI DOES NOT ANSWER"+(lastOk?" since "+new Date(lastOk).toLocaleTimeString():""),
      detail:"Power off or battery empty? (that is how it went on 2026-11-02) - or the Wi-Fi is down. Check its red LED."}]);}
  if(!document.hidden)timer=setTimeout(poll,2000);
}
document.addEventListener("visibilitychange",()=>{if(document.hidden){if(camOn)camSet(false,"stopped while the tab was hidden");}else{poll();pollSensors();pollLive();pollLearn();}});
poll();
pollSensors();
pollLive();
pollLearn();
</script></body></html>
"""
PAGE_BYTES = PAGE.encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "ArcbyDashboard"
    sys_version = ""

    def _send(self, code, ctype, body, head=False, cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _route(self, head):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE_BYTES, head)
        elif path == "/api/status":
            try:
                body = status_json()
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self._send(200, "application/json", body, head)
        elif path == "/api/live":
            try:
                body = live_json()
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": f"{type(e).__name__}: {e}", "state": None, "stale": True}).encode()
            self._send(200, "application/json", body, head)
        elif path == "/api/learn":
            try:
                body = json.dumps(learn_state(), separators=(",", ":")).encode()
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": f"{type(e).__name__}: {e}", "state": None}).encode()
            self._send(200, "application/json", body, head)
        elif path == "/api/sensors":
            try:
                body = sensors_json()
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self._send(200, "application/json", body, head)
        elif path == "/api/people":
            self._json(lambda: archive.people(), head)
        elif path == "/api/people/looks":
            q = parse_qs(urlparse(self.path).query)
            pid = (q.get("id") or ["unknown"])[0]
            try:
                offset, n = max(0, int((q.get("offset") or ["0"])[0])), min(200, int((q.get("n") or ["60"])[0]))
            except ValueError:
                offset, n = 0, 60
            if pid != "unknown" and not archive.PID.match(pid):
                self._send(400, "application/json", b'{"error": "id: a person id or unknown"}', head)
            else:
                self._json(lambda: archive.looks_of(pid, offset, n), head)
        elif path == "/api/heard":
            self._json(lambda: archive.heard(), head)
        elif path.startswith("/media/"):
            found = archive.media_file(path)
            if found is None:
                self._send(404, "text/plain; charset=utf-8", b"not found\n", head)
            else:
                # a recorded picture / clip never changes under its name (deleted ones 404): cache it a week
                self._send(200, found[1], found[0].read_bytes(), head, cache="private, max-age=604800, immutable")
        elif path == "/camera.mjpg" and not head:
            self._stream_camera()
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n", head)

    def _json(self, fn, head):
        try:
            body = json.dumps(fn(), separators=(",", ":"), ensure_ascii=False).encode()
        except Exception as e:  # noqa: BLE001 - one broken file must not break the page
            body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
        self._send(200, "application/json; charset=utf-8", body, head)

    def _stream_camera(self):
        """MJPEG stream (multipart/x-mixed-replace): the browser shows it in a plain <img>."""
        if _camera is None:
            self._send(503, "text/plain; charset=utf-8", b"camera turned off (ARCBY_DASHBOARD_CAMERA=0)\n")
            return
        _camera.open_viewer()
        try:
            got = _camera.wait_frame(0 if _camera.frame is None else _camera.seq - 1, CAM_FIRST_FRAME_S)
            if got is None:
                snap = _camera.snapshot()
                msg = snap["error"] or "no picture from the camera"
                self._send(503, "text/plain; charset=utf-8", f"{msg}\n".encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            while got is not None:
                seq, jpg = got
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                self.wfile.flush()
                got = _camera.wait_frame(seq, 5.0)
        except (BrokenPipeError, ConnectionResetError):
            pass                                # browser closed the view
        finally:
            _camera.close_viewer()

    def do_GET(self):
        self._route(False)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/speaker-level", "/api/learn-decide", "/api/command", "/api/heard-delete"):
            self._send(404, "text/plain", b"not found")
            return
        # JSON only: another web page cannot send JSON here without a CORS preflight, which this server refuses -
        # so only the dashboard's own page can change the robot's loudness
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            self._send(415, "application/json", b'{"error": "send application/json"}')
            return
        if path in ("/api/command", "/api/heard-delete"):
            try:
                raw = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 4096))
                res = archive.command(raw, archive.alive_running) if path == "/api/command" else archive.heard_delete(raw)
                self._send(200 if res.get("ok") else 409, "application/json; charset=utf-8",
                           json.dumps(res, ensure_ascii=False).encode())
            except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as e:
                self._send(400, "application/json", json.dumps({"error": str(e)}).encode())
            return
        if path == "/api/learn-decide":
            try:
                n = min(int(self.headers.get("Content-Length") or 0), 1024)
                res = learn_decide(self.rfile.read(n))
                self._send(200 if res["ok"] else 409, "application/json", json.dumps(res).encode())
            except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as e:
                self._send(400, "application/json", json.dumps({"error": str(e)}).encode())
            return
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 1024)
            body = json.dumps(set_speaker_level(self.rfile.read(n))).encode()
            code = 200
        except (ValueError, KeyError, TypeError, OSError) as e:
            body, code = json.dumps({"error": str(e)}).encode(), 400
        global _status
        with _status_lock:
            _status = (0.0, b"")                # the next poll shows the new level at once
        self._send(code, "application/json", body)

    def do_HEAD(self):
        self._route(True)

    def log_message(self, fmt, *args):
        pass  # no access log: keeps dashboard.log small


def main():
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"arcby-dashboard: cannot listen on {HOST}:{PORT}: {e}", file=sys.stderr)
        sys.exit(1)
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    PID_FILE.write_text(str(os.getpid()))
    cpu_usage()  # first CPU sample so the first poll already has a delta
    if _distance is not None:
        threading.Thread(target=_distance.run, name="distance", daemon=True).start()
    print(f"arcby-dashboard: pid {os.getpid()} listening on http://{HOST}:{PORT}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            if PID_FILE.read_text().strip() == str(os.getpid()):
                PID_FILE.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
