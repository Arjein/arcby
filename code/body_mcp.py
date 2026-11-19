#!/usr/bin/env python3
"""
Arcby "body" for Claude: an MCP server (stdio, JSON-RPC, stdlib only) that lets
an agent drive, look and see through the robot core (robot/: motors, IMU, safety).

  python3 code/body_mcp.py           # started by an MCP client, e.g. Claude Code with this .mcp.json:
                                     #   {"mcpServers": {"arcby-body": {"command": "python3",
                                     #     "args": ["<repo>/code/body_mcp.py"]}}}
  python3 code/body_mcp.py --sim     # simulated robot (hwtest/base.py SimRobot)

Tools (short text answers to keep tokens low):
  status        pose estimate, heading, tilt, front distance, head, battery, safety
  move cm       straight with IMU heading hold; + forward (max 100), - back (max 30);
                a forward move stops early when the front sensor sees < 20 cm
  turn deg      on the spot with the IMU, + = left
  look pan tilt point the camera head (deg, + = left / up)
  see           small JPEG from the camera
  stop          motors off now
  reset         clear a latched safety stop (bump / tipped / low battery)

The hardware starts on the first tool call (keep the robot still ~2 s for the
gyro calibration). All safety rules of robot/ apply: safety stops, the 0.3 s
motor watchdog, low battery. When Claude Code quits, stdin closes and the
motors stop. Only one program can own the motors: not together with drive_web.py.
No wheel encoders: distances are time x speed (carpet calibration), so the pose
is an estimate. Log: ~/scratch/body.log
"""
import argparse
import base64
import io
import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from robot import config, imu                 # noqa: E402
from robot.motors import to_pwm               # noqa: E402

LOG = Path.home() / "scratch" / "body.log"
log = logging.getLogger("arcby")
PROTOCOL = "2025-06-18"


class Blocked(Exception):
    """A command that cannot run now (safety stop, obstacle); the text goes back to the agent."""


class Body:
    """Agent-level actions on a started Robot. sonar / head / camera may be None (not fitted / sim)."""

    def __init__(self, robot, sonar=None, head=None, camera=None, clock=time.monotonic, sleep=time.sleep):
        self.robot, self.sonar, self.head, self.camera = robot, sonar, head, camera
        self.clock, self.sleep = clock, sleep
        self.x = self.y = 0.0               # cm, start frame: x forward, y left
        self.lead = config.BODY_TURN_LEAD_S  # learned from each turn's overshoot

    # ---- readings ----
    def _imu(self):
        return self.robot.state.get("imu", max_age=0.5)

    def _safety(self):
        return self.robot.state.get("safety") or {"stop": True, "reasons": ["starting"], "latched": []}

    def front_cm(self):
        return None if self.sonar is None else self.sonar.measure()

    def status(self, front=True):
        r = self._imu()
        s = self._safety()
        p = self.robot.state.get("power") or {}
        parts = [f"pos x{self.x:+.0f} y{self.y:+.0f}cm"]
        parts.append("hdg ?" if r is None else f"hdg {r.yaw_deg:+.0f}° tilt {r.tilt_deg:.0f}°")
        if front:
            f = self.front_cm()
            parts.append(f"front {'?' if f is None else f'{f:.0f}cm'}")
        if self.head is not None and self.head.pan is not None:
            parts.append(f"head {self.head.pan:+.0f}/{self.head.tilt:+.0f}")
        parts.append("batt " + ("?" if not p.get("ok") else "LOW" if p.get("under_voltage") else "ok"))
        if self.robot.fault:
            parts.append(f"FAULT {self.robot.fault}")
        parts.append("safe" if not s["stop"] else "STOP " + ",".join(s["reasons"]))
        return " | ".join(parts)

    def _ready(self, timeout=4.0):
        """Wait for the safety verdict to clear (IMU calibrating); latched stops need reset()."""
        end = self.clock() + timeout
        while True:
            s = self._safety()
            if not s["stop"]:
                return
            if s["latched"] or self.clock() > end or not self.robot.alive:
                raise Blocked(f"blocked: {', '.join(s['reasons'])}"
                              + (" (call reset if it is safe)" if s["latched"] else ""))
            self.sleep(0.05)

    def _halt(self, settle_s):
        """Stop and wait until the robot stops turning (max settle_s)."""
        m = self.robot.motors
        m.request_velocity(0.0, 0.0, "body")
        end = self.clock() + settle_s
        quiet = 0
        while self.clock() < end:
            m.request_velocity(0.0, 0.0, "body")
            r = self._imu()
            quiet = quiet + 1 if r is not None and abs(r.yaw_rate_dps) < 3.0 else 0
            if quiet >= 5:
                break
            self.sleep(0.02)
        m.stop("body")

    # ---- actions ----
    def move(self, cm):
        cm = float(cm)
        if cm != cm:
            raise ValueError("cm must be a number")
        cm = max(-config.BODY_MAX_BACK_CM, min(config.BODY_MAX_CM, cm))
        if abs(cm) < 2:
            return "moved 0cm | " + self.status()
        self._ready()
        sign = 1.0 if cm > 0 else -1.0
        if sign > 0:
            f = self.front_cm()
            if f is not None and f < config.BODY_STOP_CM:
                raise Blocked(f"blocked: front {f:.0f}cm")
        v_cm = config.BODY_CM_PER_S
        ramp = to_pwm(config.BODY_SPEED) / config.DRIVE_ACCEL
        duration = abs(cm) / v_cm + ramp / 2
        why, close = "", 0
        m = self.robot.motors
        t0 = last = self.clock()
        next_ping = t0
        try:
            while True:
                now = self.clock()
                el = now - t0
                if el >= duration:
                    break
                m.request_velocity(sign * config.BODY_SPEED, 0.0, "body")
                s = self._safety()
                if s["stop"]:
                    why = " STOPPED " + ",".join(s["reasons"])
                    break
                r = self._imu()
                if r is not None:
                    v = v_cm * min(1.0, el / ramp) * sign
                    h = math.radians(r.yaw_deg)
                    self.x += v * math.cos(h) * (now - last)
                    self.y += v * math.sin(h) * (now - last)
                last = now
                if sign > 0 and self.sonar is not None and now >= next_ping:
                    p = self.sonar.ping()
                    close = close + 1 if isinstance(p, float) and p < config.BODY_STOP_CM else 0
                    next_ping = now + 0.08
                    if close >= 2:
                        why = f" obstacle {p:.0f}cm"
                        break
                self.sleep(0.03)
        finally:
            self._halt(0.6)
        el = self.clock() - t0
        done = sign * max(0.0, min(abs(cm), v_cm * (min(el, duration) - ramp / 2)))
        log.info("body move %.0f -> ~%.0f cm%s", cm, done, why)
        return f"moved ~{done:+.0f}cm{why} | " + self.status()

    def turn(self, deg):
        deg = float(deg)
        if deg != deg:
            raise ValueError("deg must be a number")
        deg = max(-180.0, min(180.0, deg))
        if abs(deg) < 3:
            return "turned 0° | " + self.status(front=False)
        self._ready()
        r0 = self._imu()
        if r0 is None:
            raise Blocked("blocked: no IMU reading")
        start, sign = r0.yaw_deg, (1.0 if deg > 0 else -1.0)
        target = start + deg
        m = self.robot.motors
        t0 = self.clock()
        limit = abs(deg) / 100.0 + 2.0
        why, rate_at_stop = "", config.BODY_TURN_DPS
        try:
            while True:
                r = self._imu()
                s = self._safety()
                if s["stop"]:
                    why = " STOPPED " + ",".join(s["reasons"])
                    break
                if self.clock() - t0 > limit:
                    why = " timeout"
                    break
                if r is not None:
                    left = (target - r.yaw_deg) * sign
                    rate = abs(r.yaw_rate_dps)
                    if left <= rate * self.lead:
                        rate_at_stop = max(rate, 50.0)
                        break
                m.request_velocity(0.0, sign * config.BODY_TURN_DPS, "body")
                self.sleep(0.02)
        finally:
            self._halt(1.5)
        r = self._imu()
        final = start if r is None else r.yaw_deg
        over = (final - target) * sign                  # + = turned too far
        if not why:
            self.lead = max(0.0, min(0.4, self.lead + 0.5 * over / rate_at_stop))
        log.info("body turn %+.0f -> %+.1f (over %+.1f, lead now %.3f)%s",
                 deg, final - start, over, self.lead, why)
        return f"turned {final - start:+.0f}° (asked {deg:+.0f}){why} | " + self.status(front=False)

    def look(self, pan, tilt):
        if self.head is None:
            raise Blocked("no pan-tilt head")
        pan, tilt = self.head.move(pan, tilt)
        return f"head pan {pan:+.0f}° tilt {tilt:+.0f}°"

    def see(self, width=480):
        if self.camera is None:
            raise Blocked("no camera")
        width = int(max(160, min(1280, width)))
        img = self.camera.capture_image("main")
        img = img.convert("RGB").resize((width, round(width * img.height / img.width)))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        return buf.getvalue()

    def stop(self):
        self.robot.motors.stop("body")
        return "stopped | " + self.status(front=False)

    def reset(self):
        self.robot.safety.reset()
        self.sleep(0.1)
        return "reset | " + self.status(front=False)


# ---------- MCP (JSON-RPC 2.0 over stdio, one message per line) ----------

def _num(desc):
    return {"type": "number", "description": desc}


def _head_range(centre, sign, servo):
    """Head-angle range (lo, hi) reachable inside the servo limits."""
    a, b = ((v - centre) * sign for v in servo)
    return min(a, b), max(a, b)


_PAN = _head_range(config.PTZ_PAN_CENTRE, config.PTZ_PAN_SIGN, config.PTZ_PAN_SERVO)
_TILT = _head_range(config.PTZ_TILT_CENTRE, config.PTZ_TILT_SIGN, config.PTZ_TILT_SERVO)


TOOLS = [
    {"name": "status", "description": "Pose estimate (cm from start, x fwd y left), heading (+left), "
     "tilt, front distance, head, battery, safety.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "move", "description": "Drive straight holding heading. +fwd max 100, -back max 30 (blind). "
     "Stops early at <20cm obstacle. Distance is time-based (+-15%).",
     "inputSchema": {"type": "object", "properties": {"cm": _num("cm")}, "required": ["cm"]}},
    {"name": "turn", "description": "Turn on the spot, deg, +left -right, max 180. Accuracy a few deg; "
     "for fine aim use look.",
     "inputSchema": {"type": "object", "properties": {"deg": _num("deg")}, "required": ["deg"]}},
    {"name": "look", "description": f"Point camera head. pan +left -right ({_PAN[0]:.0f}..{_PAN[1]:.0f}), "
     f"tilt +up -down ({_TILT[0]:.0f}..{_TILT[1]:.0f}). 0/0 = centre (tilt 0 looks down a bit).",
     "inputSchema": {"type": "object", "properties": {"pan": _num("deg"), "tilt": _num("deg")},
                     "required": ["pan", "tilt"]}},
    {"name": "see", "description": "Photo from the head camera (wide lens). width px, default 480; "
     "bigger costs more tokens.",
     "inputSchema": {"type": "object", "properties": {"width": _num("px")}}},
    {"name": "stop", "description": "Motors off now.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "reset", "description": "Clear a latched safety stop (bump/tipped/low battery). Only if safe.",
     "inputSchema": {"type": "object", "properties": {}}},
]

INSTRUCTIONS = ("Arcby robot body (2-wheel, ~20x15cm, camera head ~20cm high, front sonar ~7cm high: "
                "low or thin obstacles may be missed). Look with see() before moving into unknown space. "
                "Keep moves short. If a reply says STOP or blocked, find out why before reset.")


class Server:
    """Tool dispatch; the hardware starts on the first call (start_body() -> Body)."""

    def __init__(self, start_body):
        self._start_body = start_body
        self.body = None

    def call(self, name, args):
        """-> (content list, is_error)"""
        if name not in {t["name"] for t in TOOLS}:
            return [{"type": "text", "text": f"unknown tool {name}"}], True
        try:
            if self.body is None:
                self.body = self._start_body()
            b = self.body
            if name == "see":
                jpg = b.see(args.get("width", 480))
                return [{"type": "image", "data": base64.b64encode(jpg).decode(), "mimeType": "image/jpeg"},
                        {"type": "text", "text": b.status(front=False)}], False
            text = {
                "status": lambda: b.status(),
                "move": lambda: b.move(args["cm"]),
                "turn": lambda: b.turn(args["deg"]),
                "look": lambda: b.look(args["pan"], args["tilt"]),
                "stop": lambda: b.stop(),
                "reset": lambda: b.reset(),
            }[name]()
            return [{"type": "text", "text": text}], False
        except Blocked as e:
            return [{"type": "text", "text": str(e)}], True
        except (KeyError, TypeError, ValueError) as e:
            return [{"type": "text", "text": f"bad arguments: {e}"}], True
        except Exception as e:  # noqa: BLE001 - report, keep serving (motors stop on their watchdog)
            log.exception("tool %s failed", name)
            return [{"type": "text", "text": f"error: {type(e).__name__}: {e}"}], True

    def handle(self, msg):
        """One JSON-RPC message -> reply dict, or None for notifications."""
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:
            return None
        params = msg.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", PROTOCOL),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "arcby-body", "version": "1.0"},
                      "instructions": INSTRUCTIONS}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            content, err = self.call(params.get("name"), params.get("arguments") or {})
            result = {"content": content, "isError": err}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, inp, out):
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                reply = self.handle(msg) if isinstance(msg, dict) else \
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "not an object"}}
            except json.JSONDecodeError as e:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            if reply is not None:
                out.write(json.dumps(reply) + "\n")
                out.flush()


# ---------- hardware ----------

class Hardware:
    """Starts the real (or simulated) robot and its add-ons; close() stops everything."""

    def __init__(self, sim=False):
        self.sim = sim
        self.robot = self.sonar = self.head = self.camera = None

    def start(self):
        from robot.main import Robot
        driver = dev = power = None
        if self.sim:
            sys.path.insert(0, str(CODE / "hwtest"))
            from base import SimDrive, SimRobot
            sim = SimRobot()
            driver, dev = SimDrive(sim, 1.0), sim.imu_device(imu.parse_axes(config.IMU_AXES))
            power = lambda: (False, False, "sim")          # noqa: E731 - the simulation has no battery
        else:
            from robot.power import require_no_alive
            require_no_alive()          # alive.py owns the IMU, head and sonar: stop it first
        self.robot = Robot(driver=driver, imu_dev=dev, power_reader=power)
        self.robot.__enter__()
        end = time.monotonic() + config.IMU_CALIB_S + 3.0
        while self.robot.state.get("imu", max_age=0.5) is None and time.monotonic() < end:
            time.sleep(0.05)            # gyro calibration: the first answer should not say "calibrating"
        if not self.sim:
            self.sonar = self._optional("sonar", self._sonar)
            self.head = self._optional("pan-tilt", self._head)
            self.camera = self._optional("camera", self._camera)
        log.info("body started (sim=%s)", self.sim)
        return Body(self.robot, self.sonar, self.head, self.camera)

    @staticmethod
    def _optional(what, make):
        try:
            return make()
        except Exception as e:  # noqa: BLE001 - drive without it, but say so in the log
            log.error("%s not available: %s", what, e)
            return None

    @staticmethod
    def _sonar():
        from robot.sonar import Sonar
        return Sonar()

    @staticmethod
    def _head():
        from robot.i2c import I2CDevice
        from robot.pan_tilt import PanTilt
        head = PanTilt(I2CDevice(config.I2C_BUS, config.PTZ_ADDR))
        head.init()
        head.set(0, 0)
        return head

    @staticmethod
    def _camera():
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (1152, 648)}))
        cam.start()
        try:
            cam.set_controls({"AfMode": 2})     # continuous autofocus (Camera Module 3)
        except Exception:  # noqa: BLE001 - fixed focus is fine too
            pass
        return cam

    def close(self):
        for what, part in (("camera", self.camera), ("sonar", self.sonar), ("pan-tilt", self.head)):
            if part is not None:
                try:
                    part.close()
                except Exception as e:  # noqa: BLE001
                    log.error("closing %s: %s", what, e)
        if self.robot is not None:
            self.robot.__exit__(None, None, None)
        log.info("body stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="simulated robot, no hardware")
    args = ap.parse_args()

    # JSON-RPC owns stdout: keep a private copy and send fd 1 (prints from libraries) to stderr
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    hw = Hardware(sim=args.sim)
    try:
        Server(hw.start).serve(sys.stdin, out)
    except KeyboardInterrupt:
        pass
    finally:
        hw.close()


if __name__ == "__main__":
    main()
