#!/usr/bin/env python3
"""
Real-time drive from a web browser (Mac or phone), car style.

  python3 code/drive_web.py          # the real robot
  python3 code/drive_web.py --sim    # simulated robot, nothing moves

It prints a link with a secret key, e.g. http://raspberrypi.local:8090/?k=ab12cd -
open it on the Mac. Keys (several at once, like a car):

  W / Up      gas             S / Down    brake, then reverse
  A / Left    steer left      D / Right   steer right   (standing still: turn on the spot)
  Space       stop now        R           reset a safety stop (bump / tipped)
  B           save the last 3 s of IMU + motor data (~/scratch/blackbox/) for tuning

The page sends the pressed keys 20 times a second. The robot stops when you let
go, when the tab loses focus, when Wi-Fi drops (no message for 0.3 s), on a
safety stop (tipped, bump, IMU lost, low battery), and when this program exits or crashes.
A / D ask for a turn RATE (deg/s, slider) that the gyro keeps exact; straight
driving holds the heading with the IMU; every start is a soft start.

Log: ~/scratch/drive.log
"""
import argparse
import json
import logging
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from robot import config, imu                 # noqa: E402
from robot.teleop import car_velocity         # noqa: E402

PORT = 8090
LOG = Path.home() / "scratch" / "drive.log"
log = logging.getLogger("arcby")


class DriveServer:
    """HTTP server that turns key states into motors.request() calls."""

    def __init__(self, robot, token, host="0.0.0.0", port=PORT):
        self.robot = robot
        self.token = token
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.last_name = None

    def serve_in_thread(self):
        th = threading.Thread(target=self.httpd.serve_forever, name="web", daemon=True)
        th.start()
        return th

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- logic (tested without HTTP) ----

    def drive(self, msg):
        """msg = {throttle, steer, speed, turn}; returns the telemetry dict."""
        def num(key, lo, hi, default):
            v = msg.get(key, default)
            if not isinstance(v, (int, float)) or v != v:      # also rejects NaN
                raise ValueError(f"{key} must be a number")
            return max(lo, min(hi, float(v)))
        throttle = num("throttle", -1.0, 1.0, 0.0)
        steer = num("steer", -1.0, 1.0, 0.0)
        speed = num("speed", 0.1, 1.0, config.KEY_SPEED)
        turn = num("turn", 10.0, config.TURN_MAX_DPS, config.KEY_TURN_DPS)
        v, rate = car_velocity(throttle, steer, speed, turn)
        self.robot.motors.request_velocity(v, rate, "web")
        name = ("stop" if (throttle, steer) == (0.0, 0.0) else
                f"throttle {throttle:+.0f} steer {steer:+.0f}")
        if name != self.last_name:
            log.info("web %s -> speed %+.2f turn %+.0f deg/s", name, v, rate)
            self.last_name = name
        return self.telemetry()

    def telemetry(self):
        st = self.robot.state
        r = st.get("imu", max_age=0.5)
        m = st.get("motors") or {}
        s = st.get("safety") or {}
        return {
            "applied": list(m.get("applied", (0.0, 0.0))),
            "reason": m.get("reason"),
            "hold": m.get("hold_yaw") is not None,
            "stop": bool(s.get("stop", True)),
            "reasons": s.get("reasons", ["starting"]),
            "latched": s.get("latched", []),
            "yaw": None if r is None else round(r.yaw_deg, 1),
            "yaw_rate": None if r is None else round(r.yaw_rate_dps, 1),
            "turn_goal": round(m.get("turn_goal", 0.0), 1),
            "tilt": None if r is None else round(r.tilt_deg, 1),
            "pitch": None if r is None else round(r.pitch_deg, 1),
            "fault": self.robot.fault,
            "power": st.get("power") or {"ok": False, "error": "not read yet"},
        }

    # ---- HTTP ----

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "ArcbyDrive"
            sys_version = ""

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self):
                url = urlparse(self.path)
                key = parse_qs(url.query).get("k", [""])[0] or self.headers.get("X-Arcby-Key", "")
                return secrets.compare_digest(key, server.token)

            def do_GET(self):
                url = urlparse(self.path)
                if url.path != "/":
                    return self._send(404, {"error": "not found"})
                if not self._authorized():
                    return self._send(403, b"wrong or missing key - use the link printed by drive_web.py\n",
                                      "text/plain; charset=utf-8")
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")

            def do_POST(self):
                if not self._authorized():
                    return self._send(403, {"error": "wrong key"})
                path = urlparse(self.path).path
                try:
                    n = min(int(self.headers.get("Content-Length", 0)), 4096)
                    msg = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(msg, dict):
                        raise ValueError("expected a JSON object")
                    if path == "/api/drive":
                        return self._send(200, server.drive(msg))
                    if path == "/api/reset":
                        server.robot.safety.reset()
                        log.info("web: safety reset")
                        return self._send(200, server.telemetry())
                    if path == "/api/snapshot":
                        saved = server.robot.safety.snapshot("manual")
                        return self._send(200, {"saved": None if saved is None else str(saved)})
                    return self._send(404, {"error": "not found"})
                except (ValueError, TypeError) as e:
                    server.robot.motors.request(0.0, 0.0, "web")      # bad message -> stop
                    return self._send(400, {"error": str(e)})

            def log_message(self, fmt, *args):
                pass

        return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Arcby drive</title>
<style>
:root{--bg:#0b0e11;--card:#141a20;--line:#26313a;--fg:#d5dee5;--dim:#6f7f8b;--ok:#4ade80;--warn:#facc15;--bad:#f87171;--acc:#38bdf8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 ui-sans-serif,system-ui,-apple-system,sans-serif;
  padding:16px;user-select:none;-webkit-user-select:none;touch-action:none}
h1{font-size:18px;margin:0 0 12px;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
h1 small{color:var(--dim);font-weight:normal;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px;max-width:980px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.banner{border-radius:10px;padding:12px 14px;font-weight:600;margin-bottom:12px;max-width:980px;display:flex;
  justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.banner.ok{background:#12301f;color:var(--ok)}.banner.bad{background:#3a1414;color:var(--bad)}.banner.wait{background:#33300f;color:var(--warn)}
button{background:#22303a;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px 14px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}
.keys{display:grid;grid-template-columns:repeat(3,64px);grid-template-rows:repeat(2,64px);gap:6px;justify-content:center;margin:6px 0}
.key{border:2px solid var(--line);border-radius:10px;display:grid;place-items:center;font-size:20px;font-weight:700;color:var(--dim);background:#0f1418;touch-action:none}
.key.on{border-color:var(--acc);color:#000;background:var(--acc)}
.key small{display:block;font-size:10px;font-weight:400}
.k-w{grid-column:2}.k-a{grid-column:1;grid-row:2}.k-s{grid-column:2;grid-row:2}.k-d{grid-column:3;grid-row:2}
.row{display:flex;justify-content:space-between;gap:10px;margin:6px 0}
.row b{color:var(--dim);font-weight:400}
.bars{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px}
.bar{height:120px;background:#0f1418;border:1px solid var(--line);border-radius:8px;position:relative;overflow:hidden}
.bar i{position:absolute;left:0;right:0;background:var(--acc)}
.bar .mid{position:absolute;left:0;right:0;top:50%;border-top:1px dashed var(--line)}
.bar span{position:absolute;bottom:4px;left:0;right:0;text-align:center;font-size:12px;color:var(--dim)}
input[type=range]{width:100%}
.dim{color:var(--dim)}.big{font-size:28px;font-weight:700}
.compass{width:120px;height:120px;border-radius:50%;border:2px solid var(--line);position:relative;margin:4px auto}
.compass i{position:absolute;left:50%;top:10px;bottom:50%;width:4px;margin-left:-2px;background:var(--acc);border-radius:2px;transform-origin:50% 100%}
</style></head><body>
<h1>Arcby drive <small id="link">connecting...</small></h1>
<div class="banner wait" id="banner"><span id="btext">starting - keep the robot still</span><button id="reset" hidden>Reset (R)</button></div>
<div class="grid">
  <section class="card">
    <div class="keys">
      <div class="key k-w" data-k="up">W<small>gas</small></div>
      <div class="key k-a" data-k="left">A<small>left</small></div>
      <div class="key k-s" data-k="down">S<small>back</small></div>
      <div class="key k-d" data-k="right">D<small>right</small></div>
    </div>
    <div class="dim" style="text-align:center">hold keys (also arrows) · Space = stop · works with touch</div>
    <div style="text-align:center;margin-top:10px"><button id="snap">Save last 3 s (B)</button>
      <div class="dim" id="snaptext" style="font-size:12px;margin-top:4px"></div></div>
  </section>
  <section class="card">
    <div class="row"><b>speed</b><span id="speedv"></span></div>
    <input type="range" id="speed" min="0.1" max="1" step="0.1">
    <div class="row"><b>turn rate (A / D)</b><span id="spinv"></span></div>
    <input type="range" id="spin" min="20" max="300" step="10">
    <div class="dim" style="font-size:12px">curves follow the slider; on the spot it spins at least 250 °/s</div>
    <div class="row"><b>motors (PWM)</b><span class="dim" id="mtext"></span></div>
    <div class="bars">
      <div class="bar"><div class="mid"></div><i id="bl"></i><span>left</span></div>
      <div class="bar"><div class="mid"></div><i id="br"></i><span>right</span></div>
    </div>
  </section>
  <section class="card">
    <div class="compass"><i id="needle"></i></div>
    <div class="row"><b>heading (yaw)</b><span class="big" id="yaw">--</span></div>
    <div class="row"><b>tilt / pitch</b><span id="tilt">--</span></div>
    <div class="row"><b>turn rate: asked / real</b><span id="rate">--</span></div>
    <div class="row"><b>straight hold</b><span id="hold">--</span></div>
    <div class="row"><b>power</b><span id="power">--</span></div>
  </section>
</div>
<script>
"use strict";
const K = new URLSearchParams(location.search).get("k") || "";
const $ = id => document.getElementById(id);
const held = new Set();
const CODES = {KeyW:"up",ArrowUp:"up",KeyS:"down",ArrowDown:"down",KeyA:"left",ArrowLeft:"left",KeyD:"right",ArrowRight:"right"};
function load(name, def){ try { const v = parseFloat(localStorage.getItem(name)); return isNaN(v) ? def : v; } catch(e) { return def; } }
function save(name, v){ try { localStorage.setItem(name, v); } catch(e) {} }
const speed = $("speed"), spin = $("spin");
speed.value = load("arcby-speed", 0.4); spin.value = load("arcby-turn", 90);
function showSliders(){ $("speedv").textContent = (+speed.value).toFixed(1); $("spinv").textContent = (+spin.value).toFixed(0) + " °/s"; }
speed.oninput = () => { save("arcby-speed", speed.value); showSliders(); };
spin.oninput = () => { save("arcby-turn", spin.value); showSliders(); };
showSliders();

function paintKeys(){ document.querySelectorAll(".key").forEach(el => el.classList.toggle("on", held.has(el.dataset.k))); }
function stopAll(){ held.clear(); paintKeys(); send(); }
addEventListener("keydown", e => {
  if (CODES[e.code]) { held.add(CODES[e.code]); e.preventDefault(); paintKeys(); }
  else if (e.code === "Space") { e.preventDefault(); stopAll(); }
  else if (e.code === "KeyR" && !e.repeat) reset();
  else if (e.code === "KeyB" && !e.repeat) snapshot();
});
addEventListener("keyup", e => { if (CODES[e.code]) { held.delete(CODES[e.code]); paintKeys(); } });
addEventListener("blur", stopAll);                      // switched window: never keep driving
document.addEventListener("visibilitychange", () => { if (document.hidden) stopAll(); });
document.querySelectorAll(".key").forEach(el => {
  const on = e => { e.preventDefault(); held.add(el.dataset.k); paintKeys(); };
  const off = e => { e.preventDefault(); held.delete(el.dataset.k); paintKeys(); };
  el.addEventListener("pointerdown", on); el.addEventListener("pointerup", off);
  el.addEventListener("pointerleave", off); el.addEventListener("pointercancel", off);
});

let busy = false, rtt = null, lastOk = 0;
async function post(path, body){
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 500);   // never hang
  try {
    const r = await fetch(path, {method:"POST", signal:ctl.signal,
      headers:{"Content-Type":"application/json","X-Arcby-Key":K}, body:JSON.stringify(body)});
    if (!r.ok) throw new Error("HTTP " + r.status);
    return await r.json();
  } finally { clearTimeout(timer); }
}

async function send(){
  if (busy) return;
  busy = true;
  const throttle = (held.has("up") ? 1 : 0) - (held.has("down") ? 1 : 0);
  const steer = (held.has("right") ? 1 : 0) - (held.has("left") ? 1 : 0);
  const t0 = performance.now();
  try {
    const s = await post("api/drive", {throttle, steer, speed:+speed.value, turn:+spin.value});
    rtt = performance.now() - t0; lastOk = Date.now(); show(s);
  } catch(e) { $("link").textContent = "NO LINK: " + e.message; $("link").style.color = "var(--bad)"; }
  busy = false;
}
async function reset(){ try { show(await post("api/reset", {})); } catch(e) {} }
$("reset").onclick = reset;
async function snapshot(){
  try { const r = await post("api/snapshot", {}); $("snaptext").textContent = r.saved ? "saved: " + r.saved.split("/").pop() : "nothing to save yet"; }
  catch(e) { $("snaptext").textContent = "not saved: " + e.message; }
}
$("snap").onclick = snapshot;

function bar(el, v){
  const h = Math.min(1, Math.abs(v)) * 50;
  el.style.height = h + "%"; el.style.top = v >= 0 ? (50 - h) + "%" : "50%";
  el.style.background = v >= 0 ? "var(--acc)" : "var(--warn)";
}
function show(s){
  $("link").textContent = "live · " + (rtt == null ? "" : Math.round(rtt) + " ms"); $("link").style.color = rtt > 150 ? "var(--warn)" : "var(--dim)";
  const b = $("banner");
  if (s.fault) { b.className = "banner bad"; $("btext").textContent = "STOPPED: " + s.fault + " - restart drive_web.py"; }
  else if (s.stop) {
    const waiting = s.reasons.length && s.reasons[0].startsWith("imu missing (calibrating");
    b.className = "banner " + (waiting ? "wait" : "bad");
    $("btext").textContent = waiting ? "calibrating the IMU - keep the robot still"
      : s.reasons.includes("low battery") ? "LOW BATTERY - switch off and charge (Pi under-voltage, battery ~7.0 V)"
      : "SAFETY STOP: " + s.reasons.join(", ");
  } else { b.className = "banner ok"; $("btext").textContent = "ready - drive with W A S D"; }
  $("reset").hidden = !(s.latched && s.latched.length);
  bar($("bl"), s.applied[0]); bar($("br"), s.applied[1]);
  $("mtext").textContent = "L " + s.applied[0].toFixed(2) + "  R " + s.applied[1].toFixed(2);
  $("yaw").textContent = s.yaw == null ? "--" : s.yaw.toFixed(0) + "°";
  $("needle").style.transform = "rotate(" + (s.yaw == null ? 0 : -s.yaw) + "deg)";
  $("tilt").textContent = s.tilt == null ? "--" : s.tilt.toFixed(1) + "° / " + s.pitch.toFixed(1) + "°";
  $("hold").textContent = s.hold ? "on" : "off";
  const p = s.power || {};
  $("power").textContent = !p.ok ? "unknown" : p.under_voltage ? "UNDER-VOLTAGE" : p.since_boot ? "ok (dipped earlier - battery getting low?)" : "ok";
  $("power").style.color = !p.ok ? "var(--dim)" : p.under_voltage ? "var(--bad)" : p.since_boot ? "var(--warn)" : "var(--ok)";
  $("rate").textContent = s.yaw_rate == null ? "--" : s.turn_goal.toFixed(0) + " / " + s.yaw_rate.toFixed(0) + " °/s";
}
setInterval(send, 50);
setInterval(() => { if (Date.now() - lastOk > 1000) { $("link").textContent = "NO LINK"; $("link").style.color = "var(--bad)"; } }, 500);
</script></body></html>
"""


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="simulated robot, no motors / IMU needed")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    from robot.main import Robot
    driver = dev = power = None
    if args.sim:
        power = lambda: (False, False, "sim")          # noqa: E731 - the simulation has no battery
        sys.path.insert(0, str(CODE / "hwtest"))
        from base import SimDrive, SimRobot
        sim = SimRobot()
        driver, dev = SimDrive(sim, 1.0), sim.imu_device(imu.parse_axes(config.IMU_AXES))

    if not args.sim:
        from robot.power import AliveRunning, require_no_alive
        try:
            require_no_alive()
        except AliveRunning as e:
            sys.exit(f"Could not start: {e}")
    token = secrets.token_urlsafe(16)      # 128 bits: not guessable from the LAN (3 bytes were)
    try:
        with Robot(driver=driver, imu_dev=dev, power_reader=power) as robot:
            web = DriveServer(robot, token, port=args.port)
            web.serve_in_thread()
            url = f"http://{lan_ip()}:{web.port}/?k={token}"
            log.info("drive_web started (sim=%s) on port %d", args.sim, web.port)
            print(f"\nArcby drive{' (SIMULATION)' if args.sim else ''} - open on your Mac or phone:\n\n    {url}\n\n"
                  "Keep the robot still for 2 s (IMU calibration). Ctrl+C here stops everything.\n")
            try:
                while robot.alive:
                    time.sleep(0.2)
                if robot.fault:
                    print(f"STOPPED: {robot.fault} (details in {LOG})")
            finally:
                web.close()
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f"Could not start: {e}\nIs the MPU6050 connected (pins 1/3/5/25)? Port {args.port} free? Details in {LOG}")
        log.error("start failed: %s", e)
    print("motors stopped, bye")
    log.info("drive_web exit")


if __name__ == "__main__":
    main()
