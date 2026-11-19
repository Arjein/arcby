#!/usr/bin/env python3
"""
Arcby drives by itself: explores carefully, goes to people (docs/navigation.md).

  python3 code/nav.py                  # the robot (alive.py stopped, battery on, robot on the floor)
  python3 code/nav.py --sim            # a simulated room on the Mac: same brain, same page
  python3 code/nav.py --sim --seed 4   # another room

It prints a link with a secret key - open it on the phone or the Mac. The page shows what the robot sees (the
camera with the floor check drawn on it: green = floor free that far, red = an obstacle's foot), a top view of
what it remembers, its state and why. Buttons:

  Explore       drive around, never into anything
  Go to people  explore, and walk up to anyone it sees (stops ~60 cm before their feet, looks up at them)
  Stop          stand still (the camera and the floor check keep running: put things in front of it and watch)

DEAD MAN: the robot drives only while the page is open and talking to it - close the tab, lock the phone or lose
Wi-Fi and it stops within 1.5 s. Safety (robot/safety.py) is under everything: tipped, bump, IMU lost, low battery.

Start it with ~50 cm of free floor in front of the robot: that patch is how it learns what the floor looks like.
Log: ~/scratch/nav.log (every decision), black boxes of safety stops in ~/scratch/blackbox/.
"""
import argparse
import io
import json
import logging
import math
import secrets
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from robot import config                                    # noqa: E402
from robot.drive_body import Head, SonarThread, people_in   # noqa: E402,F401 - Head / people_in: tests use nav.*
from robot.floor import FloorSeer, pixel_of_ground          # noqa: E402
from robot.nav import Command, Navigator, Percept           # noqa: E402
from robot.nav_rec import NavRecorder                      # noqa: E402

LOG = Path.home() / "scratch" / "nav.log"
DEAD_MAN_S = 1.5
log = logging.getLogger("arcby")


# ---------- the real robot ----------

class RealBody:
    """The robot's hardware behind the navigator: motors + IMU + safety (robot.main.Robot), head, camera, sonar,
    the floor model, the pose model (people mode)."""

    def __init__(self, want_people, record=True):
        from robot.camera import Camera
        from robot.i2c import I2CDevice
        from robot.main import Robot
        from robot.pan_tilt import PanTilt
        from robot.power import require_battery, require_no_alive
        from robot.sonar import Sonar
        require_no_alive()
        require_battery("navigation (motors + head)")
        self.rec = None                 # set before anything can fail: close() looks at it
        self.stop = threading.Event()
        self.robot = Robot().__enter__()
        self.closers = [lambda: self.robot.__exit__(None, None, None)]
        try:
            self.head = Head(PanTilt(I2CDevice(config.I2C_BUS, config.PTZ_ADDR)))
            self.sonar = SonarThread(Sonar())
            self.closers.append(self.sonar.sonar.close)
            self.cam = Camera().start()
            self.closers.append(self.cam.close)
            self.pose = None
            if want_people:
                from robot.body import load_pose
                self.pose = load_pose()
                if self.pose is None:
                    raise RuntimeError("people mode needs the body model (get_models.py --derive --only yolo26n_pose)")
        except BaseException:
            self.close()
            raise
        self.seer = FloorSeer()
        self.view = None
        if record:
            try:
                self.rec = NavRecorder()
                log.info("nav: recording into %s", self.rec.dir)
            except OSError as e:          # no stick, disk full: drive without recording
                log.warning("nav: not recording (%s)", e)
        self.people, self.people_t = [], None
        self.force_learn = False
        self.jpeg = None
        self._pose_job = None
        self._pose_ready = threading.Event()
        for name, fn in (("head", self.head.run), ("sonar", self.sonar.run), ("camera", self._camera),
                         ("pose", self._pose_loop)):
            threading.Thread(target=self._guard, args=(name, fn), name=name, daemon=True).start()

    def _guard(self, name, fn):
        try:
            fn(self.stop)
        except Exception as e:  # noqa: BLE001 - a dead sensor thread stops the robot
            log.exception("nav: %s thread crashed", name)
            self.robot.fault = f"{name} crashed: {e}"
            self.robot.stop_event.set()

    def sonar_free(self):
        last = self.sonar.last
        if last is None or time.monotonic() - last[0] > 0.5:
            return None
        return math.inf if last[1] is None else last[1] - (config.NAV_FRONT_CM - config.NAV_SONAR_X_CM)

    def _camera(self, stop):
        w, h = self.cam.lores_size
        next_jpeg = next_pose = 0.0
        while not stop.is_set():
            with self.cam.frame() as f:
                rgb = f.lores_rgb().copy()
                t = f.t
            pan, tilt, speed = self.head.at(t)
            imu = self.robot.state.get("imu", max_age=0.3)
            spinning = imu is not None and abs(imu.yaw_rate_dps) > 20.0
            if speed < 40.0 and not spinning:
                v = self.seer.see(rgb, pan, tilt, t, sonar_free_cm=self.sonar_free(), force_learn=self.force_learn)
                self.view = v
                if self.rec is not None:
                    self.rec.frame(rgb, v, t, self.sonar_free())
                if time.monotonic() >= next_jpeg:
                    next_jpeg = time.monotonic() + 0.4
                    self.jpeg = overlay_jpeg(rgb, v, w, h)
            if self.pose is not None and time.monotonic() >= next_pose and self._pose_job is None:
                next_pose = time.monotonic() + 0.25
                self._pose_job = (t, pan, tilt, rgb)
                self._pose_ready.set()

    def _pose_loop(self, stop):
        while not stop.is_set():
            if not self._pose_ready.wait(0.5):
                continue
            self._pose_ready.clear()
            job, self._pose_job = self._pose_job, None
            if job is None:
                continue
            t, pan, tilt, rgb = job
            bodies = self.pose.detect(rgb)
            h, w = rgb.shape[:2]
            self.people, self.people_t = people_in(bodies, w, h, pan, tilt), t

    def percept(self):
        st = self.robot.state
        imu = st.get("imu", max_age=config.SAFETY_IMU_STALE_S)
        son = self.sonar.last
        return Percept(t=time.monotonic(), yaw=None if imu is None else imu.yaw_deg,
                       yaw_rate=0.0 if imu is None else imu.yaw_rate_dps,
                       sonar=None if son is None else son[1], sonar_t=None if son is None else son[0],
                       floor=self.view, people=list(self.people), people_t=self.people_t,
                       safety=st.get("safety", max_age=config.SAFETY_MAX_AGE_S), head=self.head.now)

    def record(self, p, cmd, drive):
        if self.rec is not None:
            self.rec.step(p, cmd, drive)

    def apply(self, cmd, drive):
        self.head.target = cmd.head
        self.force_learn = cmd.learn_floor and drive
        if drive:
            self.robot.motors.request_velocity(cmd.v, cmd.turn, "nav")
        else:
            self.robot.motors.request_velocity(0.0, 0.0, "nav")
        if cmd.reset_safety and drive:
            self.robot.safety.reset()

    @property
    def alive(self):
        return self.robot.alive

    @property
    def fault(self):
        return self.robot.fault

    def close(self):
        self.stop.set()
        time.sleep(0.1)
        if self.rec is not None:
            self.closers.insert(0, self.rec.close)     # closed last: after the camera thread stopped
        for c in reversed(self.closers):
            try:
                c()
            except Exception:  # noqa: BLE001 - close everything we can
                log.exception("nav: close")


def overlay_jpeg(rgb, view, w, h):
    """The camera picture with each floor ray drawn: green up to where the floor is free, red dot = a foot."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    if view is not None and view.known:
        for ex, ey, hit in zip(view.ex, view.ey, view.hit):
            end = pixel_of_ground(ex, ey, w, h, view.pan, view.tilt)
            if end is None:
                continue
            d.line([(end[0], h - 1), end], fill=(60, 220, 90), width=2)
            if hit:
                d.ellipse([end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4], fill=(240, 50, 50))
        if view.edge:
            d.text((8, 8), "floor edge", fill=(255, 220, 0))
    elif view is not None:
        d.text((8, 8), "floor not learnt yet", fill=(255, 220, 0))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


# ---------- the simulated robot ----------

class SimBody:
    def __init__(self, want_people, seed):
        from sim.navworld import random_room
        self.world = random_room(seed, people=1 if want_people else 0, walking=True)
        self.lock = threading.Lock()
        self.jpeg = None
        self.fault = None
        self.alive = True
        self._cmd = Command()
        self._drive = False
        self.stop = threading.Event()
        threading.Thread(target=self._run, name="sim", daemon=True).start()

    def _run(self):
        while not self.stop.is_set():
            with self.lock:
                c = self._cmd
                if not self._drive:
                    c = Command(head=c.head)
                self.world.apply(c)
                self.world.advance(0.02)
            time.sleep(0.02)

    def percept(self):
        with self.lock:
            p = self.world.percept()
        if p.floor is not None:
            from dataclasses import replace
            p.floor = replace(p.floor)
        return p

    def apply(self, cmd, drive):
        with self.lock:
            self._cmd, self._drive = cmd, drive

    def truth(self):
        w = self.world
        return {"room": [w.W, w.H], "pose": [round(w.x, 1), round(w.y, 1), round(w.h, 1)],
                "collisions": w.collisions, "cables": w.cable_touches,
                "things": [_shape(o) for o in w.things] + [_shape(c) for p in w.people for c in p.legs()]}

    def close(self):
        self.stop.set()


def _shape(o):
    if hasattr(o, "x0"):
        return {"k": o.kind, "box": [o.x0, o.y0, o.x1, o.y1]}
    if hasattr(o, "r"):
        return {"k": o.kind, "c": [o.x, o.y, o.r]}
    return {"k": o.kind, "line": o.pts}


# ---------- the loop ----------

class Pilot:
    """Runs the navigator at NAV_RATE_HZ on a body; the page switches modes and keeps the dead man alive."""

    def __init__(self, body):
        self.body = body
        self.nav = None
        self.mode = "stopped"
        self.heartbeat = 0.0
        self.lock = threading.Lock()
        self.last_cmd = Command()
        self.last_p = None
        self.paused_by_link = False

    def set_mode(self, mode):
        with self.lock:
            if mode == "stopped":
                self.mode = "stopped"
            elif mode in ("explore", "people"):
                if self.nav is None or self.nav.mode != mode or self.mode == "stopped":
                    self.nav = Navigator(mode)
                elif self.nav.state == "HELP":
                    self.nav.resume()
                self.mode = mode
            else:
                raise ValueError(f"unknown mode {mode!r}")
            log.info("nav: mode %s", self.mode)

    def beat(self):
        self.heartbeat = time.monotonic()

    def step(self):
        p = self.body.percept()
        with self.lock:
            link = time.monotonic() - self.heartbeat <= DEAD_MAN_S
            drive = self.mode != "stopped" and link
            if self.mode != "stopped" and not link and not self.paused_by_link:
                log.warning("nav: no page for %.1f s - standing still", DEAD_MAN_S)
            self.paused_by_link = self.mode != "stopped" and not link
            if self.nav is not None and drive:
                cmd = self.nav.step(p)
            else:
                cmd = Command(head=(0.0, config.NAV_TILT_DEG))
                if self.nav is not None:
                    self.nav.pause()
            self.body.apply(cmd, drive)
            if hasattr(self.body, "record"):        # only the real robot keeps its drives
                self.body.record(p, cmd, drive)
            self.last_cmd, self.last_p = cmd, p

    def run(self, stop):
        period = 1.0 / config.NAV_RATE_HZ
        nxt = time.monotonic()
        while not stop.is_set() and self.body.alive:
            self.step()
            nxt += period
            stop.wait(max(0.0, nxt - time.monotonic()))
            if nxt < time.monotonic() - 1.0:
                nxt = time.monotonic()

    def status(self):
        with self.lock:
            nav, p, c = self.nav, self.last_p, self.last_cmd
            out = {"mode": self.mode, "link_paused": self.paused_by_link, "state": c.state or "-", "note": c.note,
                   "v": round(c.v, 2), "turn": round(c.turn), "head": [round(a) for a in (p.head if p else (0, 0))],
                   "fault": self.body.fault}
            if p is not None:
                out["yaw"] = None if p.yaw is None else round(p.yaw, 1)
                out["sonar"] = None if p.sonar_t is None else ("nothing" if p.sonar is None else round(p.sonar))
                out["safety"] = (p.safety or {}).get("reasons", ["no verdict"])
                out["people"] = [[round(q.bearing), None if q.dist is None else round(q.dist)] for q in p.people]
            if nav is not None:
                m = nav.map
                if len(m.px):
                    rx, ry = m.to_robot(m.px, m.py)
                    keep = np.hypot(rx, ry) < 160
                    out["points"] = np.stack([rx[keep], ry[keep], m.ps[keep]], 1)[-600:].round(1).tolist()
                out["stats"] = {k: round(v) for k, v in nav.stats.items()}
                out["events"] = [[round(t or 0, 1), e] for t, e in nav.events[-12:]]
            if hasattr(self.body, "truth"):
                out["sim"] = self.body.truth()
        return out


# ---------- the page ----------

class Web:
    def __init__(self, pilot, token, port):
        self.pilot, self.token = pilot, token
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, name="web", daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _handler(self):
        web = self

        class H(BaseHTTPRequestHandler):
            server_version, sys_version = "ArifNav", ""

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(data)

            def _ok(self):
                url = urlparse(self.path)
                key = parse_qs(url.query).get("k", [""])[0] or self.headers.get("X-Arcby-Key", "")
                return secrets.compare_digest(key, web.token)

            def do_GET(self):
                path = urlparse(self.path).path
                if not self._ok():
                    return self._send(403, b"wrong or missing key - use the link nav.py printed\n", "text/plain")
                if path == "/":
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if path == "/view.jpg":
                    jpg = web.pilot.body.jpeg
                    return self._send(200, jpg, "image/jpeg") if jpg else self._send(404, {"error": "no picture"})
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                if not self._ok():
                    return self._send(403, {"error": "wrong key"})
                path = urlparse(self.path).path
                try:
                    n = min(int(self.headers.get("Content-Length", 0)), 4096)
                    msg = json.loads(self.rfile.read(n) or b"{}")
                    if path == "/api/status":
                        web.pilot.beat()
                        return self._send(200, web.pilot.status())
                    if path == "/api/mode":
                        web.pilot.set_mode(str(msg.get("mode", "stopped")))
                        web.pilot.beat()
                        return self._send(200, web.pilot.status())
                    return self._send(404, {"error": "not found"})
                except (ValueError, TypeError) as e:
                    web.pilot.set_mode("stopped")
                    return self._send(400, {"error": str(e)})

            def log_message(self, *a):
                pass

        return H


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Arcby drives</title>
<style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#dedcd6;--ok:#2f8f4e;--bad:#c4372c;--acc:#2f5fb3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe7;--mut:#9a9893;--card:#22211f;--line:#383633}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 12px}
.row{display:flex;gap:12px;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px;flex:1 1 300px;min-width:0}button{font:inherit;padding:12px 16px;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
#stop{background:var(--bad);color:#fff;border-color:var(--bad);font-weight:600}
.big{font-size:22px;font-weight:600}.mut{color:var(--mut)}img,canvas{width:100%;border-radius:6px;background:#000;display:block}
ul{margin:6px 0 0;padding-left:18px;font-size:13px}td{padding:1px 8px 1px 0}
</style></head><body><main>
<h1>Arcby drives</h1>
<div class="row" style="margin-bottom:12px">
<button id="explore">Explore</button><button id="people">Go to people</button><button id="stop">STOP</button>
<span id="link" class="mut" style="align-self:center"></span></div>
<div class="row">
<div class="card"><div class="big" id="state">-</div><div id="note" class="mut"></div>
<table id="t" style="margin-top:8px"></table><ul id="ev"></ul></div>
<div class="card"><div class="mut">Camera - green: floor free that far, red: an obstacle's foot</div>
<img id="cam" alt="camera"><div class="mut" style="margin-top:8px">Memory from above (robot in the middle, facing up; 1.5 m each way)
- blue: camera, orange: sonar, red: bump</div><canvas id="map" width="400" height="400"></canvas></div>
</div></main>
<script>
const K=new URLSearchParams(location.search).get("k");const $=id=>document.getElementById(id);let last=Date.now(),mode="stopped";
async function post(p,b){const r=await fetch(p+"?k="+K,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})});return r.json()}
["explore","people","stop"].forEach(m=>$(m).onclick=async()=>{show(await post("/api/mode",{mode:m=="stop"?"stopped":m}))});
document.addEventListener("visibilitychange",()=>{if(document.hidden)post("/api/mode",{mode:"stopped"})});
function show(s){mode=s.mode;last=Date.now();
 ["explore","people"].forEach(m=>$(m).className=s.mode==m?"on":"");
 $("state").textContent=(s.mode=="stopped"?"stopped":s.state)+(s.link_paused?" (no link)":"");
 $("note").textContent=s.fault?("FAULT: "+s.fault):(s.note||"");
 const rows=[["sonar",s.sonar==null?"-":s.sonar+(s.sonar=="nothing"?"":" cm")],["heading",s.yaw==null?"-":s.yaw+"°"],
  ["asked","speed "+s.v+", turn "+s.turn+"°/s"],["head","pan "+s.head[0]+"°, tilt "+s.head[1]+"°"],
  ["safety",(s.safety||[]).join(", ")||"ok"],["people",(s.people||[]).map(p=>p[0]+"°"+(p[1]?" "+p[1]+" cm":"")).join("; ")||"-"]];
 if(s.stats)rows.push(["done",Object.entries(s.stats).map(([k,v])=>k+" "+v).join(", ")]);
 if(s.sim)rows.push(["sim","collisions "+s.sim.collisions+", cables "+s.sim.cables]);
 $("t").innerHTML=rows.map(r=>"<tr><td class=mut>"+r[0]+"</td><td>"+r[1]+"</td></tr>").join("");
 $("ev").innerHTML=(s.events||[]).slice().reverse().map(e=>"<li>"+e[0]+" s: "+e[1]+"</li>").join("");
 draw(s)}
function draw(s){const c=$("map"),g=c.getContext("2d"),W=c.width,S=W/300;g.fillStyle="#111";g.fillRect(0,0,W,W);
 const X=(x,y)=>[W/2-y*S,W/2-x*S];
 if(s.sim){const p=s.sim.pose,a=-p[2]*Math.PI/180;const R=(x,y)=>{const dx=x-p[0],dy=y-p[1];return X(dx*Math.cos(a)-dy*Math.sin(a),dx*Math.sin(a)+dy*Math.cos(a))};
  g.strokeStyle="#555";g.lineWidth=1;for(const t of s.sim.things){g.beginPath();if(t.box){const b=t.box;[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]],[b[0],b[1]]].forEach((q,i)=>{const[u,v]=R(q[0],q[1]);i?g.lineTo(u,v):g.moveTo(u,v)})}
  else if(t.c){const[u,v]=R(t.c[0],t.c[1]);g.arc(u,v,Math.max(1.5,t.c[2]*S),0,7)}else t.line.forEach((q,i)=>{const[u,v]=R(q[0],q[1]);i?g.lineTo(u,v):g.moveTo(u,v)});g.stroke()}
  g.beginPath();[[0,0],[s.sim.room[0],0],[s.sim.room[0],s.sim.room[1]],[0,s.sim.room[1]],[0,0]].forEach((q,i)=>{const[u,v]=R(q[0],q[1]);i?g.lineTo(u,v):g.moveTo(u,v)});g.stroke()}
 const col=["#4a8cff","#ff9a3c","#ff3b3b"];for(const p of s.points||[]){const[u,v]=X(p[0],p[1]);g.fillStyle=col[p[2]];g.fillRect(u-2,v-2,4,4)}
 g.strokeStyle="#ddd";g.lineWidth=2;const[a,b]=X(9.5,7),[d,e]=X(-5.5,-7);g.strokeRect(Math.min(a,d),Math.min(b,e),Math.abs(d-a),Math.abs(e-b))}
async function tick(){try{show(await post("/api/status"));$("link").textContent=""}catch(e){$("link").textContent="NO LINK - the robot stands still"}
 $("cam").src="view.jpg?k="+K+"&t="+Date.now()}
setInterval(tick,500);tick();
</script></body></html>"""


def lan_ip():
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="a simulated room (no hardware)")
    ap.add_argument("--seed", type=int, default=0, help="--sim: which random room")
    ap.add_argument("--no-people", action="store_true", help="do not load the body model (Explore only)")
    ap.add_argument("--port", type=int, default=config.NAV_PORT)
    ap.add_argument("--no-record", action="store_true", help="do not keep the drive (pictures + steps) in DATA_DIR")
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    try:
        body = SimBody(not args.no_people, args.seed) if args.sim else RealBody(not args.no_people, not args.no_record)
    except Exception as e:  # noqa: BLE001 - say why in plain words
        log.exception("nav: start failed")
        sys.exit(f"Could not start: {e}\n(details in {LOG})")
    pilot = Pilot(body)
    crumb = getattr(getattr(body, "robot", None), "breadcrumb", None)
    if crumb is not None:                       # the power breadcrumb says what the drive was doing
        crumb.state = lambda: f"{pilot.mode}/{pilot.last_cmd.state or '-'}"
    token = secrets.token_urlsafe(16)
    web = Web(pilot, token, args.port)
    stop = threading.Event()
    print(f"\nArif drives{' (SIMULATION, room %d)' % args.seed if args.sim else ''} - open:\n\n"
          f"    http://{lan_ip()}:{web.port}/?k={token}\n\n"
          "Keep the robot still for 2 s (IMU calibration), ~50 cm of free floor in front of it.\n"
          "It drives only while that page is open. Ctrl+C here stops everything.\n")
    log.info("nav: started (sim=%s)", args.sim)

    def on_term(signum, frame):     # pkill / kill: stop like Ctrl+C (motors off, recording closed)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    try:
        pilot.run(stop)
        if body.fault:
            print(f"STOPPED: {body.fault} (details in {LOG})")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        web.close()
        body.close()
        log.info("nav: exit")
    print("motors stopped, bye")


if __name__ == "__main__":
    main()
