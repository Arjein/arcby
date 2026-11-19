// Runs the drive page's real <script> in node with a pretend browser and prints what it sends.
// Used by tests/test_drive_page.py. Usage: node page_harness.js <script.js>
"use strict";
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");

function el(id) {
  return { id, textContent: "", className: "", hidden: false, value: "", style: {}, dataset: {},
           listeners: {}, classList: { set: new Set(), toggle(c, on) { on ? this.set.add(c) : this.set.delete(c); } },
           addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); } };
}
const els = {}; const keys = ["up", "left", "down", "right"].map(k => { const e = el("key-" + k); e.dataset.k = k; return e; });
const docListeners = {}, winListeners = {};
global.document = { hidden: false, getElementById: id => (els[id] = els[id] || el(id)),
  querySelectorAll: () => keys, addEventListener: (t, f) => (docListeners[t] = docListeners[t] || []).push(f) };
global.addEventListener = (t, f) => (winListeners[t] = winListeners[t] || []).push(f);
global.location = { search: "?k=sekret" };
const store = {};
global.localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
let intervals = [], timeouts = [];
global.setInterval = (f, ms) => intervals.push({ f, ms });
global.setTimeout = (f, ms) => { const t = { f, ms, done: false }; timeouts.push(t); return t; };
global.clearTimeout = t => { if (t) t.done = true; };
global.performance = { now: () => 0 };
const sent = []; let fetchMode = "ok";
const tele = { applied: [0.1, 0.2], reason: null, hold: false, stop: false, reasons: [], latched: [],
               yaw: 12.0, yaw_rate: 0.0, turn_goal: 0.0, tilt: 3.0, pitch: 1.0, fault: null };
global.fetch = (path, opt) => {
  sent.push({ path, key: opt.headers["X-Arcby-Key"], body: JSON.parse(opt.body) });
  if (fetchMode === "fail") return Promise.reject(new Error("network down"));
  if (fetchMode === "hang") return new Promise((res, rej) => opt.signal.addEventListener("abort", () => rej(new Error("aborted"))));
  return Promise.resolve({ ok: true, status: 200, json: async () => tele });
};
eval(src);

const sendTick = intervals.find(i => i.ms === 50).f;
const flush = () => new Promise(r => setImmediate(r));
async function tick() { sendTick(); await flush(); await flush(); return sent[sent.length - 1]; }
function fire(target, type, ev) { for (const f of (target[type] || [])) f(Object.assign({ preventDefault() {}, repeat: false }, ev)); }
const key = (type, code) => fire(winListeners, type, { code });
const out = {};
(async () => {
  out.defaults = (await tick()).body;
  key("keydown", "KeyW"); out.w = (await tick()).body;
  out.w_key = sent[sent.length - 1].key;
  key("keydown", "KeyA"); out.w_a = (await tick()).body;
  key("keyup", "KeyW"); out.a_only = (await tick()).body;
  key("keyup", "KeyA"); out.none = (await tick()).body;
  key("keydown", "ArrowUp"); key("keydown", "ArrowRight"); out.arrows = (await tick()).body;
  fire(winListeners, "blur", {}); await flush(); out.after_blur_immediate = sent[sent.length - 1].body;
  out.after_blur = (await tick()).body;
  key("keydown", "KeyS"); document.hidden = true; fire(docListeners, "visibilitychange", {}); await flush();
  out.after_hidden = sent[sent.length - 1].body; document.hidden = false;
  key("keydown", "KeyD"); key("keydown", "Space"); await flush(); out.after_space = sent[sent.length - 1].body;
  keys[0].listeners.pointerdown[0]({ preventDefault() {} }); out.touch_hold = (await tick()).body;
  keys[0].listeners.pointerleave[0]({ preventDefault() {} }); out.touch_leave = (await tick()).body;
  els.speed.value = "0.7"; els.speed.oninput(); els.spin.value = "200"; els.spin.oninput();
  out.sliders = (await tick()).body; out.stored = { ...store };
  key("keydown", "KeyR"); await flush(); out.reset_path = sent[sent.length - 1].path;
  key("keydown", "KeyB"); await flush(); out.snap_path = sent[sent.length - 1].path;
  out.banner_ok = els.banner.className; out.yaw_text = els.yaw.textContent;
  tele.stop = true; tele.reasons = ["bump"]; tele.latched = ["bump"]; await tick();
  out.banner_stop = els.banner.className; out.banner_text = els.btext.textContent; out.reset_hidden = els.reset.hidden;
  tele.stop = true; tele.reasons = ["imu missing (calibrating)"]; tele.latched = []; await tick();
  out.banner_calib = els.banner.className;
  tele.reasons = ["low battery"]; tele.latched = ["low battery"];
  tele.power = { ok: true, under_voltage: true, since_boot: true, source: "rpi_volt" }; await tick();
  out.banner_battery = els.btext.textContent; out.power_text = els.power.textContent;
  tele.stop = false; tele.reasons = []; tele.latched = []; tele.power = { ok: true, under_voltage: false, since_boot: true }; await tick();
  out.power_dipped = els.power.textContent;
  fetchMode = "fail"; await tick(); out.link_fail = els.link.textContent;
  fetchMode = "hang"; const n0 = sent.length; sendTick(); await flush(); sendTick(); await flush();
  out.hang_requests = sent.length - n0;                     // busy: second tick must not send
  const abortTimer = timeouts.filter(t => !t.done && t.ms === 500).pop();
  abortTimer.f(); await flush(); await flush();               // 500 ms later: aborted
  fetchMode = "ok"; await tick(); out.recovered = sent.length - n0;
  console.log(JSON.stringify(out));
})().catch(e => { console.log(JSON.stringify({ error: String(e && e.stack || e) })); });
