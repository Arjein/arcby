# Arcby status dashboard

A small read-only web page that shows what the Pi is doing: health, the 40-pin header with live pin states,
devices, network, processes and logs. One Python file, standard library only:
`pi/dashboard/arcby_dashboard.py`.

## Open it
- From any device on the home Wi-Fi: **http://raspberrypi.local:8080/** (works on a phone too).
- Raw data: `curl -s http://raspberrypi.local:8080/api/status`, `.../api/sensors` and `.../api/live` (the robot's live state)
- Live camera: the **Camera** card -> *Start camera* (or open http://raspberrypi.local:8080/camera.mjpg directly).
- The page refreshes every 2 s (sensors and the robot's live state every 0.5 s) and pauses while the browser tab is hidden.

## Alarm bar (2026-11-02) - on top of every tab
Red and pulsing: **under-voltage now** (battery nearly empty / supply too weak), the Pi at >= 80 °C, the SD card
under 1 GB free, **the USB stick not mounted** (nothing recorded), and **the Pi not answering** for 6 s (the page
itself notices: power off / battery empty / Wi-Fi). Orange: under-voltage since the last boot, 75 °C, throttled, SD
under 3 GB, the stick under 5 GB, alive.py not running. Grey line: on wall power (no head, no motors). From
`/api/status` (`alerts`, every 2 s; `alerts()` in arcby_dashboard.py).

## Tabs (2026-11-02)
**Live** (what the robot does now) · **People** · **Heard** · **Learning** · **System** (health, sensors, camera,
speaker, wiring, logs). The last tab is remembered per browser.

- **People**: a card per person (best photo, recorded pictures, face prints, voice prints, last seen) + "Not
  recognised". Inside: rename, how it calls them, **merge into** another person, photo on the wall yes / no / auto,
  **forget completely** (face, voice prints and every recorded picture of them); every recorded face with
  **"this is..."** (moves it to that person or a new one: stored for them, removed from whoever had it) and
  **delete** (the picture and its stored copy). Dangerous buttons need a second click within 4 s.
- **Heard**: the kept sentences (newest 40) with a player, the text and the kind; delete.
- Changes never edit the robot's files from the dashboard: they are requests in `~/scratch/alive/commands/` that
  alive.py carries out within ~1 s (`robot/commands.py`) - or `people_cli.py apply` when alive.py is not running.
  Every label goes into `~/.local/share/arcby/labels.jsonl` (ground truth for the learning jobs).
- Code: `pi/dashboard/arcby_archive.py` next to the dashboard (deploy.sh copies both). Pictures / clips only from the
  people store, the recordings, collect and heard folders (`/media/...`, nothing outside them). No password (Selim 2026-11-02) - anyone on the home network can see the family's faces and voices here.

## What it shows
**At the top: the robot live** (`GET /api/live` = `~/scratch/alive/state.json`, written by `code/alive.py` twice a
second, plus `age_s`, `stale` (older than 3 s), `running` (alive.py process found) and `why`). Names and numbers
only - no picture, no sound. When the file is missing, broken or older than 3 s, a red banner replaces these cards
("Robot not running (alive.py)" / "State stale (N s old)"); a missing key (no ears, no head, not recording) only
blanks that part.

| Live card | Shows |
|---|---|
| Who it sees | one chip per face, robot's left first: name, `?` (not recognised yet), `stranger`, `photo` (a picture on the wall); *looking* badge + engagement bar; left / ahead / right, pan / tilt, size; blue border + *attention* badge = the face its attention is on. "nobody in view" when empty |
| Attention | mode (follow, hold, search, glance, listen, break, orient, scan, idle) with a plain explanation, on whom, conversation partner, gaze pan / tilt, head on (pose, °/s) or off (no battery) |
| Drive | alive.py's drive (`robot/legs.py`): mode (explore / follow / stopped / off + why: battery off, `--no-drive`), its state, for how long, who started it or why it stopped, sonar, safety; **STOP** and **Explore (2 min)** buttons = `drive_stop` / `drive_start` requests through `robot/commands.py` (alive.py reads them 4x a second; alive.py not running = nothing to stop) |
| Conversation now | talking with whom + seconds left, what it waits for (e.g. a name), dropped count |
| Ears | stopped / muted (robot speaks) / hearing someone / quiet, level dBFS + gain, heard + barge-ins, STT model, live model, STT requests today |
| Voice + mind | speaking / busy / quiet, TTS model + voice, requests today / daily limit bar; mind model + requests / limit |
| Robot sensors | the robot's own readings: IMU (roll, pitch, yaw, tilt, tipped, bump, chip temp), sonar (cm or why not), system (CPU °C, load, free RAM / disk, fan, undervoltage), camera (fps, EV, detect / embed ms), recording (folder, failed); status per sensor |
| People it knows | names (+ how it addresses them), voice-print count ("needs more" under 10), green = in view now |
| Conversation | the history card below (moved up so all the talk is together) |

**Below: the Pi itself.**

| Section | Source | Refresh |
|---|---|---|
| Health: CPU temp (green < 65 °C, yellow < 75 °C, red above), power flags from `vcgencmd get_throttled` (under-voltage, freq capped, throttled, soft temp limit; *now* and *since boot*), CPU % per core, core MHz, core voltage, load, uptime, memory, disk `/` | `/proc`, `/sys`, `vcgencmd` | every poll |
| GPIO header: all 40 pins laid out like the real header (pin 1 top-left, odd pins left), Arcby connection, group colour, live function / pull / level. ▲boot marks GPIO0–8 (boot HIGH). Red warning if a motor pin is on GPIO0–8 or GPIO16 is used. | `pinctrl get` (read only — never claims pins, so `robot_test.py` / `robot_drive.py` can run at the same time) | every poll |
| Wiring: every module wire → Pi pin (+ breadboard column), and the breadboard column map | `WIRING` / `BREADBOARD` lists in the code | static |
| Devices: camera, sound cards, I2C bus 1, USB | `rpicam-hello --list-cameras`, `/proc/asound/cards`, `i2cdetect -y 1`, `lsusb` | cached 30–60 s |
| Network: hostname, IPs per interface, Wi-Fi SSID + signal | `ip -j addr`, `/proc/net/wireless`, `iwgetid` | IPs/SSID 30 s, signal every poll |
| Processes: top 8 by memory and by CPU, dashboard's own RSS | `/proc/<pid>/stat` | every poll |
| Logs: last 30 journal lines | `journalctl -n 30` | cached 10 s |
| Conversation: what `code/alive.py` heard (text), said, dropped (heard but not sent: nobody looking), whose voice, head turns to a voice (face / last seen / sweep), and its ears status line (heard count, gain, level, VAD max) | `~/scratch/alive/events.jsonl` + `alive.log` (text only - the robot keeps no audio) | every poll |
| Learning: what `code/learn.py` is doing on the wall adapter (job, progress bar, time left, what comes next), the proposals waiting for Selim with *apply* / *reject* buttons (`POST /api/learn-decide` -> runs `code/learn.py apply|reject ID` - the rules stay in one place), the last results (findings + changes and how they went: applied / proposed) | `~/scratch/learn/state.json` (learn.py writes it 2x a second) | every 2 s |
| Speaker: level slider 0-100 % - one of its two writes (`POST /api/speaker-level` -> `~/.config/arcby/speaker_level.json`); the robot's voice reads it before each sentence | file | on change |
| Camera: live picture 640×480 ~10 fps, *Start / Stop* button, *Rotate 180°* (remembered per browser), status line (live / fps / viewers / error such as "camera busy") | `rpicam-vid --codec mjpeg` started by the dashboard, frames shared by all viewers on `/camera.mjpg` | **only while someone watches**: starts on *Start*, stops 3 s after the last viewer (Stop, tab closed or hidden) |
| Sensors: HC-SR04 distance (big number, status, median + spread of the last 8 pings, 10 s graph). IMU slot for later. | HC-SR04 pinged by the dashboard itself via `python3-lgpio` (GPIO5 TRIG / GPIO6 ECHO) | 4 pings/s, **only while a page is open** (stops 10 s after the last poll) |

If one data source fails, only that section shows a red error line.
The wiring lives in the `WIRING` and `BREADBOARD` lists in the code; the GPIO header labels are generated from `WIRING`, so the two cards always agree. **Update them together with `docs/pinout.md` and `docs/wiring.md`.**

## Install / restart
- Files on the Pi: `/srv/arcby/dashboard/` (`arcby_dashboard.py`, `dashboard.log`, `dashboard.pid`).
- Deploy or restart (from the Mac, or on the Pi itself): `pi/dashboard/deploy.sh`
  — copies the file, stops the old instance (pid file), starts a new one and checks it answers.
- Stop: `kill $(cat /srv/arcby/dashboard/dashboard.pid)`
- Run by hand in the foreground: `python3 /srv/arcby/dashboard/arcby_dashboard.py` (Ctrl+C stops).
  Optional env vars: `ARCBY_DASHBOARD_PORT` (default 8080), `ARCBY_DASHBOARD_HOST` (default 0.0.0.0).
- Autostart at boot (no sudo) — add this line with `crontab -e` as `arjein`:
  ```
  @reboot cd /srv/arcby/dashboard && flock -n /srv/arcby/dashboard/.lock nohup python3 /srv/arcby/dashboard/arcby_dashboard.py >> /srv/arcby/dashboard/dashboard.log 2>&1 < /dev/null
  ```
  `flock -n` (same lock as `deploy.sh`) makes sure it never runs twice; a second copy would also fail to bind port 8080.

## Resource use (measured 2026-10-28)
Test run on the Pi (bound to 127.0.0.1), polled every 2 s for 64 s:
**RSS 20.2 MB** (VSZ 104 MB), **1.9 % of one core** on average (≈ 1.2 s CPU in 64 s).

## Safety
- Read-only except four JSON POSTs: `/api/command` (People tab, checked here and again by the robot), `/api/heard-delete` (one clip), `/api/speaker-level` (a number, clamped) and `/api/learn-decide` (`{"id": "p<digits>", "accept": bool}` - checked, then `learn.py apply|reject` with that id; no other command can come from the request). JSON only: another web page cannot send it without a CORS preflight, which the server refuses. Everything else: `GET /`, `/api/status`, `/api/sensors`, `/api/live`, `/api/learn`, `/camera.mjpg`.
- The camera can only be used by one program at a time. While the Camera card is on, `rpicam-still`,
  `code/hwtest/bringup.py --only camera` and later the robot get "camera busy" - press Stop first (it is free
  ~3 s later). Turn the camera card off completely with `ARCBY_DASHBOARD_CAMERA=0`.
  Cost while streaming: one `rpicam-vid` (hardware JPEG) + ~50 KiB per frame over Wi-Fi; nothing when stopped.
- Exception: the distance card drives the HC-SR04 TRIG pin (GPIO5). The pins are claimed only for each ping
  (a few ms), so `code/hwtest/distance.py` can still run; the card then shows "pins busy". Turn it off with
  `ARCBY_DASHBOARD_DISTANCE=0`. Once the robot software owns the sensor, the dashboard should read the robot's
  published value instead of pinging.
- All text is HTML-escaped in the page.
- No password: anyone on the home network can see process command lines and the journal. Do not open port 8080 to the internet.
