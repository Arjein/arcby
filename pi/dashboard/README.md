# pi/dashboard/: the robot's web dashboard

A status page served by the Pi on port 8080. It is a single Python file using only the standard library, ~20 MB RSS
and ~2 % of one core. Deploy with `deploy.sh`, which copies it to `/srv/arcby/dashboard/`, restarts it and checks
that it answers.

| Tab | Shows |
|---|---|
| **Live** | what the robot is doing now: faces by name (*looking* / *photo* badges), attention mode and target, drive state with STOP / Explore, ears level and gain, voice and LLM budgets, IMU / sonar / camera numbers, the conversation log |
| **People** | every person the robot knows, with recorded faces and voice-print counts: rename, merge, forget, "this face is…", delete pictures |
| **Heard** | the last 40 sentences it kept, with text and a player |
| **Learning** | what learning mode is doing (job, progress, time left) and its proposals, with apply / reject |
| **System** | health (temperature, throttling and under-voltage flags, CPU, memory, disk), the 40-pin header with live pin states, the wiring map, devices, network, processes, logs, live camera |

An **alarm bar** on every tab: under-voltage now, ≥ 80 °C, disk nearly full, the data drive missing, the Pi not
answering.

## Design

- **Read-only by default.** The page never changes the robot's files directly. Edits (People, Learning) are JSON
  requests dropped into a queue folder, and `alive.py` carries them out within ~1 s (`robot/commands.py`), so the
  people store has exactly **one writer**. The only direct write is the speaker level, a number that is clamped.
- **Live state without coupling.** `alive.py` writes its state to a JSON file twice a second, and the dashboard
  serves it with its age. Older than 3 s = "robot not running". The dashboard never imports the robot.
- **Pins without claiming them.** Pin states come from `pinctrl get`, which never takes ownership, so the motor
  programs can run at the same time.
- **Cheap when nobody looks.** The page polls only while its tab is visible. The camera stream (hardware MJPEG)
  runs only while someone is watching it.
- The wiring card and the GPIO labels are generated from one `WIRING` list, so they cannot disagree.
- No password, so it is for the home network only; never expose port 8080 to the internet. Screenshots should blur
  the Network card.

More: [`docs/dashboard.md`](../../docs/dashboard.md).
