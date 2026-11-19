# code/

The programs that run on the robot (a Raspberry Pi 4 running Raspberry Pi OS 64-bit) and the tools around them. The
logic lives in the [`robot/`](robot/) package, and every program here is a thin entry point on top of it.

## Programs

| Program | What it does |
|---|---|
| `alive.py` | **The robot.** Camera → faces → identities → attention → head motion, plus ears, conversation, voice and (with the battery on) driving. Started at boot by `run_alive.sh` |
| `run_alive.sh` | supervisor: restarts `alive.py` 30 s after a crash. After a failed start (camera busy, model missing) it backs off 60 s → 10 min; a clean stop ends it |
| `nav.py` | autonomous driving as its own program, for tuning: Explore / Go to people / STOP on a web page, with a **dead man's switch** (it stops within 1.5 s when the page closes). Every drive is recorded |
| `drive_web.py` | real-time browser drive, car style (W/A/S/D held, or touch on a phone). The key presses travel as a 20 Hz lease, so the motors stop when the messages stop |
| `drive.py` | terminal fallback drive. Terminals send no key-release events, hence a 0.6 s lease per key press |
| `body_mcp.py` | MCP server (stdio, stdlib only) that gives a Claude session tools to move, turn, look and see through the safety layer |
| `learn.py` | **learning mode**: jobs that re-tune settings from the robot's own recordings while it runs on wall power. Results are proposals: `show` / `apply` / `reject` / `undo` |
| `demo.py` | showcase without the head: greets known people, meets a new face (asks the name, confirms it, stores it) |
| `say.py` | make the robot say a sentence (TTS + cache) |
| `people_cli.py` | list, name, merge, unmerge or forget the people the robot knows; manage voice prints |
| `data_cli.py` | browse the recorded datasets: sessions, rates, gaps, first rows of each stream |
| `get_models.py` | download + sha256-check the ONNX models. `--derive` (on a Mac) re-exports YuNet with a free input size, quantises ArcFace R50 to int8 and exports YOLO26n-pose, all reproducibly |

## Rules every program follows

- **One owner per device.** Only one program may hold the camera, the motors or the IMU. The drive programs refuse to
  start while `alive.py` runs.
- **Only `robot/motors.py` drives the motors**, and `robot/safety.py` can always stop them.
- **No battery = no head, no motors.** The Pi can run on a wall adapter. Programs check whether the servo board
  answers (it is powered from the battery) before they touch any actuator.
- **`--sim`** runs most programs on a laptop with fake hardware.

## Where things live (outside the repo)

| Path on the Pi | Contents |
|---|---|
| `~/.config/arcby/` | secrets (`secrets.env`, chmod 600), household names, speaker level |
| `~/.local/share/arcby/` | models, the people store (face + voice prints), learnt settings (`tuned.json`), datasets |
| `~/scratch/` | logs, live state for the dashboard, black boxes |

Nothing the robot sees or hears is ever written into the repository; `.gitignore` and a pre-commit hook enforce this.

## Setup

```bash
pip install -r code/requirements.txt          # + on the Pi: sudo apt install python3-picamera2 python3-gpiozero
python3 code/get_models.py
cp code/examples/secrets.env.example ~/.config/arcby/secrets.env && chmod 600 ~/.config/arcby/secrets.env
python3 code/hwtest/bringup.py                # check every module
```

Tests: `python3 -m unittest discover -s code/tests -t code` (see [`tests/`](tests/)).
