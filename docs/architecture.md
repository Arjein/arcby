# Arcby — software architecture

Decided 2026-10-28. **No ROS for now**: one Python process on the Pi, split into modules.
The structure deliberately mirrors ROS (modules ≈ nodes, shared state ≈ topics) so modules can
become ROS 2 nodes later (ROS 2 in Docker on the Pi) if we add SLAM / navigation / a lidar.

## Why not ROS 2 yet
- Heavy (hundreds of MB), officially targets Ubuntu; on Raspberry Pi OS it would run in Docker.
- Steep learning curve while the robot itself is still being built.
- The real-time needs are modest (see below) and fit easily in one Python process on a Pi 4.

## Load budget (Pi 4, 4 cores, 4 GB)
| Module | Rate | Load |
|---|---|---|
| motors | 50 Hz commands | tiny |
| imu (MPU6050, I2C) | 100 Hz | tiny |
| distance (HC-SR04) | 10–20 Hz | tiny |
| pan_tilt (Arducam PTZ board, I2C) | on demand; the board generates the servo PWM | negligible |
| camera | 15–30 fps | **heaviest** |
| audio (I2S amp + mic) | continuous stream | light |
| speech / LLM | on demand | heavy → offload to cloud or the Mac |

No hard real-time requirement. The only time-critical job is **safety** (stop the motors fast).

## Layout
```
robot/
  state.py      shared state (latest readings + timestamps), thread-safe
  motors.py     drive: speed/direction, straight-line correction from imu yaw rate
  imu.py        MPU6050: yaw angle/rate, tilt, bump detection (board mounted upside down → invert axes)
  distance.py   HC-SR04 range
  pan_tilt.py   camera head
  camera.py     frames (picamera2)
  audio.py      speaker + microphone
  safety.py     50 Hz loop: obstacle too close, tipped over, bump, stale sensor data, lost connection → stop motors
  main.py       starts every module, owns shutdown (motors stopped on exit or crash)
```

## Rules
- Each module runs its own loop at its own rate and only **writes its own fields** in `state`.
- Only `motors.py` drives the motors; other modules request motion through `state` / a command queue.
- `safety.py` has the last word: it can always override a motion command with stop.
- Every reading carries a timestamp; data older than its timeout counts as missing → safety stop.
- Motors must stop on: Ctrl+C, exceptions, process exit, and loss of the remote-control connection (watchdog).
- Pin numbers come from `docs/pinout.md`; keep them in one config module, not scattered.
- Hardware access stays inside its module so the rest can be tested on the Mac with fakes.

## Code and tests
- The package lives in `code/robot/`: `config.py`, `state.py`, `i2c.py`, `imu.py`, `motors.py`, `safety.py`,
  `teleop.py` (keyboard logic), `main.py` (`Robot`: starts imu / safety / motors threads, owns shutdown).
- `code/drive_web.py` = real-time drive from a browser on top of `Robot`: the page sends the held keys
  20x/s (`POST /api/drive`), `robot/teleop.car_velocity` turns them into speed + turn rate (car style: W/S gas
  and reverse, A/D = turn rate from a slider in deg/s, turning on the spot when standing). Stops when keys are released, the tab loses focus, or messages
  stop (motors watchdog 0.3 s). Port 8090, a random key in the link guards it on the LAN.
- `code/drive.py` = terminal fallback (terminals send no key-release events, hence a 0.6 s key lease).
- Unit tests in `code/tests/` use fakes (e.g. a register-level fake MPU6050), standard library only:
  `python3 -m unittest discover -s code/tests -t code -v` (runs on the Pi and the Mac).
- Hardware tests in `code/hwtest/` (one per unit, interactive, report to `~/scratch/hwtest/`).

## Motors + safety (`robot/motors.py`, `robot/safety.py`, done 2026-10-29)
- Two ways to move, both repeated faster than `DRIVE_CMD_TIMEOUT_S` (0.3 s) or the motors stop (watchdog):
  `motors.request(left, right)` (wheel speeds -1..1) and `motors.request_velocity(v, turn_dps)` (speed -1..1
  + turn rate in deg/s). Velocity mode is the interface for drivers and a future robot brain: the gyro
  closes the loop on the turn rate (feed-forward `TURN_K` 400 deg/s per PWM difference + PI), straight
  driving holds the heading. Turning on the spot (`TURN_ON_SPOT = "spin"`, chosen; `"auto"` = spin when fast,
  pivot when slow): both wheels opposite for
  fast turns (power from the measured spin curve; friction makes them all-or-nothing below ~250 deg/s on
  carpet, `hwtest/base.py --only spinramp`), one wheel at a time for slower ones (forward and backward
  pivots mixed so the robot does not walk away).
- 50 Hz loop: dead band (any non-zero request >= `DRIVE_MIN` 0.35 PWM), soft start (`DRIVE_ACCEL` 2/s up,
  6/s down), IMU heading hold on straight requests (`HEADING_KP` 0.01, max steer 0.1).
- Safety 50 Hz: IMU missing/stale, tipped (latched), bump while driving at steady speed (latched, ignored for
  0.3 s after reaching the target speed, and ignored when the pitch rate was > 40 deg/s in the 0.3 s before
  the jolt: the nose lifts at take-off and lands with a thump ~0.5 s later). Every new stop writes a
  black box CSV (last 3 s of IMU + motors) to `~/scratch/blackbox/`; stops are logged to `~/scratch/drive.log`. Stop is instant; motors also stop if the safety verdict is older
  than 0.2 s. Latched stops clear with `safety.reset()` (keyboard `r`).
- `Robot`: a crash in any thread stops everything; leaving the `with` block stops the motors and pulls the
  pins down.
- Power (`robot/power.py`): no battery ADC, so the Pi's own under-voltage alarm (kernel `rpi_volt`, fallback
  `vcgencmd`) stands in for "battery low": the Pi buck sags when the 2S pack is at ~7.0 V. Under-voltage for
  1 s -> latched "low battery" stop, shown on the drive page; reset only once the voltage is back.

## IMU (`robot/imu.py`, done 2026-10-29)
- 100 Hz, chip low-pass 44 Hz, +-1000 deg/s (floor spins reach 500 deg/s), +-4 g. Config is read back after writing.
- Axis mapping `config.IMU_AXES` turns sensor axes into robot axes (board upside down);
  must be a rotation (mirror images are rejected). Measured by `hwtest/imu.py --only axes`.
- Gyro bias measured at start (2 s, robot still; restarts if it moved).
- Publishes `imu` (ImuReading: yaw, yaw rate, roll, pitch, tilt, tipped, bump, accel, gyro, temp)
  and `imu_status`. Yaw: trapezoid integration of the rotation about the vertical, + = left, not wrapped,
  drifts slowly. Tilt: an 'up' vector rotated by the gyro and pulled toward gravity (complementary
  filter; gravity ignored while |a| is outside 0.85-1.15 g) - stays right during fast spins on a tilted floor.
- Driving: start softly (speed ramp, ~2 /s). A 0 -> 0.5 jump lifts the nose ~15 deg (weight sits on the
  wheel axle, little on the caster).
- Bumps: motor start/stop jerks read 0.5-1.1 g (steady driving <= 0.26 g, threshold 0.6 g), so `safety.py` must
  ignore bumps for ~0.3 s after every motor command change (as `hwtest/base.py` does).
- Faults: single I2C glitch skipped; 5 in a row or 25 identical frames (chip reset after a
  brown-out) -> status `error`, no new `imu` (goes stale -> safety stops), re-init at most 1/s.

## Build order
1. `state` + `motors` + `safety` (+ the existing keyboard drive rewritten on top of them).
2. `distance` → obstacle stop.
3. `imu` → straight driving and exact turns.
4. `pan_tilt`, `camera`.
5. `audio`, then voice / LLM.

## Navigation (`code/nav.py`, written 2026-11-04, simulation only)
Its own program (alive.py stopped) on top of `Robot`: a 20 Hz loop feeds `robot/nav.py` (pure logic: a Percept in, a
Command out - speed + turn rate for `motors.request_velocity`, head angles, "clear the bump stop") from threads for the
head (50 Hz, remembers where it pointed when), the sonar (12 Hz), the camera (floor check every picture,
`robot/floor.py`) and the pose model (people mode). The same brain runs in `sim/navworld.py`. Safety stays below it.
Details: `docs/navigation.md`.

## Alive robot, phase A: eyes + head life (`code/alive.py`, written 2026-10-31, not run on the robot yet)
Layers (each its own module, no module waits for another):
- **camera** (`robot/camera.py`): one sensor mode 2304x1296, two streams - lores 576x324 (find faces) and main
  2304x1296 (cut faces out to recognise them, 4x the pixels). `FaceExposure` steers ExposureValue so the face
  (not a ceiling lamp) is well exposed, and "squints" (+2 EV for 0.8 s every 3 s) when no face was seen for 3 s,
  to find faces too dark to detect. Sensor HDR switch (`CAMERA_HDR`, off until tried).
- **vision** (`robot/vision.py`): YuNet on lores -> tracker (head angles, not pixels) -> for tracks that need a
  look: crop from main, SCRFD landmarks on the crop, align, MobileFaceNet, match, vote.
- **tracker** (`robot/tracker.py`): identity = majority of the last 3 looks; one very clear look decides a new
  track; "unknown" only when all looks match nobody and never for a track that was recognised.
- **people** (`robot/people.py`): several vectors per person (new angles learned from confident matches),
  enrol only from big frontal faces. Duplicates: a track that matched two ids >= 2x each = same person (merge
  when the sets also look alike); two ids in one frame = never the same; similar sets without evidence ->
  `ask_same` event (for the voice phase). Every vector keeps its origin id: merges can be undone.
- **attention** (`robot/attention.py`): follow a face (known > unknown, closer first, sticky), glance at
  newcomers, hold / search when lost, idle wandering; suggests a body turn when the face stays > 45 deg aside.
- **head life** (`robot/head_life.py`): 50 Hz critically damped spring toward the focus (soft when following,
  fast saccades for big jumps), micro-saccades, breathing, blink dips; remembers its pose for 3 s so a camera
  frame is matched with where the head pointed when it was taken.
Threads: head (50 Hz) + the perception loop (camera rate). Motors are not used yet.

