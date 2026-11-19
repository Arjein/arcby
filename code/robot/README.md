# robot/: the robot package

~50 modules, one job each. Hardware access stays inside its module, so everything above it can be tested with fakes.
Modules run in their own threads at their own rates and talk through `state.py` (the latest value of every reading +
the time it was taken). No module waits for another, and stale data counts as missing.

```
 hardware        perception             behaviour               actuation
 ────────        ──────────             ─────────               ─────────
 camera ──────►  faces → tracker ──┐    attention ─────────►    head_life → pan_tilt
                 body → gestures   ├─►  greeter, show           tts → speakerd
                 gaze, static_spots│    conversation → mind ─►  (voice)
 mic ─────────►  vad → stt/live_stt├─►  legs → nav ────────►    motors ◄── safety
                 voices, bargein   │       ▲ floor, localmap
 imu, sonar ──►  state ────────────┘       │
 power ───────►  breadcrumb, recorder, sensors (datasets)       learning (wall power)
```

## Hardware

| Module | Notes |
|---|---|
| `i2c.py` | register access through `/dev/i2c-*` with `ioctl`, standard library only |
| `imu.py` | MPU6050 at 100 Hz: axis remap (the board is mounted upside down; the mapping must be a proper rotation), gyro bias at start, trapezoidal yaw integration, complementary-filter tilt, fault detection (a chip reset after a brown-out repeats identical frames) |
| `motors.py` | the **only** code that drives the L298N. Wheel mode or velocity mode (feed-forward + PI on the gyro's yaw rate), dead band, slew-rate-limited soft start, 0.3 s watchdog |
| `safety.py` | 50 Hz verdict: IMU stale, tipped, bump (only out of calm driving), low battery. Latched stops; a black-box CSV of the last 3 s on each |
| `power.py` | no battery ADC, so the Pi's own under-voltage flag (the buck sags at ~7.0 V on the 2S pack) stands in for "battery low". `battery_on()` = a read-only probe of the servo board |
| `pan_tilt.py` | PCA9685 servo PWM, angle ↔ pulse calibration, hard limits, smooth moves |
| `sonar.py` | HC-SR04 timed from **kernel GPIO edge timestamps**, not Python wall clocks (a 1 ms scheduling jitter = 17 cm of error) |
| `camera.py` | one sensor mode, two streams (576×324 to detect, 2304×1296 to recognise). Exposure is steered by the face, not the ceiling lamp; it "squints" (+2 EV) to look for dark faces. Background phase correlation measures the real head pose |
| `speakerd.py` | the speaker as its own process: ALSA streams silence forever, so the amplifier never pops |
| `main.py` | `Robot`: starts IMU / safety / motors / power; a crash in any thread stops everything |

## Perception

| Module | Notes |
|---|---|
| `faces.py` | YuNet detect → SCRFD landmarks + quality → similarity-transform alignment → ArcFace R50 int8 embedding (512-d). NMS and alignment written in NumPy |
| `tracker.py` | tracks in head angles; gate scaled by the track's own face size; identity = majority of the last 3 looks |
| `vision.py` | ties them together. The embedder (~0.4 s per face) and the gaze refiner run in worker threads; results are applied to the track they were cut for, and late results are dropped |
| `people.py` | several vectors per person (new angles learnt from confident matches). Duplicate detection: one track matching two ids repeatedly = same person; two ids in one frame = never the same. Every merge can be undone |
| `static_spots.py` | learns spots where non-faces keep appearing and checks them rarely (silenced 78 % of the fake tracks and no real face) |
| `gaze.py` | head yaw/pitch from 5 landmarks → engagement, with hysteresis (on 0.5 / off 0.3) |
| `body.py` | YOLO26n-pose: tracking, face hints ("legs seen → the face is up there"), wave detection by frame differencing at the wrist, crops of the big stream for people far away |
| `gestures.py` | "come" (beckon) and "stop" signs from forearm motion, in shoulder-width units |
| `show.py` | a hand held up and still + an object detector (YOLO26n) on a crop around the wrist = "someone is showing me something" |
| `vad.py`, `mic.py` | Silero VAD on 32 ms chunks; high-pass, AGC that learns from speech only, peak limiter, 48→16 kHz FIR resampling, pre-roll, utterance end after 0.45 s of silence |
| `stt.py`, `live_stt.py`, `wsock.py` | speech-to-text: streamed over our own minimal RFC 6455 WebSocket client (stdlib only) while the sentence is spoken; a clip-based backend as fallback |
| `voices.py` | speaker verification: CAM++ 192-d embeddings from Kaldi-style filterbanks, k-NN over up to 40 prints per person + a margin |
| `bargein.py` | per-band echo power prediction from what speakerd played; self-tuning on false alarms |

## Behaviour

| Module | Notes |
|---|---|
| `attention.py` | pure logic: whom to look at (known > unknown, closer, looking at it), glances at newcomers, boredom, eye-contact breaks, turning to voices, a calmer head at night |
| `head_life.py` | 50 Hz critically damped spring towards the target, minimum-jerk moves for big jumps, micro-saccades, breathing, blinks, nods at pauses while listening. It remembers its pose for 3 s, so each camera frame is matched to where the head pointed when it was taken |
| `conversation.py` | who is talking to the robot (the face looking + the voice agreeing), meeting new people, ignoring side remarks, its own echo and coughs; the robot's name as the speech-to-text mishears it ("Arif Bey", "Ark" → Arkbi) |
| `mind.py` | the LLM (Claude Haiku), streamed sentence by sentence; a rule-based fallback; a daily budget |
| `tts.py` | Google Cloud TTS first, Gemini streamed TTS as fallback; 200 MB cache, daily budgets, a hard cap on the voice level |
| `greeter.py` | greets known people on eye contact or a wave, not every time it sees them |
| `nav.py` | navigation brain: percept in → command out (scan / choose / turn / drive / follow / help). Pure, so it runs unchanged in the room simulator |
| `floor.py` | floor segmentation by colour histograms + inverse-perspective ground projection + ray casting; rug-edge detection |
| `localmap.py` | short-lived obstacle memory in odometry coordinates; each sensor clears only what it can see |
| `legs.py`, `drive_body.py` | driving inside `alive.py` (follow / explore / wander, every stop counts); hardware glue shared with `code/nav.py` |

## Data and learning

`recorder.py` + `sensors.py` write every stream into size-capped datasets; `collect.py` keeps face crops;
`nav_rec.py` records drives; `breadcrumb.py` writes one fsynced line a second, so a power cut can be explained on
the next boot; `learning.py` + `learn_faces.py` re-fit the face-match threshold from same-track / same-frame pairs
in the recordings (positive and negative pairs come for free, no hand labels needed); `commands.py` carries out the
dashboard's edits inside the running process, so the people store has a single writer.

## Configuration

`config.py` holds every pin, bus address, mounting measurement and threshold, each with the measurement it came
from. Keys listed in `config.TUNABLE` may be overridden by learning mode through `tuned.json`, and only within
their limits.
