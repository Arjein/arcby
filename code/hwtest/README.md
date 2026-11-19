# hwtest/: hardware tests, calibrations, bake-offs

One script per hardware unit, guided and interactive (the script tells the operator what to do, then measures).
Reports go to `~/scratch/hwtest/` on the robot. Nothing about the hardware was guessed: every number in
`robot/config.py` comes from one of these scripts.

## Bring-up and units

| Script | Checks |
|---|---|
| `bringup.py` | every module after power-on, in a safe order: power flags, `config.txt`, I²C bus, IMU, sonar, pan-tilt, camera, speaker, mic. `--sim` = no hardware |
| `motors.py` | L298N wiring and direction per wheel, ENA/ENB, start (stall) PWM per wheel and direction |
| `imu.py`, `imu_level.py` | MPU6050 identity, self-test, the **axis mapping** (tilt the robot, read which sensor axis moved), level trim from gravity |
| `distance.py` | HC-SR04 range and spread against a tape measure |

## On the floor

| Script | Measures | Result |
|---|---|---|
| `base.py` | pivot start speeds, the spin curve (turn rate vs PWM), straight-line drift, IMU-closed 90° turns | turn error ≤ 2.4° on carpet |
| `route.py` | speed calibration, then a 100 × 50 cm rectangle with heading hold; closure error at the start mark | closes within 16 cm, with no encoders |
| `navcal.py` | navigation's crawl and cruise speeds, measured by the sonar while driving at a wall | 38.3 / 40.2 cm/s |

## Model bake-offs (choose models on the robot's own data)

| Script | Question | Outcome |
|---|---|---|
| `faces.py` | which face detector / embedder: speed on the Pi, range, recognition on LFW under synthetic degradations (small, dark, noisy) | YuNet + ArcFace R50 int8 |
| `faceshots.py` | voice-guided face shots at 1 / 2 / 3 m, backlit and at night: do full-resolution crops beat the video stream? | yes: recognise from the 2304×1296 stream |
| `gazeshots.py` | labelled "look at me / at the phone / away" frames: landmark gaze vs gaze CNNs | landmarks, AUC 1.00 at 1–2 m |
| `earshots.py` | speech-to-text on the robot's mic: local Whisper sizes vs cloud models, WER and name accuracy | cloud STT, 0 % WER with a names hint |
| `mouthshots.py` | can mouth motion tell who is talking? | 62 % ≈ chance at 3 m → dropped in favour of face + voice |
| `signs.py` | the hand-sign detectors replayed on hours of recorded pose output: false "come" count | target 0 |
| `showshots.py` | frames of things held up to the robot, for the showing detector | |
| `bargecal.py` | per-band echo coupling of *this* speaker in *this* body, echo lag, replayed false barge-ins before / after | run once after fitting a speaker |

`common.py` holds the shared prompts, result records and report writer.
