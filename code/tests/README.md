# tests/: unit tests with fake hardware

```bash
python3 -m unittest discover -s code/tests -t code      # 1044 tests, ~100 s, any laptop
ARCBY_SLOW_TESTS=1 python3 -m unittest discover -s code/tests -t code   # + every hwtest script in --sim
```

Standard library `unittest` + NumPy only. The motor-driver tests also need `gpiozero`. The tests always use the
example household (Emre / Meltem / Halil / Doğu); `__init__.py` forces it, so no real name can end up in a fixture.

## How hardware is faked

| Fake (`fakes.py`) | Level |
|---|---|
| `FakeMpu6050` | **register level**: the real `imu.py` driver writes its config registers and reads raw bytes, so axis mapping, scaling and the fault paths are all tested |
| `FakePca9685` | the servo board's registers; the tests check pulse widths and limits |
| `FakeDriver` | the H-bridge pins; records every PWM the motor code applied |
| `FakeClock` | time is injected everywhere, so 10 s of a 50 Hz loop runs in milliseconds and deterministically |

**Closed-loop tests.** `test_motors.py` has a `Plant`: a first-order lag model of the robot's yaw response, with
a start-PWM dead zone, a weaker wheel, and the measured carpet spin curve (stuck below 0.24 PWM, then ~250 °/s
jumping up), fed back through the fake IMU. The velocity controller is tested on what it actually achieves (the
turn rate and heading it reaches), not on its internal numbers.

## What is covered

- **Hardware modules**: IMU, motors, safety (every stop reason, latching, the bump calm-window rule), sonar,
  pan-tilt, power, the I²C wrapper.
- **Perception**: alignment geometry, NMS, the tracker gate and votes, the people store (merge / unmerge / picture
  flag), static spots, gaze, body tracks, hand signs on synthetic keypoint tracks, showing.
- **Hearing and voice**: VAD resets, AGC, the resampler, utterance cutting, barge-in band math, voice prints, live
  STT over a fake WebSocket, TTS cache and budgets, speakerd's protocol.
- **Behaviour**: attention, head life, the greeter, conversation rules, the mind's sentence streaming, learning jobs.
- **Navigation**: the brain on simulated rooms (`sim/navworld.py`), the floor projection geometry, the local map,
  follow mode, recordings.
- **Web pages**: the dashboard and drive APIs, plus `page_harness.js`, which runs the drive page's real JavaScript
  under Node with a pretend browser and checks what it sends (key releases, a hidden tab = stop).
- **End to end**: `test_system_sim.py` runs the real `Robot` core (IMU + safety + motor threads) and the drive server
  on a simulated robot, driven over real HTTP like a browser would.

The conversation-level scenarios live in [`../sim/`](../sim/).
