# docs/

Engineering notes, written while building. They record what was measured, what was tried, what failed and why the
current design won, with dates.

| Doc | Read it for |
|---|---|
| [`architecture.md`](architecture.md) | one process vs ROS 2, the load budget on a Pi 4, module rules, motor control and safety, the IMU filter |
| [`vision.md`](vision.md) | the face pipeline, model bake-offs (speed / range / recognition), gaze, telling photos from people, bodies and hand signs, threshold from the robot's own data |
| [`ears.md`](ears.md) | the mic chain, STT comparison (WER), speaker verification bake-off, latency breakdown, barge-in |
| [`navigation.md`](navigation.md) | sensor blind spots, the camera floor check, local memory, the brain's states, simulation and first real runs, known limits |
| [`hardware.md`](hardware.md) | bill of materials with measured dimensions |
| [`wiring.md`](wiring.md) | power path, distribution, H-bridge, breadboard map, first power-on checklist |
| [`pinout.md`](pinout.md) | Raspberry Pi GPIO allocation and why (boot states, audio overlay) |
| [`printing.md`](printing.md) | printer settings, lessons, part list |
| [`fusion.md`](fusion.md), [`parameters.md`](parameters.md) | how the CAD model is organised; every parameter |
| [`dashboard.md`](dashboard.md) | the status dashboard in detail |
| [`images/`](images/) | README photos (cleaned of EXIF/GPS) |
