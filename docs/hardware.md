# Arcby — hardware / bill of materials

Measurements marked (M) were measured by Selim; others are datasheet values.

## Drive
| Part | Details |
|---|---|
| 2 × yellow TT gear motor | (M) box 22 mm tall, 18 mm thick (22 with shaft stub), mounting holes 3 mm Ø, 17.5 mm apart, 22 mm from hole to shaft, shaft 5 mm with flats, shaft centre 11 mm from bottom, rear pin/clip stub 8.5 mm, shaft 12 mm from gearbox end. 3–6 V, < 1 A. Re-wired with ~15 cm thin wire (soldered, tested). |
| 2 × wheel | (M) 60 mm Ø, 12 mm wide, tyre 5 mm, 7 mm gap to motor |
| Steel ball caster | (M) 23.5 mm tall without spacers, holes 31 mm apart (M3). Mounted on a printed pad (19.5 mm) so the plate sits ~43 mm above the floor front and back (verified). |
| L298N dual H-bridge | 43 × 43 mm, holes 37 mm square, 3 mm. (M) ~25–26 mm tall incl. heatsink; on 5 mm printed spacers → top ≈ 31 mm above plate. |

## Brain / sensors
| Part | Details |
|---|---|
| Raspberry Pi 4B | holes 49 × 58 mm, M2.5. Raspberry Pi OS 64-bit, SSH. Camera tested OK. |
| Armor Lite heatsink + PWM fan | sits on the Pi, fixed with screws from the Pi's back. Thermal pads. GPIO header stays free. Fan: 3 wires (red 5 V, black GND, blue control). Sheet says blue -> GPIO14; on Arcby it goes to **GPIO4 / pin 7** instead (see `pinout.md`). |
| Pi Camera Module 3 **Wide** | same size as Camera Module 3, fits the pan-tilt holder. Check ribbon length (Pi on upper deck, camera on pan-tilt). |
| Arducam **B0283** pan-tilt (upgraded) | NOT an SG90 kit. 2 × GH-S37D digital servos (**3.6–4.8 V**, < 350 mA), PTZ control board 38 × 28 mm (I2C, 12-bit PWM, 3–5 V) inside the base. (M) base 43 × 30 mm, total height 65 mm, mounting ears with M2.5 holes: 26 mm c-c (long direction) × 36 mm c-c (across the ears, holes 3 mm outside the long sides). Cables exit left and right of the base. Tilt cannot hit the Pi (Selim confirmed). |
| B0283 PTZ board connector | 4 male pins, labels printed on the board **underside** (not visible when mounted). Order **left to right: VCC, SDA, SCL, GND** (Selim read it 2026-10-30). VCC also feeds the servos -> 4.7 V rail only. Chip = PCA9685 at I2C 0x40 (+0x70 all-call); servo PWM 50 Hz: **ch0 = tilt** (1300 us down, 1700 us up), **ch1 = pan** (longer pulse = RIGHT; the 2026-10-30 note "1700 us = left" was wrong); level camera at tilt servo ~122 deg with the robot on the floor (90 = ~30 deg down); straight ahead at pan servo ~64 deg (90 = ~26 deg right); measured 2026-10-31 with camera photos. |
| HC-SR04 ultrasonic | **Header pins bent 90° to the rear** so the female jumpers run horizontally backwards (straight-down pins + jumpers would hit the base plate / the area below the sensor). |
| HC-SR04 (details) | 3 chips on the back (looked like the old 5 V version), but **tested 2026-10-29 on 3.3 V: works, correct readings up to 256 cm, spread < 0.2 cm**. Runs on Pi 3.3 V with ECHO direct - no divider. (M) fits the printed plate (16.4 mm holes, 26 mm spacing) snugly. |
| MPU6050 (GY-521) 6-axis IMU | (M) board 20 × 15.5 mm, 2 × M3 holes (3 mm) 16 mm c-c (caliper) along the long edge opposite the pins, 2.5 mm from that edge; tallest chip-side part 3.3 mm; header pins soldered, 8.5 mm. I2C address 0x68. Mounted **upside down** (chip down, pins up) on the printed `MPU6050 Mount v2` with 2 × M3×10, centred just in front of the L298N (near the wheel axis), pins facing the L298N → axes are inverted in software: sensor +X = robot forward, sensor +Y = robot right, sensor +Z = down (measured 2026-10-29). Uses: straight-line driving correction (gyro Z), accurate turns, tip-over / bump detection. No magnetometer (no absolute heading); gyro drifts slowly. |
| MAX98357A I2S amp | VIN 5 V. 7 male header pins, **soldered** (LRC, BCLK, DIN, GAIN, SD, GND, VIN); GAIN and SD stay unconnected (9 dB, left+right mix). Plugged straight into the breadboard under the deck (rows A6 VIN .. A12 LRC, outermost holes, board sticking out toward the speaker; see `wiring.md`), next to the speaker; the other edge has the 2-screw speaker terminal with a mounting hole on each side of it (not used; the header pins in the breadboard hold the board). |
| INMP441 I2S MEMS mic | bought with header pins already soldered. Bottom-port: the hole in the PCB faces **up** toward the deck's 2.5 mm sound port. VDD **3.3 V**. L/R → GND. |
| Speaker 20 × 30 mm 8 Ω | round driver in a rectangular frame, adhesive ring on the front; fits the deck frame exactly (M). |

## Power
| Part | Details |
|---|---|
| Zeee 2S LiPo (2600/2700 mAh) | 7.4 V / 8.4 V full. (M) ~105 × 33.5 × 18 mm, ~107–108 mm incl. lead exit. Thick pre-tinned red/black leads with **bare ends, ~10 cm** out of the rear notch; white JST-XH balance plug (used for charging + LiPo alarm). |
| 2 × SANEC LM2596 buck | 43 × 21 × 14 mm, 2 corner holes (diagonal). **Solder pads, no screw terminals** (IN+/IN−/OUT+/OUT−); the servo-buck feed is a second wire soldered onto the Pi buck IN+/IN− pads. #1 "brain" → 5.1 V → Pi USB-C (on base, zip-tied). #2 "muscle" → **4.7 V** → pan-tilt servos (upper deck, double-sided foam tape). |
| KCD1 mini rocker, 2-pin | cutout 19.2 × 13.2 mm, clicks in (verified). No polarity. Solder tabs. |
| Inline fuse holder ARDUINO OMV13 5×20 | came as a wire loop — cut in the middle. Held on the lower deck with a rubber band. |
| Fuses KGPart 5 A F5AL 5×20 glass (10 pcs) | fast-blow; if robot dies suddenly, check the fuse. |
| 3 × Wago 221-413 | (M) ≈ 18 × 18.6 × 8.4 mm. 3 holes are all connected (like a breadboard column). Strip 11 mm. |
| T-plug (Deans) pair | **not used** (soldering to battery failed). Horizontal blade = +, vertical = −. Kept for later. |
| LiPo voltage alarm | on the balance plug. |
| 18 AWG silicone wire | red/black for the power path. Thinner wire for motors/signals. |

## Fasteners / consumables
- M3 YHB screw set: motor brackets (M3x25 + nut), L298N (M3x10 into plate), pillars (M3x10), caster (self-tap into 2.8 mm pilots).
- On hand: SYB-170 mini breadboard (45 x 35 mm, adhesive back) -> under the upper deck; 20-30 female-male jumpers.
- **To buy:** 4 × M2.5 × ~14–16 mm (Pi + Armor Lite through deck + spacer; confirm with the kit's short screw length), heat-shrink tubing, thin zip ties (2.5 mm), double-sided foam tape, flux paste if soldering thick wire again. Optional: 4-channel BSS138 I2C level shifter (PTZ board pulls SDA/SCL to 4.7 V, measured 2026-10-30; running direct for now, see FLAGS).
- Tools: soldering iron (wattage unknown — thick LiPo lead would not take solder), multimeter, female-female jumpers (10 cm + 20 cm).
- Ordered 2026-10-27: WellHise ETK-10 iron stand, WellHise 45×30 magnetic silicone mat (blue).
- Ordered 2026-10-30 (amazon.com.tr): 2 x Arduino FC-33 motor speed sensor (LM393 slot opto, 3.3-5 V) = wheel encoders, run on 3.3 V; 20-slot discs (black PLA) still to design + print. LiDAR: only RPLIDAR A1M8 (8,822 TL, ~10 cm wide) on amazon.com.tr - not bought; ArUco wall tags first.
- In the cart 2026-10-30 (Selim to order, arrive 3-4 Nov): YSB M2.5x16 (DIN 7985, 20 pcs) for the Pi, YSB M2x16 (20 pcs) spare. The M2.5 bought on 2026-10-26 were only 6 mm (pan-tilt). Armor Lite own screws are 7.5 mm -> Pi needs 7.5 + 9 = ~16 mm from below.
