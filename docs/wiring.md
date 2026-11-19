# Arcby — wiring

## Power path (current design, no T-plug)
```
Battery + (red, ~10 cm) ──► [+ Wago] ──► fuse 5 A ──► KCD1 switch ──► [Wago 3] ──┬─► L298N 12V
                                                                               └─► Pi buck IN+ ─┬─► (OUT 5.1 V) ► Pi USB-C
                                                                                                └─► (IN+ tap, 25 cm) ► servo buck IN+
Battery − (black, ~10 cm) ──► [− Wago] ──┬─► L298N GND ◄── Pi GND (pin 14, jumper)
                                         └─► Pi buck IN− ──► (IN− tap, 25 cm) ► servo buck IN−
```
- Servo buck output **4.7 V** -> breadboard 4.7 V rail (col 1/2 bottom) -> PTZ board VCC/GND (servos are fed from that VCC). Nothing else uses the 4.7 V rail.
- Fuse stays **out of its holder** until all wiring is finished and checked.
- To remove the battery: switch OFF, open the two Wago levers, tape both bare ends separately.
- LiPo alarm on the balance plug whenever the switch is on: black wire on the alarm's "-" pin, plug uses only the first 3 pins (-, 1, 2). Threshold 3.5 V/cell (= 7.0 V, where the Pi buck starts to sag). Charge at once when it beeps; never below 3.0 V/cell.

## Wago map (3 × Wago 221-413; all 3 holes of one Wago are connected)
| Wago | Where (model) | Hole 1 | Hole 2 | Hole 3 |
|---|---|---|---|---|
| **+ Wago** | own 1-Wago tray, left of the L298N (front half, clear of the motor wire hole), entries facing the L298N | Battery red | Fuse holder lead A | spare |
| **− Wago** | own 1-Wago tray, right of the L298N (front half, clear of the motor wire hole), entries facing the L298N | Battery black | L298N GND (12 cm black) | Pi buck IN− (black) |
| **Wago 3** (switched +) | front-centre Wago tray (1-Wago tray v3) | from switch (15 cm red) | L298N 12V (12 cm red) | Pi buck IN+ (10 cm red) |

Battery leads reach the two side Wagos by passing **under the L298N board** (5 mm gap) and exiting between the rear and front spacers.
Each Wago sits in its own printed 1-Wago tray (`Wago Tray v3 x3.3mf`), glued/taped to the plate; open end of each tray faces the wires.

## Switch / fuse
- Fuse holder lead B soldered to one switch tab; 15 cm red soldered to the other tab → Wago 3.
- Insulate each tab separately (heat-shrink or tape). Switch body hangs below the plate;
  wires come up through the left cable hole.

## L298N
| L298N | Connection |
|---|---|
| 12V | Wago 3 |
| GND | − Wago **and** Pi pin 14 (cut one jumper end, put both wires in the same screw terminal) |
| 5V | **nothing** (keep 5V-EN jumper on; never feed the Pi from here) |
| ENA / ENB | jumpers **removed**; the signal pin (closer to the IN pins) goes to the Pi |
| OUT1/OUT2 | **left** motor |
| OUT3/OUT4 | **right** motor |
If a wheel spins backwards, swap that motor's two wires. If "left" turns the right wheel, swap motors.

## First power-on checklist
1. Multimeter continuity between + and − side (e.g. Wago 3 vs − Wago) with switch ON: no constant beep.
2. Switch OFF, insert fuse.
3. Switch ON: L298N red LED + buck LED on; no smoke/smell/heat.
4. Pi buck OUT = 5.1 V. Switch OFF.

## Signals (Pi ↔ modules) — see `pinout.md`
- L298N: 6 signals + GND (pins 11–22 block), female-female jumpers down through the upper deck's left slot.
- Audio (I2S): amp plugged into the breadboard; BCLK (pin 12) and LRCLK (pin 35) shared with the mic in the amp's breadboard columns (no Y jumpers).
- HC-SR04 on **3.3 V** (tested 2026-10-29: correct readings up to 256 cm): VCC from breadboard col 4 (3.3 V, Pi pin 1), GND Pi pin 30, TRIG GPIO5 (pin 29), ECHO **direct** to GPIO6 (pin 31). No divider: ECHO never goes above the sensor's own VCC. **Never** power it from 5 V / 4.7 V while ECHO is direct.
- MPU6050: VCC from breadboard col 4 (3.3 V, Pi pin 1), GND Pi pin 25 (direct), SDA/SCL via breadboard cols 2/3. No soldering; ~4 mA.
- Pan-tilt PTZ board (4 pins, left to right **VCC, SDA, SCL, GND**; single VCC): VCC + GND from the 4.7 V rail on the breadboard, SDA/SCL from breadboard cols 2/3. **Never on Pi 3.3 V** (tried 2026-10-28: the servos dragged the Pi into under-voltage).

## Old design (for reference)
Originally: battery → female T-plug | male T-plug (in the printed plug tray) → fuse → switch → two
Wagos in the front tray. Abandoned 2026-10-27 because the thick pre-tinned battery lead would not
take solder (no flux / weak iron). Plug tray now unused.

## Breadboard (SYB-170 mini, 45 x 35 mm, under the upper deck)
Stuck (adhesive back) to the **underside of the upper deck, centred, front half** (x -17.5..17.5, y 12.5..57.5 mm),
between the pan-tilt screws and behind the HC-SR04 jumpers, clear of every deck screw hole
(model body `Breadboard Dummy (SYB-170, under deck centre)`, params `bb_*`). Holes face down; jumpers hang ~15 mm. Keep the front ~4 columns (over the HC-SR04 jumpers, only ~10 mm below) for short or no jumpers.
Plug everything at the desk **before** the deck goes onto the pillars.
Columns are named by position because the printed numbers were not checked when it was stuck on:
**A1 = rearmost column (Pi side) ... A17 = front (HC-SR04 side).** Halves: **LEFT** (robot left, speaker side)
and **RIGHT** (robot right, servo-buck side); each column-half = 5 connected holes. With the deck upside down
on the desk, robot-left is on your right.

| Column | Half | Net | From | To |
|---|---|---|---|---|
| A1 | RIGHT | **4.7 V** | servo buck OUT+ (soldered lead with a male pin) | PTZ VCC |
| A2 | RIGHT | GND (servo) | servo buck OUT- | PTZ GND |
| A2 | LEFT | SDA | Pi pin 3 | MPU SDA, PTZ SDA |
| A3 | LEFT | SCL | Pi pin 5 | MPU SCL, PTZ SCL |
| A4 | LEFT | **3.3 V** | Pi pin 1 | MPU VCC, HC-SR04 VCC |
| A6 | LEFT | **5 V** | amp VIN (plugged in) | Pi pin 2 |
| A7 | LEFT | GND | amp GND (plugged in) | Pi pin 34 |
| A8 / A9 | LEFT | amp SD / GAIN | amp | nothing (9 dB, L+R mix) |
| A10 | LEFT | amp DIN | amp DIN (plugged in) | Pi pin 40 |
| A11 | LEFT | BCLK | amp BCLK (plugged in) | Pi pin 12, mic SCK |
| A12 | LEFT | LRCLK | amp LRC (plugged in) | Pi pin 35, mic WS |
| A14-A17 | both | spare | | only ~10 mm below (HC-SR04 jumpers): short parts only; BSS138 level shifter here if needed |

**MAX98357A amp plugged straight into the breadboard** (2026-10-30, Selim): its 7 header pins in the LEFT half,
in the **outermost hole** of rows A6 (VIN) ... A12 (LRC); the board sticks out past the breadboard edge toward the
robot's left (under the speaker, ~6 mm below it). The 4 holes of each row on the channel side stay free for jumpers.
Speaker wired to the amp terminal (red +, black -; BTL output: never tie either speaker wire to GND).

Everything else stays direct Pi <-> module (female-female): MPU GND (pin 25), HC-SR04 TRIG/ECHO/GND (29/31/30),
mic SD/VDD/GND (38/17/39), fan (4/6/7), motors.
Jumper count: about 15 female-male (Pi -> breadboard 5, modules -> breadboard 10) + 2 leads soldered to the servo buck OUT pads.
I2S clocks: keep those jumpers short (10-20 cm is fine).

## Jumper colours (as installed, 2026-10-30)
| Module | Wire | Colour | Goes to |
|---|---|---|---|
| Mic (INMP441) | SD | purple | Pi pin 38 |
| Mic (INMP441) | VDD | blue | Pi pin 17 (3.3 V) |
| Mic (INMP441) | GND | green | Pi pin 39 |
| Mic (INMP441) | SCK / WS / L/R | - | breadboard A11 / A12 / A7 (LEFT = right half when the deck is upside down) |
