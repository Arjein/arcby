# Arcby — 3D printing

Printer: **Bambu Lab A1** (256 mm bed), PLA, Bambu Studio on the Mac.
Process preset **"Robot parts"**: 3 walls, 25 % infill, supports OFF.
Print files are exported to `cad/print-files/` in this repo and opened in Bambu Studio from there.

## Rules learned
- Print small **test coupons** before big parts (motor bracket, caster pad, rails, switch, pan-tilt holes, ultrasonic plate all were tested first).
- Plates with features on both faces need supports → move features from one face into separate parts
  (L298N spacers, Pi spacers, trays) and print the plate **flipped** (top face on the bed).
- Bambu "floating cantilever" warning on the base → fixed by opening the motor pin slot to the bracket edge.
- Holes: 2.8 mm pilot = M3 self-tap, 3.4 mm = M3 clearance, 2.2 mm = M2.5 self-tap, 2.8 mm = M2.5 clearance.
- Printed fits: 33.5 mm battery rail gap = slight 0.5 mm play (OK). 16 mm T-plug pocket was 2 mm loose → 14 mm.
- Tall thin parts (pillars 8 × 40 mm): brim (Others → Bed adhesion → Brim type: Outer brim only, 3–5 mm).
- Dimension sketches from the projected **origin**, never from projected edges of other features
  (stale references broke the caster holes on the first base print).

## Part list / status
| File | Status |
|---|---|
| `Base Plate - FLIPPED (print this).3mf` | printed (final: caster holes 31 mm, wide 34 mm lead notch) |
| `Pillars x4.3mf` | printed |
| `Fit Test + Small Parts.3mf` / L298N spacers | printed |
| `Trays v2 (Plug 14mm + Wago x2).3mf` | printed (plug tray now unused) |
| `Wago Tray v3 (1 Wago).3mf` | single-Wago tray (19.1 mm pocket) |
| `Wago Tray v3 x3.3mf` | three single-Wago trays on one plate (one per Wago: +, -, switched +) |
| `Upper Deck - FLIPPED (print this).3mf` | v1, printed, used for fit tests (pan-tilt, Pi, pillars all OK) |
| `Upper Deck v2 - FLIPPED (print this).3mf` | **final upper deck** (adds HC-SR04 front plate) — printing |
| `Pi Spacers x4.3mf` | printing |
| `MPU6050 Mount v2.3mf` | IMU mount v2: 22.4 × 16 mm base (glue/tape centred just in front of the L298N, ~4 mm from its front edge), two M3 posts 16 mm apart: 6 mm Ø lower part + 4.4 mm Ø × 4 mm top collar (only touches the ring around the hole, clears the 3.3 mm chip-side parts), 2.8 mm pilots. Post height sized for **M3×10** (ends 0.6 mm above the base bottom). Base-down, no supports. v1 (thick posts, rests under the header) is obsolete. |
| `Test - *.3mf` | fit coupons (done) |
| _Base v2 encoder parts below: **parked 2026-11-04, do not print** (v2 moved to N20 encoder motors)_ | |
| `Test - Encoder Hub Fit.3mf` | 3 hubs with the TT double-D bore, clearance 0.1 / 0.2 / 0.3 mm (1 / 2 / 3 dots): push each on the motor's inner shaft stub, pick the tightest one that goes on by hand -> `enc_fit`. Flat as exported, any colour, ~10 min. |
| `Encoder Disc.3mf` | wheel encoder disc: 40 mm, 20 open slots (r 10-20 mm), 2.5 mm, hub 11 mm with a double-D bore (`enc_fit` 0.2) 8 mm deep. **Black PLA** (thin PLA lets IR through), disc on the bed, hub up, no supports. Print 2 (one per wheel) after the hub fit test. |
| `Test - Encoder Sensor Rig.3mf` | bench rig = the Base v2 geometry: wall = motor face, 3.4 mm round pin the disc spins on (25 mm up), shelf the FC-33 board screws under (fork down, M3 slots +-3 mm, 2.5 mm standoffs), arm tips 7 mm from the axle. Print **lying on the wall's outer face** (pin up), no supports. Spin the disc by hand: the board's OUT LED must blink 20x per turn. |
