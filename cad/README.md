# cad/: mechanical design

| File | What |
|---|---|
| `Arcby.f3d` | Fusion 360 archive of the whole robot (snapshot of the live cloud design) |
| `Arcby.step` | the same model as STEP, for any CAD tool |
| `print-files/` | every printed part as 3MF, ready for Bambu Studio |
| `LICENSE.md` | CC BY 4.0 |

## How the model is built

- **One file, one component, many bodies.** Bought parts (Pi, L298N, motors, bucks, battery, pan-tilt) are modelled
  as measured "Dummy" bodies, so every clearance is checked against the real part in the model, not on the printer.
  Even the cables are modelled as visual pipes, so their routing can be checked too.
- **Fully parametric.** 155 user parameters ([`docs/parameters.md`](../docs/parameters.md)) drive the geometry, and
  derived values are expressions: `caster_pad_height = bracket_height / 2 + wheel_dia / 2 - caster_height` keeps the
  plate level whatever wheel or caster is fitted. A change goes into a parameter, never into the geometry.
- **Dimensions from the origin.** Sketches are dimensioned from the projected origin, never from edges of other
  features: a stale reference once moved the caster holes on a printed base.
- **The model matches the robot.** Every physical change (a moved part, a re-routed wire) is made in the model too.
- Coordinates: origin at the centre of the base plate, +Y = front (caster side), +X = right, Z up.

```
         ┌──────── upper deck (z 43–46): Pi 4, pan-tilt + camera, sonar plate, speaker, mic port
  4 × pillar (40 mm)
         └──────── base plate (z 0–3): L298N, bucks, Wagos, IMU; motors + caster below; LiPo in rails underneath
```

## Printing (Bambu Lab A1, PLA)

- Preset: 3 walls, 25 % infill, **no supports**. A plate with features on both faces is split: the features become
  separate parts (spacers, trays), and the plate prints **flipped**, top face on the bed. `tools/export_flipped.py`
  exports a body from Fusion and mirrors its 3MF vertices `(x, y, z) → (x, −y, zmax − z)` so the flipped copy is
  ready to print (`... - FLIPPED (print this).3mf`).
- **Test coupons first.** Motor bracket, caster pad, battery rails, standoffs, pan-tilt holes and sonar plate were
  each printed as a small coupon (`Test - *.3mf`) before the big part.
- Hole sizes that work in PLA: 2.8 mm pilot = M3 self-tap, 3.4 mm = M3 clearance, 2.2 mm = M2.5 self-tap.
- Tall thin parts (8 × 40 mm pillars) need a brim.

| Part | File |
|---|---|
| Base plate | `Base Plate - FLIPPED (print this).3mf` |
| Upper deck (final) | `Upper Deck v2 - FLIPPED (print this).3mf` |
| Pillars | `Pillars x4.3mf` |
| Pi spacers, L298N spacers | `Pi Spacers x4.3mf`, `Fit Test + Small Parts.3mf` |
| IMU mount | `MPU6050 Mount v2.3mf` |
| Wago trays | `Wago Tray v3 x3.3mf` |
| Encoder parts (for a future base, not used on v1) | `Encoder Disc.3mf`, `Test - Encoder *.3mf` |

More: [`docs/printing.md`](../docs/printing.md), [`docs/fusion.md`](../docs/fusion.md).
