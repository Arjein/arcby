# Photos for the main README

Always add photos through the cleaner. Phone photos contain the GPS position where they were taken:

```bash
python3 tools/clean_photos.py ~/Desktop/IMG_1234.HEIC --name hero
python3 tools/clean_photos.py --check docs/images/*.jpg      # every line must say "clean"
```

| File | Shows |
|---|---|
| `hero.jpg` | 3/4 front view (top of the main README) |
| `front.jpg` | straight from the front: camera head, ultrasonic "eyes" |
| `three-quarter.jpg` | from the front left: both decks, buck converter, caster |
| `top.jpg` | the upper deck from above: Pi, pan-tilt, wiring |
| `on-the-rug.jpg` | on the living-room rug |

Worth adding later: `cad.png` (the Fusion 360 model), `dashboard.png` (blur the Network card), `drive-web.png`.

Tips: shoot at the robot's height rather than down from above. A plain wall or desk makes a good background. Daylight
from a window gives soft light. Before shooting, check that no family photos, documents or screens with
personal information are in the frame.
