# Navigation: Arcby drives by itself

Written 2026-11-04 (Mac, simulation only - **not run on the robot yet**). Code: `code/nav.py` (the program + web
page), `code/robot/floor.py` (seeing the floor), `code/robot/localmap.py` (memory), `code/robot/nav.py` (the brain),
`code/sim/navworld.py` (simulated rooms), `code/hwtest/navcal.py` (measure the speeds). Settings: `NAV_*` in
`code/robot/config.py`. Tests: `code/tests/test_nav.py`, `code/tests/test_nav_program.py`.

Goal (Selim, 2026-11-04): explore the room carefully, never drive into anything, and walk up to people it sees.

## What it has to see with

| Sensor | Sees | Misses |
|---|---|---|
| HC-SR04 sonar, front, 7 cm up | one ~24 deg cone, 2 cm - 2 m, anything > ~5 cm tall | cables, shoes, low things; thin chair legs except straight ahead; walls hit at a slant (the sound bounces away = "nothing"); soft fabric; everything to the side and behind |
| Camera on the head, lens 14 cm up | the floor from ~13 cm to ~3 m ahead, 102 deg wide (more by panning the head) | overhangs (a sofa / table edge above the floor - it looks DOWN); glass; anything the floor model mistakes for floor |
| IMU | exact turns, bumps (safety.py), tipping | where it is (no encoders: distance = speed x time) |

Nothing looks behind: it never reverses more than 10 cm, and only along the way it just came.

## Seeing the floor (`robot/floor.py`)

```
        lens (14 cm up), head tilted 28 deg down
          \
           \  ray through one pixel row
            \
   ==========*=====================  floor
             d = how far that row looks
```

The lens height and the head's tilt / pan say, for every pixel, where on the floor it looks (a ray from the lens cut
with the floor: `ground_points`). So the picture is a floor map:

1. **Learn the floor.** The patch 5-35 cm in front of the robot is taken as floor whenever the sonar says
   that much is free: a colour histogram (r, g chromaticity - shading does not change it) and a brightness one.
   Only a patch that already looks mostly like floor is learnt (at most 30 % different), so a shoe in the patch is
   not learnt as floor. **Start the robot with ~50 cm of free floor in front of it** - that is the first lesson.
2. **Find what is not floor.** Every 4 x 4 px cell whose colour or brightness is rare in the histograms.
3. **36 rays** go up the picture from the bottom; the first two non-floor cells in a row are an obstacle's foot,
   and its row says how far it is. A ray that reaches 150 cm without one = free that far.
4. **Floor edges** (a rug, a tiled floor after a carpet), judged only in the robot's path (rays within 20 deg of
   straight ahead - about where the sonar listens): most of them stop on one straight line and the ground beyond is
   non-floor nearly everywhere. Then, ONLY if the sonar has a real echo 30+ cm past the edge, at least 40 cm (a wall's foot or a box looks the
   same to the camera, but the sonar hears it), it crawls over and learns the new floor (keeping the old one). While
   crossing, camera "obstacles" within 12 cm of the edge line are the edge itself (pictures that failed the test left
   them in the memory) and do not stop it. The whole-picture test was too strict: a rug is seen at a corner or a
   slant most of the time (2026-11-04 drive: 14 of 112 edge pictures passed, now 48; the robot never left the rug).

~5 ms a picture on the Pi 4. Weak spots: a floor with the same colour as the obstacle (white socks on a white
floor), strong shadows (seen as obstacles: it stops early - safe, not dangerous), patterned rugs (many "edges"),
**multi-tone parquet** (the reference patch holds one or two planks; darker planks and sun glare look non-floor -
`NAV_FLOOR_MIN` 0.04 -> 0.01 cut it from 22 % to 15 % of the view on 2026-11-04; a floor memory that keeps every tone
seen while the sonar said free would fix the rest), glare on parquet next to a grey rug (looks like the rug).

## Memory (`robot/localmap.py`)

Points where obstacles stand, in "odom" coordinates (where it started; heading from the IMU, distance = estimated
speed x time). Each sensor only clears its own points (the sonar cannot clear a cable it cannot see; the camera
cannot clear a table edge 7 cm up that it cannot see). Points are forgotten after 30 s or 2 m - dead reckoning
drifts. Besides obstacles it keeps what the camera saw FREE in the last seconds: **it only drives through ground it
saw free in the last second** - unknown ground counts as blocked.

## The brain (`robot/nav.py`)

```
START -> SCAN (look around: head at +60, +30, 0, -30, -60 deg, 0.55 s each)
          -> CHOOSE the best direction: free way x seen way, minus a little per degree of turning,
             plus a bonus for ground not visited yet
          -> TURN on the spot (gyro; back up first if something is inside the 10.5 cm spin circle)
          -> DRIVE at ~35 cm/s, crawl (~25 cm/s) when anything is within 55 cm,
             stop 18 cm before anything, steer gently (<= 60 deg/s), head looks where it goes,
             every 8 s stop and look around again
          -> nothing good: turn 135 deg (always the same way round) and look again; 4 times -> HELP
```

- **Bump** (the IMU felt a hit no sensor saw - glass, a dark corner): remember it, wait 1 s, clear the safety stop,
  back up, look around. More than 3 in 5 minutes -> HELP (stands still until someone presses a button).
- **Stuck** (driving 1.5 s at one steady echo that does not get closer): back up, look around.
- **Tipped / low battery** -> HELP. IMU lost / no safety verdict -> stand still and wait.
- **People mode**: a body (YOLO26n-pose, the model alive.py uses) seen -> turn to them, crawl, stop when their feet
  are 60 cm ahead, look up at them 4 s, then leave them alone 30 s. Their legs are obstacles like any others.

The safety layer (`robot/safety.py`) is under all of this: whatever the brain asks, a bump / tip / lost IMU /
low battery stops the motors.

## The program (`code/nav.py`)

```
pkill -f "code/[a]live.py"            # alive.py owns the IMU, head, sonar and camera
python3 code/nav.py                   # prints a link: open it on the phone
python3 code/nav.py --sim             # same brain and page, a random simulated room (on the Mac)
```

Buttons: **Explore**, **Go to people**, **STOP**. The page shows the camera with the floor check drawn on it (green =
free that far, red dot = an obstacle's foot), a top view of what it remembers, the state and why.
**Dead man**: it drives only while the page is open - close the tab / lock the phone / lose Wi-Fi and it stands still
within 1.5 s. STOP keeps the camera running: put things in front of it and watch whether the red dots land right.
`pkill -f "code/[n]av.py"` stops it like Ctrl+C (motors off, recording closed) - run it alone, not in one ssh line
with `code/nav.py` in it (the pattern then matches that shell too).

**Recording a drive** (`robot/nav_rec.py`, off with `--no-record`): every drive is a session in `DATA_DIR` (the USB
stick, same size limit as alive.py's recordings): `nav.jsonl.gz` = every step while driving (yaw, sonar, head, v,
turn, state, why), `floor.jsonl.gz` + `navframes/<t>.png` = one lossless picture a second with its floor verdict.
That is what the floor check is tuned on - no hand-taken photos: a place the camera called an obstacle that the
robot later drove over with the sonar free and no bump was floor after all. A 5 min drive = ~270 pictures, ~85 MB.
**Head errors**: one failed I2C write to the PTZ board is retried after re-initialising it; the board silent for
`NAV_HEAD_DOWN_S` (1 s) stops the drive.

## Driving inside alive.py (2026-11-08, `robot/legs.py`)

The same brain runs inside the living robot: with the battery on, alive.py opens the motors + IMU + safety
(`robot.main.Robot`), the sonar thread and `Legs` (one navigation step every 1 / `NAV_RATE_HZ` in its own thread).
nav.py is still the tool for tuning (its page, its dead man); both share `robot/drive_body.py`. No dead man in
alive.py - every stop counts instead, and a stopped drive stays stopped until the next start:

| Starts | Stops |
|---|---|
| "gel gel" hand sign -> **follow** that person | stop sign (a raised still hand), "dur" / "Arcby dur" (any voice) |
| Drive card **Explore** / "Arcby gez" -> explore `WANDER_MAX_S` | the Drive card's **STOP** |
| bored (nobody for `BORED_EXPLORE_S`, by day) -> explore | any latched safety stop (bump, tipped, low battery) |
| | HELP, lost them (follow), time up, the motors' threads failing |

While it drives the head belongs to the drive (HeadLife mode "drive": straight there at `NAV_HEAD_DPS`, no life on
top); the camera loop feeds the floor check with the lores picture and the head pose of that frame. Face, body and
voice keep running: the greeter may greet, a conversation goes on.

**Follow** (Navigator mode "follow"): the head looks at the person (chest height), so the floor check is off:
obstacles = sonar + memory + bump - at crawl speed behind someone who walked there. More than `FOLLOW_TURN_DEG` aside
-> turn on the spot, else crawl steering at them; stand `FOLLOW_STOP_CM` (60) from their feet or the sonar, go again
when they are `FOLLOW_GO_CM` farther. Not seen `FOLLOW_LOST_S` (3 s) -> stand, look around once, found within
`FOLLOW_REFIND_S` (5 s) -> on, else stop ("Seni kaybettim."). Sim (`tests/test_nav.py` Follow): 8 rooms with a walking
person, 0 contacts.

## Simulation results (2026-11-04)

`sim/navworld.py`: random living rooms 3-4.5 x 2.5-3.5 m with furniture boxes, chairs (4 legs, 3 cm thick), cables,
a rug, people (standing or walking). The sim is unkind: the robot's real speed is 0.8-1.2x what it believes, walls
at a slant give no echo, thin legs answer only near the beam's middle, 3 % of pings are lost, 1 % of floor cells
look like obstacles. The sonar and the floor rays use the real code (`FloorSeer._rays`, the real ground geometry).

- 24 rooms x 2 minutes, explore: **0 contacts, 0 cables touched**, 2.3-9.4 m driven each; 2 rooms ended in HELP
  (boxed in a corner - it asks instead of forcing its way).
- 12 rooms x 3 minutes, people mode (1 person, standing or walking): 0 contacts; walking up to someone it stops
  30-60 cm before their legs (someone it only notices once beside them can be 15-25 cm away - it stopped there for
  an obstacle, as for any other).
- It is careful, so it is slow: ~77 % of the time it stands still looking around.

## First real runs (2026-11-04)

The checklist before the first run, done:
1. Battery on, no under-voltage.
2. Speeds (`hwtest/navcal.py` / one run each, hard floor, tape-checked: ~50 cm in 1.5 s): crawl **38.3**, cruise
   **40.2** cm/s - not the guessed 25 / 35, and barely apart. Point the sonar squarely at a bare wall with
   nothing within ~40 cm of the path and the head straight ahead: other echoes (112 / 57 cm at a 150 cm wall) spoilt
   the first runs.
3. Head at -28 deg (tilt servo 52): quiet, the ribbon loose, nothing touches.
4. Red dots sit on the foot of a box 50 cm ahead: `NAV_CAM_X_CM` / `NAV_CAM_H_CM` are right.
5. Explore on a grey 1.5-2.5 m rug on parquet: 2 min, ~25 decisions, **0 contacts, 0 safety stops**, turns within
   8.6 deg; ended in HELP inside the rug (it took the rug edge for a fence). A 5 min run: 52 looks, 0 contacts, no
   HELP, still never left the rug -> the edge test in the robot's path + crossing past remembered edge points
   (above). With those: 3 crossings started, each stopped by the remembered edge after ~1 s -> fixed.
   A third run (with the head retry): 2 crossings started and stopped at once, "1 cm ahead": dark tufts of the shaggy
   rug at the bottom of the picture (2 cells there = 0.6 cm of floor) were taken for an obstacle's foot -> an obstacle
   must now also be `NAV_FLOOR_RUN_CM` (1.5 cm) long on the floor (on that drive: 119 -> 14 such false feet), and a
   crossing starts only with nothing remembered between the front and the edge. NOT yet tried on the robot (the
   battery ran low: the PTZ board dropped out twice, the Pi showed under-voltage).
6. Evening, charged: with the body model loaded. Run 4 stood at the rug's edge looking at parquet: everything beyond
   the edge was "obstacle" (only the rug was learnt), the edge test only ran while DRIVING, so after 3 looks round it
   asked for HELP -> a look round that finds no way but saw an edge ahead with the sonar free now crosses. Run 5:
   **2 crossings, out onto the parquet**, 0 contacts; ended by the head board (EIO, then the re-init raised
   PanTiltError, which the retry did not catch -> caught now). The edge sonar rule was a flat 80 cm (it stayed on the
   rug with ~50 cm of parquet free before a chair) -> echo `NAV_EDGE_SONAR_PAST_CM` (30) past the edge, at least
   `NAV_EDGE_SONAR_MIN_CM` (40); with it the sim robot took the foot of a wall it drove ALONG for an edge and came
   1.5 cm from it -> an edge line must face the robot (`NAV_EDGE_MAX_SLANT_DEG` 45; 73 of the 86 real rug edges
   recorded that day pass). Still to do (finish plan step 1): 5 of 13 stops in run 5 were "0 cm ahead", and the head
   ticks while driving (the sweep).
7. 2026-11-08 (cable re-seated): stuck at a wall - every 135 deg turn stopped 52-76 deg short. The spin's stop lead was
   one value learnt from all turns; small turns (10-35 deg) went 10-32 deg over, the lead grew to its 0.4 s cap, and
   misses over 40 deg were not learnt from, so it never recovered -> HELP. Now small (< `NAV_TURN_SMALL_DEG`) and big
   turns learn apart and a big miss counts as 40 deg. Next run: 10 turns, +3..+31 deg over (the 135 at the wall +6),
   0 HELP, 5 crossings started, 2 completed; it ended in a power cut mid-spin.

## Known limits

- **Overhangs**: a sofa, bed or low table whose edge is above the sonar (7 cm) and out of the camera's view (it
  looks down) - the robot may drive under it until the head hits (bump stop). Robot height ~20 cm.
- **No position**: it does not know where it is in the house, cannot go back to a place, and the memory fades in 30 s.
  Wheel encoders (Base v2, parked) would fix the drift.
- **Glass, mirrors, black cables on black floors**: only the bump stop.
- **Low things under the sonar in the path** (a chair's wheel base ~5 cm high): the camera can take its straight
  front for a floor edge (seen on 2026-11-04), and the sonar beam passes over it - the robot may crawl into it
  (bump stop). Taller ones answer the sonar and are never crossed.
- The person's feet distance assumes a flat floor and the feet in the picture; cut off at the bottom = "near, unknown".
- **Follow** has no way round: furniture between the robot and the person makes it stand (and then lose them).
  Low things (cables) are not seen while following - the head looks at the person.
- **Wander** has no position memory: it does not come back to where it was.
