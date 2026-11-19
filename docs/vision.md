# Arcby — vision (faces and bodies)

How the robot sees people, which models it uses and why. Code: `code/robot/faces.py`;
bake-off script: `code/hwtest/faces.py`; models: `python3 code/get_models.py` (into `~/.local/share/arcby/models`).

## Pipeline (faces)
1. **Detect**: YuNet (OpenCV Zoo) on the frame shrunk to 480x288 → boxes + 5 landmarks.
2. **Landmarks + quality**: SCRFD-500M on the face region of the big stream. Its score is the face quality:
   < 0.3 not a face (not embedded), < 0.5 never attended, < 0.6 never enrolled / learnt (lamps, hands 0-0.14).
3. **Align**: similarity transform onto the ArcFace 112x112 template (`faces.align`).
4. **Embed**: ArcFace R50 w600k (InsightFace buffalo_l) quantised to int8 (`get_models.py --derive`) → 512-d unit
   vector, ~0.4-0.65 s / face on the Pi 4 in its own thread. Was MobileFaceNet until 2026-10-31 (see below).
5. **Match**: best similarity over each person's stored vectors vs one threshold.

All in onnxruntime + numpy + PIL (no OpenCV, no TensorFlow). Licences: YuNet MIT; SCRFD and
MobileFaceNet (InsightFace) non-commercial research use — fine for a personal robot.

## Bake-off 2026-10-31
**Speed on the Pi 4** (2 threads, full 1536x864 frame incl. shrinking):

| detector | 320x192 | 480x288 | 640x384 |
|---|---|---|---|
| SCRFD-500M | 44 ms | 80 ms | 131 ms |
| YuNet | 35 ms | 60 ms | 91 ms |

Embedders: MobileFaceNet 69 ms / face, SFace 101 ms / face.

**Range** (Selim standing, robot camera 14 cm above the floor): at ~4 m the face is ~32-36 px wide in a
1536 px photo. 320x192 input misses it with both detectors; at 480x288 YuNet finds 3/3, SCRFD 1/3; at
640x384 both 3/3 (YuNet with higher scores). No false faces. **Backlit at ~1 m (ceiling lamp behind the
head): no detector found the face** - an exposure problem, not a model problem.

**Recognition** (LFW: 21-person household enrolled from 3 clean photos each + 600 strangers; Selim's own
robot photos added; one threshold per model, set so 1 % of strangers get greeted by a household name):

| condition | MobileFaceNet right / verification TAR@0.1% | SFace |
|---|---|---|
| clean | 99 % / 99.3 % | 99 % / 98.8 % |
| 32 px wide | 99 % / 99.5 % | 99 % / 98.0 % |
| 40 px + sensor noise | 99 % / 99.3 % | 99 % / 98.5 % |
| dark (x0.2 + noise) | 99 % / 98.5 % | 99 % / 97.1 % |
| 40 px + dark | 98 % / 96.6 % | 97 % / 93.0 % |

No household member was ever taken for another. SCRFD landmarks beat YuNet landmarks slightly
(99 vs 98 % household, 2/3 vs 1/3 of Selim's real probes). `faces.even_light` (stretching dark crops)
helps MobileFaceNet by ~0.3 % in the dark, not at all when small + dark, and hurts SFace - not used by default.
Reports: `~/scratch/hwtest/faces/` (Pi) and the session scratchpad (Mac).

**Rejected**: RTMO-s pose / deepface (TensorFlow, re-implemented weights) / UltraFace (no landmarks) /
EdgeFace-XS (not tried: needs running its GitHub code; similar size and accuracy to MobileFaceNet on paper).

## Robot face shots, embedders compared (2026-10-31)
`hwtest/faceshots.py`: Selim at 1/2/3 m, straight / turned, night, ceiling lamps behind (camera 14 cm high, looking
30-45 deg up). Spot metering + HDR found the face at 1 m where plain AE did not, and lifted 3 m from ~0.28 to ~0.43.
`compare`, enrolled at 1 m only, threshold = 1 % of 600 LFW strangers: mbf 98 % recognised (margin 0.20),
r50 100 % (0.29), r50_int8 100 % (0.28). Pi 4 per face (2 / 4 threads): mbf 68 / 47 ms, r50 802 / 521 ms,
r50_int8 410 / 267 ms → r50_int8. LFW strangers are clean photos: re-check with strangers seen by the robot.

## Who is looking at the robot (gaze, 2026-10-31)
`robot/gaze.py`: head turn from the 5 landmarks of the BIG-stream crop (SCRFD at 128 px, ~4x a second per face),
smoothed into `Track.engagement` (tau 0.7 s); looking on at 0.5, off at 0.3. Labelled test set
(`hwtest/gazeshots.py`, Selim at 1 / 2 m, night): "look at me" vs phone / right / left / ceiling -> AUC 1.00 at both
distances, looking 0.65-0.77 vs 0.00 elsewhere. Small-stream landmarks fail at 2 m (AUC 0.67) - that is why the
big crop is used. Attention multiplies interest by 0.5-2.0 with engagement and glances round the people looking;
the greeter waits for ~1 s of eye contact.

Eye-contact / gaze models tried on the same crops (yakhyo/gaze-estimation, MIT, Gaze360, ONNX, 448x448 input):
MobileOne-S0, MobileNetV2, ResNet-18/34/50. "Eyes on the robot vs elsewhere": AUC 0.88-0.97 at 1 m, 0.58-0.84 at
2 m - all worse than the landmarks. "Head turned, eyes on the robot" (what landmarks cannot see): best MobileNetV2
0.94 at 1 m, chance (~0.5) at 2 m - the models mostly follow the head. Not used; revisit for close-range
conversation (< 1 m, maybe on the full-resolution still) or with a small classifier trained on collected data.

## Tracks, static spots and the threshold on recorded data (2026-11-01)
From the robot's first recordings (`robot/recorder.py`, 7 sessions, 873 tracks):
- **Real faces get lost for 0.8-3 s** by the detector and come back < 2 deg away (21 of 22 restarts): a confirmed
  face now ends only after `TRACK_FACE_MAX_MISSING_S` = 2.5 s (things that never passed the face check: 0.8 s).
- **823 of 873 tracks were not faces**, 685 of them at two fixed spots low in the room (jitter 0.0-0.1 deg; real
  faces 1.6 deg). `robot/static_spots.py` learns such spots from ended tracks (still, never a face, 5 within 2 deg)
  and checks a detection there only every 10 s: replayed on the recordings it silences 78 % of the fake tracks and
  no real face (0 of 50).
- **Threshold** on the family's own robot faces (time split: first half = profile, rest = new looks; clean stored
  profiles for the per-pair numbers): same person median 0.60; most alike other member = the brothers, median 0.29,
  p95 0.35-0.37; others 0.16-0.21. Per look the best match is wrong 4.5 %, per track (3-look majority) 0.6 % - much
  of it label noise from before the track-split fix. `FACE_MATCH_SIM` 0.30 -> 0.38 (per-track recognition 99.4 ->
  98.1 %): the threshold keeps a not-yet-known look-alike from being taken for a family member. A margin rule
  (best - second >= 0.05) removed every wrong track but cost 15 % recognition on the noisy labels - re-test on clean data.

## Review fixes (2026-11-01): pictures, track gate, refiner crop, lost head
- **The "unknown face" at pan +47 was a framed graduation photo on the wall**, enrolled as `bf1137bf`, followed and
  "looking". Now: a track that stays within `TRACK_STILL_DEG` (1 deg, 90 % of its still-head positions) for
  `TRACK_STILL_S` (10 s) is `Track.still` - never enrolled, never looking (`Track.picture` forces engagement 0). A new
  stranger is enrolled only once it moved or was watched 10 s. Per person the store keeps where its still tracks were
  (`still_spots`, per mode) and counts tracks clearly somewhere else; `PEOPLE_PICTURE_TRACKS` still tracks and hardly
  ever elsewhere = `picture` (index.json). `people_cli.py picture ID on|off|auto` overrides.
  Measured on the recordings: with no battery (camera fixed) the photo stays within 0.2-0.6 deg -> still; with the
  head powered its pose error alone makes it jitter 1.1-1.6 deg, about as much as a person sitting still, so with the
  head on a photo is not caught by stillness alone (open; the per-person flag learnt without the head still holds).
- Tracker gate uses the **track's own** face size and refuses a size change over `TRACK_SIZE_RATIO` (a 40-60 deg false
  box stole a 3 deg face 30 deg away). Replay of the 05:43 session: 44 -> 58 tracks (the stolen ones are separate).
- The refiner / gaze crops are padded: their face counts only if its centre lies in the track's box or IoU >= 0.3
  (`vision.own_face`) - the gown box under the photo borrowed the face above it.
- `HeadLife.pose_at` keeps the last pose really sent while the PTZ board is lost (speed 0), not the spring's.
- Static spots: evidence = still non-faces by the spread of positions (not the running max) + 1-2 frame flickers
  the face check rejected; spots are kept per mode (`head` / `nohead`), `clear()` when the base moved.
- Exposure: no EV push / squint while the sensor is at its longest exposure and top gain (`camera.saturated`).
- onnxruntime sessions no longer busy-spin (`allow_spinning 0`): one worker thread sat at 100 % of a core.
- `index.json` is fsynced, keeps a `.bak`, and a broken one falls back to it (no restart loop after a power cut).

## Calm head, listening nods, gaze thread, background stillness (2026-11-01, afternoon)
**Why the head looked random** (Selim), measured on the powered-head sessions 04:09-05:38 (frames.jsonl, ~4.5 rows/s):
- **Idle** (28-66 % of the time): a new random place every ~7-9 s, pan std 30 deg -> jumps median 27-40 deg, p90
  60-97 deg, each done as a 240 deg/s "saccade" to an empty spot, never coming back.
- **Track churn**: 6-13 new tracks a minute (the detector drops and restarts real faces); each new one got a 0.8 s
  glance (1.6-4.1 a minute) - 6-16 per session at undecided tracks under 5 hits, i.e. the same face found again.
- **Lost -> hold -> search -> idle**: the followed face dropped out 2-6 times a minute; after 1.5 s hold half went
  on to a +-15 deg search swing (30 deg jumps) and then a random idle jump. With another face in view it jumped to
  that one at once and back when the first returned.
- **Photo ping-pong**: Selim <-> the wall photo (bf1137bf, 36-42 deg apart) up to 4 switches in 4 s (05:14 session;
  47 s of following the photo vs 130 s Selim).
- **Voice sweeps**: 12 +-60 deg room sweeps in the night's events, 11 of them for voices that did not call it.
- Plus eye breaks every 3-7 s, micro-saccades every 1.2-3.5 s eased in 60 ms, blinks every 3.5-9 s.

**Now**: idle rests ahead (tilt +5, night 0) for 6-14 s, then one slow look (often where someone was seen, else
pan std 20 deg, at most 40) held 2.5-5 s, then back; big target jumps are minimum-jerk moves (to a face 0.2 s +
d/200 deg/s, idle 0.6 s + d/45 deg/s - top ~58 deg/s instead of 240). Simulated 10 min idle: 19 deg median /
34 deg p90 moves. Newcomer glances only for a track there 0.5 s, 12+ deg from the followed face and not where a
face was lost in the last 4 s; the followed face dropping out is waited for 2.5 s (a face reappearing within
10 deg of it is them), then a +-8 deg look around and it stays there. Switching faces needs 4 s of following
unless 3x more interesting; boredom after ~21 s (was 13); eye breaks every 6-12 s; pictures (`Track.picture`)
are never looked at; an unplaced voice that is not its name = a calm look ahead at face height (event
`turn how=ahead`), its name = the 3-stop sweep. Micro-saccades 0.8/0.5 deg every 2-5 s, blinks 2.5 deg every 6-14 s.
`HeadLife.run` catches and logs (every 30 s) a failing step instead of the thread dying with the head frozen.

**Listening**: attention mode `listen` while the followed person (partner, or looking at it) speaks - `hearing()`
says so: no breaks, glances or boredom. `HeadLife` nods at pauses (quiet >= 0.25 s after >= 0.8 s of speech, at most
every 2.5 s, 70 % of the pauses; 3-5 deg down and back in 0.6-0.9 s) and in 40 % of listening turns holds a slight
"head tilt" (3 deg pan + 1.5 deg down, eased). `hearing()` (alive.make_hearing) = None while the robot speaks or
nobody does, else seconds of quiet in the current utterance: `Ears.pause_s` when it exists, else `ears.seg.buf` /
`ears.seg.quiet`.

**Gaze thread**: the gaze refiner (SCRFD 128 px on big crops) runs in `vision.GazeWorker`; crops are cut in the camera
thread, results applied on a later frame to the track id they were cut for (gone tracks, results > 0.5 s old or
older than the track's last big-stream check are dropped).

**Background-relative stillness**: `camera.Background` measures the camera's true pose from the picture (phase
correlation of the 2x smaller grey frame, Hann window, whitened + low-passed; ~1 ms on the Mac) against reference
frames per 4 deg of commanded pose (linked to each other by measurement); `Track.path` keeps face positions relative
to the background, so servo jitter cancels. Synthetic test (head error random walk +-1.5 deg): photo spread 0.01
deg, swaying person 0.77. `TRACK_STILL_DEG` 1.0 -> 0.4 (no-head recordings: photo 0.12-0.29 p90 per whole track,
Selim sitting still 0.45-0.9). Not yet checked on the robot with the head on.

## Real robot photos are much harder than LFW
Selim's 3 m photos vs his 2 m photos scored 0.30 / 0.27 / 0.22 cosine (threshold 0.265); LFW same-person
pairs score ~0.5-0.7. Causes: few pixels (face 53 px at 2 m, 40 px at 3 m), steep upward angle from 14 cm
above the floor, ceiling lamps behind people. Fixes, biggest first:
1. **More pixels**: detect on a small stream, recognise from the face cut out of a high-resolution stream
   (sensor mode 2304x1296 = 1.5x, which also supports hardware HDR; 4608x2592 = 3x at 14 fps).
2. **Exposure**: hardware HDR (`wide_dynamic_range`), metering that ignores the ceiling, face-based exposure.
3. **Enrol close up**: ask a new person to come closer (or approach) before storing their first vectors.
4. **Learn from the robot's own camera**: add a vector after every confident match (different light / angle).
5. **Threshold from real robot photos**, not from LFW.

## Bodies (2026-11-01, `code/robot/body.py`)
Multi-person pose to find people whose face the detector cannot see (turned away, too close, only legs or a hand
in view) and to see waves. RTMO-s (COCO AP 67.7) took ~2.2 s / frame on the Pi 4 - too slow.
- **Model**: YOLO26n-pose (Ultralytics 8.4.197, AGPL), exported on the Mac to ONNX 320x192, opset 17
  (`python3 code/get_models.py --derive --only yolo26n_pose`). Output 1x56x1260 (box, score, 17 x (x, y, vis));
  own numpy NMS. ~250 ms a run on the Pi in its own thread (`BodyWorker`), every 0.3 s (1 s when nothing changes,
  0.6 s after 5 s with nobody, every frame-free slot while an arm is up). fps 12-14 -> ~9.5.
- **Tracks** in head angles (`BodyTracker`): matched by keypoint median distance, a gate growing with the gap, lost
  tracks revived up to 4 s (one person: 78 -> 38 tracks on replay).
- **Face hint** (`face_hint`): where the face should be, from body proportions - head keypoints, else shoulders, hips,
  knees, the box top, or only "someone at the edge" (a hand reaching in). Attention mode **seek** turns there
  (4 s, 20 s when head keypoints are seen), gives a spot up for 60 s; tilt clamped to `HEAD_TILT_SOFT` (up to 60).
- **Waves**: the pose model is too slow for a 2-3 Hz wave (3.3 Hz sampling sees a drift), so while a forearm is up
  `WristWatch` diffs a patch round the wrist every frame and counts side-to-side swings of the moved pixels. The
  forearm must point clearly up (`BODY_WAVE_FOREARM_UP` 0.7 = ~35 deg): the only "wave" in a 48 min PlayStation
  session was Selim's forearm on the desk moving the controller. The greeter answers a wave (waits 1.5 s for the name).
- **Results** (sessions 2026-11-01 200759 and 202222, 48 min): 5 / 5 real waves answered, 0 still objects sought;
  after the fixes 16 of 19 body seeks found the face. Weak spot: a person sitting ~3 m away is found in only 13 %
  of the frames that have their face (320x192 too small) - next: run the model on a full-resolution crop under a
  face that has no body. Not done yet: clothes memory, guided body shots.

## Hand signs (2026-11-08, `code/robot/gestures.py`)

The pose model has no fingers, so signs are read from the forearm, in the person's own shoulder widths, relative to
their own elbow / shoulder (walking, the head turning or the robot driving is no sign):
- **"gel gel"** (Beckon): forearm up in front, the wrist going toward the shoulder and back - up-down in the picture -
  `GESTURE_BECKON_TURNS` (3) turns within `GESTURE_BECKON_WINDOW_S` (2.5 s) over `GESTURE_BECKON_AMP` (0.3), at least
  `GESTURE_BECKON_RATIO` (1.5) x as much up-down as sideways (else a wave), the elbow within `GESTURE_BECKON_ELBOW`
  (0.3). A beckoning arm gives no "waved" event, an arm that waved gives no beckon.
- **"dur"** (StopSign): both shoulders seen (facing the robot), a wrist at most `GESTURE_STOP_LEVEL` below its shoulder
  and above its elbow, held `GESTURE_STOP_HOLD_S` (0.7 s) within `GESTURE_STOP_STILL` (0.25).
A false beckon is a drive nobody asked for (strict); a false stop sign only stops a drive (lenient: a phone at the ear
counts). First guesses on synthetic tracks (`tests/test_gestures.py`: beckons at 1-2 Hz yes; waves, talking with the
hands, lifting, walking past no) - `hwtest/signs.py replay` runs them on the recorded pose results (the "frames"
stream), target 0 beckons in hours of normal life; Selim's own signs at 1 / 2 / 3 m are still to be recorded.

## Far bodies: the pose model on a big-stream crop (2026-11-02)
The wide lens makes faces tiny: 15-21 px wide at 1 m, 12 px at 2 m, 10-12 px at 3 m in the small stream (576 x
324). On the whole frame (shrunk to 320 x 192) the pose model found Selim at 3 m with confidence only 0.37-0.70 (cut-off
0.35 - the seated / desk case falls under it: 13 % of frames). Now, while a face <= `BODY_CROP_FACE_MAX_PX` wide is
in view, every other pose run is on a crop of the **big** stream (2304 x 1296) under it (`body.crop_for_face`): the
model's 5:3 shape (no letterbox waste), the face near the top centre, `BODY_CROP_FACES_TALL` = 7 face heights tall
(at least `BODY_CROP_MIN_PX`); results are mapped back into small-stream pixels (`body.to_frame`), so tracks, face
links, hints and the wrist watch work unchanged. Faces waiting longest are cropped first; frames without a big stream
use the whole frame. Same ~250 ms a run (same model input).

Face-shot sessions of 2026-10-31 (big-stream JPEGs, Selim at 1 / 2 / 3 m, 32 shots with a face):

| distance | body found: whole frame | crop | confidence whole -> crop |
|---|---|---|---|
| 1 m | 9 / 9 | 9 / 9 | 0.82-0.86 -> 0.88-0.92 |
| 2 m | 7 / 10 | 10 / 10 | 0.72-0.80 (3 missed) -> 0.88-0.92 |
| 3 m | 13 / 13 | 13 / 13 | 0.37-0.70 -> 0.89-0.92 |

The crop ends near the knees (13-15 keypoints instead of 16-17) - enough for the face hint and waves. Not yet tried
in alive.py on the robot (`stats["crops"]` counts them); the seated-at-3-m case should be checked there.

## Showing things (2026-11-02, `code/robot/show.py`)
- **Hand held up**: from the body tracks (head angles, so the head moving does not matter): a wrist above the hips,
  within the shoulders +- `SHOW_SIDE_SW`, the forearm not hanging down, still (< `SHOW_STILL_SW` shoulder widths) -
  a wave moves too much to count.
- **The thing**: YOLO26n detection (80 COCO classes, optional model `yolo26n`) on a big-stream crop round the wrist
  (`SHOW_CROP_SW` shoulder widths), every `SHOW_EVERY_S` while a hand is held (~340 ms a run on the Pi next to alive.py); furniture / vehicles / animals
  (`NOT_HELD`) never count.
- **Sure** = held `SHOW_HOLD_S` + the same thing in 2 of the last 3 runs + that person's face looking at the robot ->
  `showing` event (once per person and thing per `SHOW_AGAIN_S`); the head goes to the thing (attention mode `show`,
  below its name / a voice, above everything else) and a 640 px JPEG of it is taken for the mind.
- **Asked** ("bak", "şuna bak", "bu ne", "ne bu", "gördün mü", "what is this") = sure: it looks at the asker's raised
  hand (else their wrists, else under their face) and the picture goes with the question - also for paper or a
  drawing, which the object model does not know.
- **The guided showing session (2026-11-02 evening, S3)**: the robot on the floor (head following, the real ceiling
  lamp behind Selim), voice-guided from the Mac, recorded by alive.py itself (`robot/session_rec.py`: big-stream frames
  4 a second + every hand crop while `~/scratch/alive/session.json` exists): phone / sheet of paper / a white book held
  up at "1 / 2 / 3 m" (3 m = behind the bed, ~4-5 m) + reading his own phone, waving, hands down - 870 frames in
  `/mnt/arcby-data/sessions/show-20261102` (the permanent test set). Bake-off on the Mac, the robot's path (pose model,
  face-anchored far-body crop with 2.5 s face memory), the upper body + hands cut out, decided as the robot does
  (the same thing in 2 of 3 checks):

  | model | showing steps caught (9) | false alarms (9 not-showing steps) | Mac |
  |---|---|---|---|
  | YOLO26n COCO (first try) | 3 (phone only) | 2 (reading own phone) | 7 ms |
  | MobileCLIP2-S0 zero-shot (6 sentences) | 6 (the white book = "paper") | 0 | 180 ms |
  | **YOLOE-11s, 24 home words** | **8** (only the book behind the bed missed) | 1 ("wallet": reading own phone - the looking rule blocks it) | 28 ms |
  | YOLOE-26n, 24 home words | 6 | 0 | 17 ms |

  "glasses" in the word list fired on Selim's own glasses in every step - nothing worn is in the list. YOLOE scores
  lower: its threshold is 0.25 (`MODELS["yoloe_home"]["min_score"]`). The winner `yoloe_home` (YOLOE-11s, 40 MB ONNX,
  `get_models.py --derive --only yoloe_home`) looks at the upper-body box (`show.upper_body`); speed on the Pi not
  measured yet (the Pi was off) - if too slow, the 26n export is the fallback. Selim does not want camera pictures
  sent to a cloud model: the mind's picture path is not used for showing.

## Face threshold from the robot's own recordings (2026-11-02, learning job `faces`)
3369 recorded looks (1764 tracks, 143 of them clearly a known person) - same person = one track, different people =
two tracks in one frame, scored against the people store leave-one-out:

| | median |
|---|---|
| two looks of one track (>= 2 s apart) | 0.50 |
| two faces in the same frame | 0.16 |
| a known person's look vs their own vectors | 0.51 (Selim 0.48, Emre 0.49, Halil 0.68, Meltem 0.65) |
| ...vs the most alike OTHER person | 0.28 (Emre's looks reach Selim up to 0.63 - the brothers) |

Single looks never get "others <= 1 %" (at 0.50 still 3.7 %), but the robot decides by the 3-look vote of a track:
0.7 % wrong (1 of 143) for every threshold 0.30-0.42; right / undecided 95 / 4 % at 0.34, 92 / 7 % at 0.38. Others in
the frame reach a known person 8 % of the time at 0.38, 16 % at 0.34 (partly not strangers: the framed photo, an
Emre duplicate 7e416de7). The job's rule (lowest threshold with <= 1 % wrong tracks and others let in at most 2
points more than now) chose **0.36** -> tracks 94 / 0.7 / 6 %; applied automatically (`tuned.json`,
`learn.py undo FACE_MATCH_SIM` to go back). config.py's own value stays 0.38.

