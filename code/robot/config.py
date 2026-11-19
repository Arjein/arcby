"""
Arcby configuration: pins, bus addresses, mounting and thresholds.
Pin numbers follow docs/pinout.md - keep both in sync.

Robot frame (same as the Fusion model): +X = right, +Y = front (caster side),
+Z = up. Yaw is positive turning LEFT (counter-clockwise seen from above).
"""

# ---------- I2C ----------
I2C_BUS = "/dev/i2c-1"      # Pi pins 3 (SDA) / 5 (SCL), shared with the PTZ board

# ---------- IMU (MPU6050 / GY-521) ----------
IMU_ADDR = 0x68             # AD0 not connected -> 0x68
IMU_RATE_HZ = 100           # loop rate; the chip samples at the same rate

# Robot axis = signed sensor axis. The board is mounted upside down (chip down),
# so Z is flipped. Measured on the robot 2026-10-29 (caster side raised 30 deg:
# gravity moved onto sensor +X): sensor +X points FORWARD, sensor +Y points RIGHT.
# Re-check with `python3 code/hwtest/imu.py --only axes` if the board is re-mounted.
IMU_AXES = ("+y", "+x", "-z")   # (robot X, robot Y, robot Z)

# Gravity at rest on a flat floor, in the robot frame (after IMU_AXES): the IMU is glued ~8 deg off level
# (2026-10-31, re-glued by Selim). Every sample is rotated so this reads straight up -> the normal stance = 0 deg
# tilt. Measure again with code/hwtest/imu_level.py whenever the IMU or the robot's stance changes. None = no trim.
IMU_LEVEL_UP = (-0.020, 0.147, 1.038)
IMU_CALIB_S = 2.0           # still time at start for the gyro bias
IMU_CALIB_MAX_GYRO_STD = 1.0    # deg/s - more means the robot was moving
IMU_CALIB_MAX_ACCEL_STD = 0.03  # g
IMU_TILT_ALPHA = 0.98       # complementary filter: share of gyro in roll/pitch
IMU_TIPPED_DEG = 45.0       # tilt above this = tipped over -> safety stop
IMU_BUMP_G = 0.6            # sudden horizontal accel change = bump (tune with hwtest)
IMU_MAX_DT_S = 0.1          # gap between samples above this is not integrated
IMU_FROZEN_SAMPLES = 25     # identical raw frames in a row = chip stopped updating
IMU_MAX_ERRORS = 5          # I2C errors in a row before the IMU counts as failed

# ---------- Pan-tilt head (Arducam B0283: PCA9685 PWM chip on the PTZ board) ----------
# Address and channels measured 2026-10-30: 0x40, tilt = channel 0, pan = channel 1. Directions and level
# measured 2026-10-31 with camera photos: a longer pulse turns the head RIGHT (pan) / UP (tilt); 20 deg of
# tilt moved the picture ~135 px = the lens' 6.7 px/deg, so the 0.5-2.5 ms = 0-180 deg scale
# (Arducam's demo, github.com/ArduCAM/PCA9685) fits.
PTZ_ADDR = 0x40
PTZ_PAN_CH = 1
PTZ_TILT_CH = 0
PTZ_PWM_HZ = 50             # standard servo frame (20 ms)
PTZ_OSC_HZ = 25_000_000     # PCA9685 internal oscillator (datasheet; real chips vary a few %)
PTZ_PULSE_US = (500, 2500)  # pulse for servo angle 0 and 180 deg
# Head angles are degrees from straight ahead: pan + = LEFT (same sense as yaw), tilt + = UP.
# servo angle = centre + sign * head angle, clamped to the servo limits below.
PTZ_PAN_CENTRE = 85.0       # servo angle where the camera looks straight ahead (2026-10-31: pan horn refitted
                            # at servo 90; bottle 1 m ahead on a tape along the centre line is centred at 85)
PTZ_TILT_CENTRE = 80.0      # servo angle where the camera is level, robot on the floor (2026-10-31: tilt horn
                            # refitted at servo 80; lens 14 cm high, bottle base 100.5 cm ahead ~87 px below the
                            # middle of a 1280x720 photo = level within ~+-1.5 deg, unsure of the lens px/deg)
PTZ_PAN_SIGN = -1           # a longer pulse turns the head right (2026-10-31 photos)
PTZ_TILT_SIGN = 1           # -1 if bring-up says the head turns DOWN for + tilt
PTZ_PAN_SERVO = (10.0, 170.0)    # servo angle limits; both ends reached without trouble after the horn refit (2026-10-31)
PTZ_TILT_SERVO = (15.0, 140.0)   # tilt servo limits: 145 reached without trouble (2026-10-31), capped at 140 =
                                 # head +60 deg up (Selim); the low end 15 (head -65) is not tried on the robot yet
PTZ_SPEED_DPS = 90.0        # default speed of move() - smooth, no jerk on the camera ribbon

# ---------- Head life (robot/head_life.py): the head never freezes ----------
HEAD_RATE_HZ = 50
HEAD_PAN_SOFT = (-60.0, 60.0)    # head angles used by behaviour: inside the servos' rated range (FLAGS
HEAD_TILT_SOFT = (-30.0, 60.0)   # 2026-10-31); beyond that the body should turn instead. Up 55 -> 60 (= the servo
                                 # limit, checked to +65 on 2026-10-31): someone standing close is 60-100 deg up
HEAD_FOLLOW_HZ = 1.1        # spring speed when following a face (natural frequency, Hz): soft, slightly late
HEAD_IDLE_HZ = 0.5          # lazier when just looking around
HEAD_SACCADE_DEG = 10.0     # a target jump bigger than this is a planned minimum-jerk move (smooth start and stop)...
HEAD_MOVE_FACE = (200.0, 0.2)    # ...to a face: length = 0.2 s + distance / 200 deg/s (40 deg: 0.4 s - vision is
                                 # blind while the head moves fast, so not slow)...
HEAD_MOVE_IDLE = (45.0, 0.6)     # ...just looking around: 0.6 s + distance / 45 deg/s (30 deg: 1.3 s, lazy)
HEAD_SACCADE_HZ = 3.0       # spring speed along a planned move
HEAD_MAX_DPS = (120.0, 240.0)    # top speed: normal, planned move (the servos do ~600 deg/s unloaded)
HEAD_MICRO_EVERY_S = (2.0, 5.0)  # micro-saccade: a new tiny offset this often (random in range)... (was 1.2-3.5:
HEAD_MICRO_DEG = (0.8, 0.5)      # ...of this size (std dev pan, tilt); halved on a face    twitchy, 2026-11-01)
HEAD_MICRO_EASE_S = 0.12         # ...each taking about this long
HEAD_BREATH = (0.8, 4.5)         # slow tilt wave: amplitude deg, period s
HEAD_BLINK_EVERY_S = (6.0, 14.0) # "blink" = quick tilt dip...
HEAD_BLINK_DEG = 2.5             # ...this deep, 70 ms down / 40 ms hold / 140 ms up; sometimes twice
HEAD_NOD_PAUSE_S = 0.25     # listening (mode "listen"): a pause this long in their speech may get a nod...
HEAD_NOD_MIN_TALK_S = 0.8   # ...after at least this much speech since the last nod...
HEAD_NOD_EVERY_S = 2.5      # ...at most this often...
HEAD_NOD_P = 0.7            # ...and not every time (people do not nod like a metronome)
HEAD_NOD_DEG = (3.0, 5.0)   # a nod: this far down and back...
HEAD_NOD_S = (0.6, 0.9)     # ...in this long, silently
HEAD_COCK_P = 0.4           # per listening turn: this often a slight sideways "head tilt" (no roll servo: a small pan
HEAD_COCK_DEG = (3.0, 1.5)  # offset + a little down, deg), eased in over ~HEAD_COCK_EASE_S and held while they talk
HEAD_COCK_EASE_S = 0.8
HEAD_LISTEN_HOLD_S = 1.5    # their speech stopped this long: listening is over (head tilt eases back)
HEAD_ERROR_LOG_S = 30.0     # a failing head step is logged at most this often (the loop goes on)
HEAD_MIN_STEP_DEG = 0.25         # do not send moves smaller than this (a PCA9685 count = ~0.44 deg)
HEAD_SERVO_LAG_S = 0.07          # the picture moves this long after a command (measured 40-95 ms, 8 deg steps)
HEAD_PROBE_S = 5.0          # the PTZ board stopped answering (battery off): a read-only look this often

# ---------- Attention (robot/attention.py): where to look ----------
ATTN_SWITCH_GAIN = 1.6      # change to another face only if it is this much more interesting than the current...
ATTN_MIN_DWELL_S = 4.0      # ...and only after following the current one this long...
ATTN_SWITCH_NOW_GAIN = 3.0  # ...unless the other is this much more interesting (Selim <-> photo ping-pong, 2026-11-01)
ATTN_REACQUIRE_DEG = 10.0   # a face this close to where the followed one was just lost is taken for them
ATTN_HOLD_S = 2.5           # face lost: keep looking where it was this long (the detector loses real faces 0.8-3 s)...
ATTN_SEARCH_S = 2.5         # ...then look a little to each side of that spot this long...
ATTN_SEARCH_DEG = 8.0       # ...this far (was 15: 30 deg swings), then stay there idle
ATTN_IDLE_EVERY_S = (6.0, 14.0)  # idle: rest ahead this long, then a slow look somewhere... (was a new random place
ATTN_IDLE_LOOK_S = (2.5, 5.0)    # ...held this long, then back           every 2.5-6 s, median 30 deg jumps)
ATTN_IDLE_HOME_TILT = 5.0   # resting ahead: this tilt (night: ATTN_NIGHT_HOME_TILT)
ATTN_IDLE_PLACE_P = 0.5     # an idle look goes where someone was seen lately this often (are they back?)...
ATTN_IDLE_PAN_DEG = 20.0    # ...else somewhere random: pan spread (std dev), at most ATTN_IDLE_PAN_MAX, tilt in
ATTN_IDLE_PAN_MAX = 40.0    # ATTN_IDLE_TILT
ATTN_IDLE_TILT = (-5.0, 25.0)
ATTN_SHARE_EVERY_S = (8.0, 15.0)   # following someone: this often a glance at another person LOOKING at the robot
ATTN_SHARE_S = 1.5          # ...for this long (people paying attention share it; the others get none)
ATTN_GLANCE_S = 1.2         # a new face appearing while following someone: look at it this long, then back...
ATTN_NEW_MIN_S = 0.5        # ...once it has been there this long...
ATTN_GLANCE_MIN_DEG = 12.0  # ...and is this far from the followed face
ATTN_CHURN_DEG = 8.0        # a new track this close to where a track ended...
ATTN_CHURN_S = 4.0          # ...this recently is the same face found again, not a newcomer
ATTN_LISTEN_HOLD_S = 1.5    # listening (mode "listen") lasts this long past the last sound of their speech
ATTN_BODY_TURN_DEG = 45.0   # following a face this far to the side for ATTN_BODY_TURN_S -> ask the body to turn
ATTN_BODY_TURN_S = 1.2
ATTN_BORED_S = 20.0         # boredom: interest x exp(-seconds watched / this) - a quiet person who just sits there...
ATTN_AWAY_BELOW = 0.35      # ...is looked away from when it falls below this (~21 s of watching)...
ATTN_BACK_ABOVE = 0.6       # ...and is worth looking at again above this (after ~13 s away)
ATTN_BORED_MIN = 0.1
ATTN_BORED_LOOKING_RATE = 0.25  # someone looking at the robot bores it 4x slower (whom it talks with: never)
ATTN_BORED_RECOVER = 0.5    # not watched: boredom fades this many seconds per second
ATTN_WAKE_KEEP = 0.3        # starting to look at the robot: boredom x this
ATTN_GLANCE_BACK_P = 0.35   # looking around, bored of someone: this often the next place is a glance back at them
ATTN_BREAK_EVERY_S = (6.0, 12.0) # eye contact: a short look aside this often...
ATTN_BREAK_DEG = (7.0, 13.0)    # ...this far to a side (and 2-6 deg down)...
ATTN_BREAK_S = (0.4, 0.9)       # ...this long
ATTN_ORIENT_S = 2.0         # a voice: look this long where that person was last seen...
ATTN_WHERE_KEEP_S = 600.0   # ...if seen there in this time (people move about)
ATTN_ORIENT_NEAR_DEG = 15.0 # ...and a face that close to that spot is taken for the speaker
ATTN_SCAN_FIX_S = 1.2       # its name, from nowhere it can place: look round in 3 stops this long (move ~0.5 s,
                            # stop, look - vision ignores frames while the head moves fast; a face needs ~2 frames)...
ATTN_SCAN_PAN_DEG = 60.0    # ...ahead, then this far left and right (HEAD_PAN_SOFT; the wide camera sees ~50 further)...
ATTN_SCAN_TILT = 25.0       # ...at the height where people were seen lately (or this before anyone was)...
ATTN_SCAN_UP_DEG = 8.0      # ...plus this (people stand up too)
ATTN_SPOKE_S = 4.0          # someone who just spoke is this long...
ATTN_SPOKE_GAIN = 3.0       # ...this much more interesting
ATTN_VOICE_AGAIN_S = 8.0    # the same voice (or unplaced voices) turns it at most this often; its name always does
ATTN_NIGHT_HOURS = (23, 7)  # night: idle looks are slower and lower (people sleep)
ATTN_NIGHT_SLOWER = 1.6
ATTN_NIGHT_TILT = (-8.0, 10.0)
ATTN_NIGHT_HOME_TILT = 0.0

# ---------- Camera (robot/camera.py): Camera Module 3 Wide (imx708_wide) on the pan-tilt head ----------
CAMERA_MAIN = (2304, 1296)  # big stream (sensor mode 2304x1296: 2x2 binned, full view, supports HDR) for face crops
CAMERA_LORES = (576, 324)   # small stream (YUV420) for finding faces - 1/4 of the big one
CAMERA_FPS = 15
CAMERA_HFOV_DEG = 102.0     # horizontal field of view (Camera Module 3 Wide datasheet); vertical 67 deg
CAMERA_LATENCY_S = 0.05     # fallback frame age; camera.frame_age() measures it per frame (30 ms + exposure / 2)
CAMERA_HDR = False          # sensor HDR (wide_dynamic_range): for ceiling lamps behind faces - try on the Pi first
CAMERA_METERING = 0         # AeMeteringMode: 0 centre-weighted, 1 spot, 2 matrix (libcamera)
CAMERA_CONSTRAINT = 0       # AeConstraintMode: 0 normal, 1 highlight, 2 shadows
CAMERA_MAX_GAIN = 15.5      # imx708 analogue gain at its top (16): with the longest exposure = no more light
FACE_EXPOSURE_TARGET = 115  # face brightness the exposure loop aims for (mean Y, 16-235 scale)
FACE_EXPOSURE_EV = (-1.0, 3.0)   # ExposureValue range it may use (libcamera allows -8..8)
FACE_EXPOSURE_GAIN = 0.5    # share of the error (in EV) corrected per update
FACE_EXPOSURE_EVERY_S = 0.25     # updates at most this often (the camera's own exposure needs a few frames)
FACE_SQUINT_AFTER_S = 3.0   # no face this long -> "squint": every FACE_SQUINT_EVERY_S brighten by
FACE_SQUINT_EVERY_S = 3.0   # FACE_SQUINT_EV for FACE_SQUINT_S, to find faces lost in the dark / against
FACE_SQUINT_EV = 2.0        # a bright ceiling (a backlit face is never detected, so the loop could not start)
FACE_SQUINT_S = 0.8

# ---------- Faces (robot/faces.py) ----------
MODELS_DIR = "~/.local/share/arcby/models"   # ONNX files, fetched by code/get_models.py (not in git)
FACE_DETECTOR = "yunet"     # "scrfd" | "yunet" - bake-off 2026-10-31 (docs/vision.md): YuNet finds faces further
                            # away and faster; SCRFD's landmarks align slightly better (use it on the face region)
FACE_EMBEDDER = "r50_int8"  # "r50_int8" | "r50" | "mbf" | "sface" | "edgeface_xs". 2026-10-31 on the robot's own
                            # face shots (hwtest/faceshots.py compare, enrolled at 1 m only, 1 % of LFW strangers
                            # let in): r50_int8 100 % recognised, margin 0.28; mbf 98 %, margin 0.20. Pi 4,
                            # 2 threads: r50_int8 410 ms / face, mbf 68 ms - so looks run in their own thread
FACE_EMBED_THREADS = 2      # onnxruntime threads for the embedder (detector: FACE_THREADS) - 4 cores in all
FACE_DET_SIZE = (480, 288)  # detector input (multiple of 32); the frame is shrunk to fit. YuNet on the Pi 4
                            # (2 threads, 1536x864 frame): 320x192 = 35 ms, 480x288 = 60 ms (finds a face at
                            # 4 m, ~36 px wide), 640x384 = 91 ms
FACE_THREADS = 2            # onnxruntime threads per model (the Pi 4 has 4 cores)
FACE_MIN_SCORE = 0.5        # detector confidence below this = not a face
FACE_MIN_PX = 10            # faces narrower than this are ignored (too small to recognise)

# ---------- People (robot/people.py, robot/tracker.py) ----------
# one folder per person: vectors + best photo; index.json = names. Vectors only compare with vectors of the
# same model, so each embedder has its own people (the mbf people of 2026-10-31 stay in .../people)
PEOPLE_DIR = f"~/.local/share/arcby/people-{FACE_EMBEDDER}"
FACE_MATCH_SIM = 0.38       # cosine similarity >= this = "same person" (r50_int8). Robot's own family faces
                            # (2026-11-01): same person median 0.60; vs the most alike OTHER family member: brothers
                            # Selim / Emre median 0.29, p95 0.35-0.37, others 0.16-0.21. Which known person a face
                            # is, is decided by the best match (+ 3-look track votes: 98 % right, <1 % wrong); this
                            # threshold keeps a not-yet-known look-alike from being taken for one of them (0.30 was
                            # the brothers' median). LFW strangers: 1 % reach 0.15-0.18
FACE_LEARN_MIN_SIM = 0.40   # learn a new vector only from a confident match...
FACE_LEARN_MAX_SIM = 0.85   # ...and only if it adds something (less similar than this to every stored vector)
PEOPLE_MAX_VECTORS = 40     # per person; the most redundant vector goes when full
FACE_ENROL_MIN_PX = 60      # a new person is stored only from faces at least this wide (in the big stream; ~2 m).
                            # Was 40: a dark wall patch 40 px wide was enrolled at night (2026-10-31)
# face quality = SCRFD's score on the big-stream crop (robot/vision.py). 40 collected robot looks (2026-10-31):
# hands / objects / background 0.00-0.14, very blurred or side-on faces 0.46-0.53, every usable face >= 0.62
FACE_LOOK_MIN_QUALITY = 0.3     # below: not a face - not embedded at all (and no vote)
FACE_CONFIRM_QUALITY = 0.5      # a track counts as a real face (attention may look at it) once a look reached this;
                                # a dark shirt scored 0.49, blurred / turned faces 0.52-0.53
FACE_LEARN_MIN_QUALITY = 0.6    # a new person is enrolled / a vector learnt only from looks this good
PEOPLE_MERGE_LINKS = 1      # tracks that matched both ids (each >= 2 looks, tracker.same_person_evidence) -> merge
                            # without asking, if their vector sets also look alike
PEOPLE_MERGE_ASK_SIM = 0.35 # two ids whose vector sets are this alike (but no track evidence) -> ask the person.
                            # LFW, 120 people enrolled twice (clean + far/dark/noisy), set sim = mean of the 3 best
                            # pairs: same person >= 0.47 for 95 %, different people max 0.29 of 7140 pairs
                            # (2026-10-31) - 0.35 catches every duplicate there with no false suggestion
# robot/gaze.py: is a face turned toward the robot (5 landmarks)? PROVISIONAL - tune on a labelled look / look away
# session (docs/vision.md). Yaw proxy ~0.2 = 20 deg; pitch proxy 0 = straight, + = chin down
GAZE_FULL_YAW = 0.25        # facing() is 1 up to this sideways turn...  Calibrated 2026-11-01 (Selim, ~2 m, 10-20 deg
GAZE_MAX_YAW = 0.45         # ...and falls to 0 here                        off the camera axis, big stream): looking at
GAZE_FULL_PITCH = 0.12      # the robot read yaw 0.0-0.26, pitch -0.14..+0.14; turned away yaw 0.45-9 (profile).
GAZE_MAX_PITCH = 0.25       # The old 0.35 / 0.20 cut his real looks to ~0.4 engagement: sentences dropped.
GAZE_CHECK_S = 0.25         # each confirmed face: landmarks again on the BIG stream this often (SCRFD, 128 px input).
                            # Small-stream landmarks are too noisy: phone vs robot read 0.2-0.4 vs 0.3-0.5 there,
                            # 0.0 vs 0.7-0.8 on the big crop (Selim, 2026-10-31)
GAZE_SMALL_AFTER_S = 1.0    # no big-stream landmarks for this long: fall back to the small-stream ones
GAZE_STALE_S = 0.5          # big-stream landmarks back from the gaze thread (vision.GazeWorker) later than this: dropped
GAZE_TAU_S = 0.7            # Track.engagement = facing() smoothed with this time constant (a glance does not count)
GAZE_LOOKING_ON = 0.5       # Track.looking switches on above this engagement... (Selim 2026-10-31, big-stream
GAZE_LOOKING_OFF = 0.3      # ...and off below this (no flicker)               gaze: robot 0.50-0.66, phone <= 0.07)
# robot/recorder.py + robot/sensors.py: everything the robot senses, as a dataset to train own models on later.
# Local only, no audio (Selim 2026-10-31). Since 2026-11-02 on the 128 GB USB stick (SanDisk 3.2Gen1, ext4, label
# arcby-data, mounted at DATA_DISK by fstab with nofail: the Pi boots without it; usb-storage.quirks=0781:5583:u in
# cmdline.txt - in UAS mode it crashed the Pi's USB controller). ~/.local/share/arcby/data and /collect are symlinks
# to it: without the stick they point nowhere and nothing is recorded - the SD card never fills up.
DATA_DISK = "/mnt/arcby-data"
DATA_DISK_MARK = ".arcby-data"   # this file on the stick = it is mounted (an empty mount point has none)


def data_disk(path=None, mark=None):
    """Is the USB stick there (mounted, with its marker file)?"""
    import os
    return os.path.exists(os.path.join(path or DATA_DISK, mark or DATA_DISK_MARK))


DATA_ON_DISK = data_disk()
DATA_DIR = "~/.local/share/arcby/data"          # -> DATA_DISK/data
# the oldest sessions go when the total passes this: the stick's 107 GB less room for collect / datasets; the SD card
# (no stick, a copy of the code elsewhere) keeps its old 8 GB
DATA_MAX_BYTES = 80_000_000_000 if DATA_ON_DISK else 8_000_000_000
DATA_MIN_FREE_BYTES = 2_000_000_000   # ...and always leave this much of the disk free (collect, tts, logs share it)
DATA_FLUSH_S = 2.0          # gzip streams flushed this often (readable up to here after a power cut)
DATA_RETAIN_EVERY_S = 300   # size limit checked this often
DATA_KEYFRAME_BUSY_S = 1.0  # a small-stream picture at most this often while faces are in view - and only when the
DATA_KEYFRAME_IDLE_S = 10.0 # scene changed (or who is in view): at most this often when nobody is there
DATA_KEYFRAME_CHANGE = 0.06 # scene change vs the last KEPT picture (48x27 grey, normalised, mean abs difference):
                            # a still room 0.00-0.03, someone moving / coming / going 0.07-0.43 (robot, 2026-11-01)
DATA_KEYFRAME_MAX_S = 300.0 # ...but one at least this often, change or not (a timeline)
DATA_FRAMES_IDLE_S = 1.0    # 'frames' records: every frame with faces in view, one per this without
DATA_KEYFRAME_QUALITY = 80  # JPEG quality (576 x 324: ~25-35 KB each)
DATA_IMU_HZ = 25            # IMU samples written per second while the robot moves (it is read at IMU_RATE_HZ)...
DATA_IMU_STILL_HZ = 1       # ...and while it stands still
DATA_IMU_MOVE_DPS = 2.0     # "moving": any gyro axis above this, or the accel changed by DATA_IMU_MOVE_G, or a bump
DATA_IMU_MOVE_G = 0.03
DATA_IMU_MOVE_HOLD_S = 2.0  # keep the fast rate this long after the last movement
DATA_SONAR_HZ = 5           # distance pings per second; written when it changed by DATA_SONAR_CHANGE_CM...
DATA_SONAR_CHANGE_CM = 3.0
DATA_SONAR_IDLE_S = 2.0     # ...or at least this often
DATA_SYSTEM_S = 10.0        # Pi health (temperature, load, memory, disk, under-voltage, fan) this often
COLLECT_DIR = "~/.local/share/arcby/collect"   # alive.py --collect: face crops of every track, local only (-> stick)
COLLECT_EVERY_S = 2.0       # at most one saved look per track this often - and only a NEW kind of look:
# a day's looks of one person are spread over condition bins (distance x head turn x light x looking at the robot,
# 4 x 3 x 3 x 2 = 72): at most COLLECT_PER_BIN_DAY per bin, and only if not too alike a look already kept in it.
# Someone sitting at their desk all day gives a few dozen faces, not thousands; rare conditions still get in.
COLLECT_PER_BIN_DAY = 4
COLLECT_ALIKE_SIM = 0.90    # a look this similar (cosine) to one kept in the same bin today adds nothing
COLLECT_STRANGER_PER_TRACK = 8   # nobody known: at most this many looks per track (they are new people)
COLLECT_MAX_BYTES = 10_000_000_000 if DATA_ON_DISK else 2_000_000_000   # stop saving above this (stick / SD card;
                                    # the oldest days can be deleted by hand)
TRACK_MAX_MISSING_S = 0.8   # a face track ends when unseen this long...
TRACK_FACE_MAX_MISSING_S = 2.5   # ...a confirmed face only after this: the detector loses real faces for 0.8-3 s
                            # and they come back <2 deg away (21 of 22 restarts in the recordings, 2026-11-01)
# robot/static_spots.py: places where the detector keeps "finding" something that never moves and never passes the
# face check (2 such spots made 685 of 823 tracks in the first recordings). Remembered across runs.
STATIC_FILE = "~/.local/share/arcby/static_spots.json"
STATIC_RADIUS_DEG = 2.0     # a spot = where this many still, never-a-face tracks ended within this (head angles)...
STATIC_MIN_HITS = 5
STATIC_KEEP_S = 24 * 3600   # ...within this time (spots not seen for a day are forgotten: the robot moved)
STATIC_MAX_DEV_DEG = 0.5    # "still": the track never moved further than this (real faces: ~1.6 deg median)
STATIC_CHECK_S = 10.0       # a detection on a static spot is checked only this often (someone may stand there)
TRACK_SWAP_LOOKS = 2        # this many looks in a row that clearly show someone else (their similarity >= FACE_MATCH_SIM
TRACK_SWAP_MARGIN = 0.1     # + this) and not the track's person (< FACE_MATCH_SIM): the track jumped to a neighbour's
                            # face - split it. Collected 2026-10-31: 32 of 435 family tracks named 2+ people
TRACK_ID_VOTES = 3          # identity = majority of the last N embeddings of a track (no single-frame decisions)
TRACK_EMBED_EVERY_S = 0.4   # embed an undecided track this often...
TRACK_RECHECK_S = 3.0       # ...and re-check a decided one this often
TRACK_SIZE_RATIO = 1.6      # a sighting joins a track only if their face sizes differ less than this (a huge false box
                            # stole small tracks 30 deg away, 2026-11-01)
TRACK_PATH_N = 900          # still-head positions kept per track (~60 s at 15 fps) for Track.still
TRACK_STILL_S = 10.0        # a track that stays within TRACK_STILL_DEG (90 % of its positions) for this long with a
TRACK_STILL_DEG = 0.4       # still head is a picture, not a person. Positions relative to the background (camera.Background):
                            # the photo 0.12-0.29 deg (p90, whole tracks, no head), Selim sitting still 0.45-0.9, moving
                            # 1-30 (2026-11-01); with the head powered and raw angles both were 1.1-1.6
TRACK_BG_EVERY = 2          # background-relative stillness (camera.Background): measure the camera's true pose from the
                            # picture every this many still frames (~5 ms each on the Pi 4)...
TRACK_BG_REREF_DEG = 4.0    # ...against a reference frame within this much commanded pose (else a new one)...
TRACK_BG_LINK_DEG = 15.0    # ...whose true pose is measured against an older one up to this far away...
TRACK_BG_REFS = 12          # ...keeping this many references...
TRACK_BG_REF_S = 60.0       # ...for this long (the room changes)
TRACK_BG_MIN_PEAK = 0.05    # phase-correlation peak below this: the scene changed, not the camera - new reference
TRACK_BG_MAX_DEG = 3.0      # a "pose error" bigger than this is a bad match (the servos err 1-1.6 deg), not a fact
PEOPLE_PICTURE_TRACKS = 2   # a person seen in this many still tracks and (almost) never moving is a picture on the wall
VISION_STILL_DPS = 15.0     # identify / learn faces only when the head moved slower than this (no blur)
VISION_SLACK_S = 0.15       # while the head moves, the tracker gate grows by head speed x this

# ---------- Bodies (robot/body.py): people, not just faces - docs/vision.md "Bodies" ----------
BODY_ENABLED = True         # pose model on the small stream (alive.py --no-body turns it off)
BODY_INPUT = (320, 192)     # model input (w, h); 2026-11-01 Pi 4 next to alive.py: ~160 ms (448x256 ~275, 576x320 ~300)
BODY_THREADS = 1            # onnxruntime threads for the pose model (1: ~250 ms, 2: ~175 ms but the camera loop fell
                            # from 12-14 to ~9.6 fps, 2026-11-01 - faces and wrist watching need the frames)
BODY_CROP = True            # far faces (<= BODY_CROP_FACE_MAX_PX wide): every other pose run on a big-stream crop
BODY_CROP_FACE_MAX_PX = 30  # under them (robot shots 2026-10-31: 1 m = 15-21 px, 3 m = 10-12 px; whole frame at 3 m
                            # found the body at confidence 0.37-0.70, the crop at 0.89-0.92; same ~250 ms a run)
BODY_CROP_FACES_TALL = 7.0  # far people (robot/body.py crop_for_face): the model on a big-stream crop this many face
BODY_CROP_ABOVE = 0.5       # heights tall, starting this many face heights above the face...
BODY_CROP_MIN_PX = 96       # ...at least this tall (small-stream pixels)
BODY_MIN_SCORE = 0.35       # person confidence below this = nobody (robot keyframes: real people 0.4-0.9)
# ---------- Showing things (robot/show.py; docs/vision.md part S) ----------
SHOW_ENABLED = True         # alive.py: things held up to the robot (needs the bodies; the object model is optional)
# set from the guided showing session 2026-11-02 (870 robot frames, docs/vision.md "Showing things")
SHOW_SIDE_SW = 1.0          # a hand held up: within the shoulders +- this many shoulder widths...
SHOW_STILL_SW = 0.25        # ...still = the wrist moves less than this many shoulder widths...
SHOW_LOOK_S = 0.4           # ...the object model starts looking after this long held still...
SHOW_HOLD_S = 0.5           # ...and "sure" needs it held this long (the session: the 2-of-3 rule does the rest)
SHOW_EVERY_S = 0.3          # object runs at most this often (one at a time, ~250 ms each)
SHOW_CROP_SW = 3.0          # the crop round the wrist: this many shoulder widths wide (big stream)
SHOW_NEAR_SW = 1.5          # the thing's centre within this many shoulder widths (+ half its size) of the wrist
SHOW_ROI_MARGIN = 0.8       # the object model looks at the head + arms + hands box grown by this many shoulder widths
SHOW_MODELS = ("yoloe_home", "yolo26n")    # the first one whose file exists (robot/faces.py MODELS)
SHOW_MIN_SCORE = 0.5        # object model confidence
SHOW_NEED, SHOW_OF = 2, 3   # the same thing in 2 of the last 3 runs
SHOW_KEEP_S = 1.5           # the head stays on it this long after the last sure sighting
SHOW_AGAIN_S = 180.0        # one "showing" event per person and thing this often
SHOW_ASK_S = 4.0            # "bak" / "bu ne": it looks at the hands this long
SHOW_PICTURE_PX = 640       # the picture for the mind: at most this many pixels a side
SESSION_FLAG = "~/scratch/alive/session.json"   # robot/session_rec.py: a guided test session is recording while
SESSION_FPS = 4.0           # it exists ({"dir", "until"}); big-stream frames this many a second + every hand crop
BODY_NMS_IOU = 0.5
BODY_KP_MIN = 0.5           # a keypoint counts as seen above this visibility
BODY_EVERY_S = 0.3          # run the model this often while no face is in view / a body has no face yet...
BODY_EVERY_SLOW_S = 1.0     # ...this often while every body in view has its face...
BODY_EVERY_IDLE_S = 0.6     # ...and this often when nobody (face or body) was seen for BODY_IDLE_AFTER_S
BODY_IDLE_AFTER_S = 5.0
BODY_EVERY_FAST_S = 0.0     # ...and as fast as it can while an arm is up (waving is 2-3 Hz)
BODY_MAX_MISSING_S = 4.0    # a body track ends when unseen this long (out of view a moment: the same person
                            # comes back - 51 of 78 tracks on 2026-11-01 broke in gaps over 1.5 s)
BODY_VISIBLE_S = 0.8        # ...and counts as in view this long after its last detection
BODY_GATE_MIN_DEG = 6.0     # a detection joins a track within max(this, BODY_GATE_SIZE x box width)
BODY_GATE_SIZE = 0.5
BODY_GATE_GROW_DPS = 10.0   # ...plus this x the seconds it was out of view (they moved meanwhile)
BODY_EDGE_PX = 3            # a box this close to the picture's edge is cut off there
BODY_EDGE_DEG = 25.0        # only an arm at a side edge: the face is about this much past that edge
BODY_HINT_MIN_UP_DEG = 8.0  # face hint: at least this far above the shoulders (x2 hips, x3 knees, above the top edge)
BODY_FACE_NEAR_DEG = 6.0    # a face within this of the body's head keypoints is its face
BODY_FACE_FRESH_S = 2.0     # a body that had its face this recently is not searched (the face tracker has it)
BODY_STILL_S = 8.0          # a body that never moved more than BODY_STILL_DEG for this long, never with a face, is a
BODY_STILL_DEG = 1.5        # thing (the gaming chair was taken for a person in the 2026-11-01 keyframes)
BODY_SEEK_MIN_HITS = 2      # seen in this many detections before the head goes looking for its face
BODY_REARM_DEG = 10.0       # a body attention gave up on is looked for again once it moved this far
BODY_WAVE_WINDOW_S = 2.0    # waving: within this...
BODY_WAVE_TURNS = 2         # ...the wrist turns back this often...
BODY_WAVE_AMP = 0.35        # ...over this sideways spread (shoulder widths), wrist above the elbow
BODY_WAVE_FOREARM_UP = 0.7  # ...the forearm rising >= this x its sideways run (~35 deg up; an arm on a desk is ~0)
BODY_WAVE_AGAIN_S = 10.0    # one "waved" event per this, however long they wave
BODY_WRIST_PATCH = 0.8      # a raised wrist is watched every frame in a patch of +-this x shoulder width...
BODY_WRIST_PATCH_MIN_PX = 14     # ...at least this (small-stream pixels)
BODY_WAVE_DIFF = 18         # a pixel "moved" when its grey level changed more than this since the last frame...
BODY_WAVE_MOVED_MIN = 0.03  # ...and the patch counts as moving with at least this share of such pixels
BODY_WAVE_FAST_TURNS = 3    # per-frame watching (15 fps): this many turns of the moving pixels' sideways middle...
BODY_WAVE_FAST_AMP = 0.15   # ...over this spread (patch half-sizes; the moved pixels sit between old and new hand) = waving

# Hand signs (robot/gestures.py): first guesses until Selim's own signs are recorded (hwtest/signs.py)
GESTURE_BECKON_WINDOW_S = 2.5   # "gel gel": within this...
GESTURE_BECKON_TURNS = 3        # ...the wrist turns back this often (toward the shoulder and back, twice)...
GESTURE_BECKON_AMP = 0.3        # ...over this up-down spread (shoulder widths), the forearm up...
GESTURE_BECKON_RATIO = 1.5      # ...up-down at least this x the sideways spread (else it is a wave)...
GESTURE_BECKON_ELBOW = 0.3      # ...and the elbow moving at most this (shoulder widths)
GESTURE_BECKON_AGAIN_S = 8.0    # one "beckoned" event per this
GESTURE_STOP_LEVEL = 0.2        # stop sign: the wrist at most this far below the shoulder (shoulder widths)...
GESTURE_STOP_HOLD_S = 0.7       # ...held this long...
GESTURE_STOP_STILL = 0.25       # ...moving at most this (shoulder widths) - a wave swings 0.35+
GESTURE_STOP_AGAIN_S = 3.0      # one "stop_sign" event per this
ATTN_SEEK_S = 4.0           # attention: look for a body's face this long, then give that body up...
ATTN_SEEK_HEAD_S = 20.0     # ...but its head keypoints (a person for sure: turned away, or too high for the face
                            # detector) are followed this long - 60 % of seeking on 2026-11-01 was at the tilt limit
ATTN_SEEK_SPOT_DEG = 5.0    # ...and any body this close to where it gave up...
ATTN_SEEK_FORGET_S = 60.0   # ...for this long (the chair flickers in and out of the model: new tracks each time)
ATTN_SEEK_NEAR_DEG = 20.0   # holding a just-lost face: only a body whose face hint is this close to it is looked for

# ---------- Voice (robot/tts.py): Gemini TTS, Turkish + English, one voice ----------
SECRETS_FILE = "~/.config/arcby/secrets.env"   # GEMINI_API_KEY=... (chmod 600, never in git)
TTS_MODEL = "cloud:Charon"   # Selim 2026-11-01 "do cloud tts, we want it cheap": Google Cloud TTS Chirp 3 HD Charon -
# ~0.8 s a phrase, ~1M characters a month free then $30/1M, no tags. Before: gemini-3.8-flash-tts (streamed, tags,
# but 100 requests a day per model on Tier 1)
# pronunciation"): the Interactions API - style as an annotation (never read out), inline <tags>, STREAMED: the first
# sound ~2.1 s after the call instead of the whole phrase ~4 s (Pi, 2026-11-01). Before: 2.5-flash-preview-tts
TTS_VOICE = "Charon"        # Selim's pick 2026-10-31 (calm, "informative"); changing it re-makes every cached phrase
TTS_STYLE = "calm, polite and composed young male assistant with a subtle dry humour; natural, unhurried"
TTS_STYLES = {"gemini-2.5-flash-preview-tts": "Say in a calm, polite, composed voice with subtle dry humour"}
# when TTS_MODEL is refused (its daily quota, Google's day in California time): the same voice from these, in order
TTS_FALLBACK_MODELS = ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-2.5-flash-preview-tts")
# (Cloud refused / no login: the Gemini voices, the same Charon)
GCLOUD_KEY_FILE = "~/.config/arcby/gcloud-tts.json"   # service account key (secret, chmod 600) for Cloud TTS
TTS_STREAM_PEAK = 0.9       # a streamed phrase's gain is set before its peak is known: Gemini speech peaks 0.8-0.9
TTS_ESPEAK = False          # no Gemini model at all (no internet / every quota out): silent, not espeak (Selim hates it)
TTS_CACHE_DIR = "~/.local/share/arcby/tts"   # one WAV per (model, voice, style, text): a phrase is paid for once
TTS_CACHE_MAX_BYTES = 200_000_000   # the least recently used phrases go beyond this (every mind reply is new)
TTS_DAILY_REQUESTS = 1000   # Selim 2026-11-01 (was 200: hit by the afternoon - silent): a 3.8 Flash TTS phrase ~0.1 cent, worst case ~1 $/day
TTS_LIMIT_PHRASE = "Bugünlük konuşma sınırım doldu."   # said (cached at start-up) when no voice is left...
TTS_LIMIT_EVERY_S = 600     # ...at most this often
TTS_TIMEOUT_S = 20
TTS_PAUSE_S = 3600          # after "no credit" / not allowed (HTTP 402, 403): every model rests an hour
TTS_RATE_PAUSE_S = 60       # after "too many requests" (HTTP 429, a per-minute limit) - or Google's retryDelay;
                            # a per-day 429 or a model that is gone (404) rests until midnight in California
TTS_DAY_AFTER_429 = 2       # this many 429s in a row from one model (no success between): it rests until Google's
                            # midnight too (its "exceeded your current quota" does not always say "per day")
TTS_AUDIO_TAGS = True       # Gemini 3.8 TTS vocal tags (<chuckle>, <sigh>, <breath>, <short pause>) in the mind's replies and
                            # the fillers; False = all stripped (if Selim hears them read out as words)
TTS_PREPARE_GAP_S = 7.0     # Voice.prepare: at most one Gemini call per this (36 phrases at once hit the 429)
TTS_TIMEZONE = "Europe/Istanbul"   # for the day count and night volume (the Pi's clock may be set to UTC)
TTS_LEVEL_DAY = 0.50        # speech peak (full scale 1; the I2S card has no volume control). Selim 2026-10-31:
TTS_LEVEL_NIGHT = 0.50      # "50% is fixed for now", day and night (was 0.30 / 0.08)
TTS_NIGHT_HOURS = (22, 8)   # night from 22:00 to 08:00
TTS_LEVEL_MAX = 0.70        # never louder than this, whatever asks (slider, --level): the speaker sounded muffled after
                            # hours at level 1.0 (2026-11-01, probably damaged) - full scale overdrives the small speaker
TTS_LEVEL_FILE = "~/.config/arcby/speaker_level.json"   # {"level": 0.5} - set from the dashboard's slider; wins
                            # over TTS_LEVEL_DAY / NIGHT, read before every sentence
TTS_SAY_AS = {"Arcby": "Arkbi", "ARCBY": "Arkbi"}   # spoken spelling (the name is said "Arkbi")
GREET_AWAY_S = 20 * 60      # robot/greeter.py: greet someone it knows when unseen this long (or never yet)
GREET_LONG_AWAY_S = 3 * 3600   # ...and say "welcome back" after this long
GREET_LOOK_S = 1.0          # ...but only once they have looked at the robot this long (Track.looked_for)
GREET_GROUP_S = 2.0         # people ready within this of the first one are greeted in one sentence
GREET_PENDING_S = 10.0      # someone who leaves without looking at the robot is not greeted (out of view this long)
GREET_WAVE_AGAIN_S = 30.0   # a wave is answered at most this often per person (robot/body.py "waved")
GREET_WAVE_NAME_S = 1.5     # a waving body's face not recognised yet: wait this long for its name (2 of 5 waves on
                            # 2026-11-01 got a plain hello, Selim recognised 0.5-1.5 s later), then a plain hello

# ---------- Ears (robot/vad.py, robot/mic.py, robot/stt.py): speech in, Turkish ----------
MIC_CARD = 0                # ALSA card of the I2S mic (googlevoicehat), read as 48 kHz stereo S32_LE, channel 0
MIC_HIGHPASS_HZ = 100       # the I2S mic's signal is 99 % rumble / drift below 80 Hz (2026-11-01): the VAD heard no
                            # speech in it at all (max 0.18) - filtered above 100 Hz it did (0.98)
MIC_GAIN_TARGET = 0.1       # ...and what is left (speech ~ -49 dBFS) is quiet: an automatic gain brings the loud
MIC_GAIN_MAX = 120.0        # parts to this RMS (at most this much gain), before the VAD and the speech-to-text.
                            # x50 was at its limit for Selim's normal (quiet) voice; whispers need more
MIC_VAD_ON = 0.5            # Silero VAD probability: speech starts above this...
MIC_VAD_OFF = 0.35          # ...and counts as silence below this
MIC_VAD_RESET_S = 4.0       # the VAD's memory is reset this often while nothing is being heard (it drifts deaf on
                            # long noise) - whatever the noise scores
MIC_START_S = 0.1           # this much speech in a row starts an utterance (a click does not)
MIC_END_S = 0.45            # this much silence ends it (was 0.6 - every 0.1 s here is 0.1 s before the answer; Selim's
                            # 28 session sentences keep together down to 0.35, longest pause inside one 0.1 s)
MIC_PREROLL_S = 0.3         # kept from before the start (the first syllable)
MIC_MIN_S = 0.16            # less actual speech than this is dropped. 0.25 dropped "Dur" (5-7 speech chunks) in
                            # every replay (2026-11-01); the start still needs MIC_START_S in a row, so clicks stay out
MIC_UNMUTE_HOLD_S = 1.0     # still deaf this long after the robot stopped speaking: up to ~0.7 s of its voice is still
                            # in the speaker's buffers when "done", plus the mic's own buffering
MIC_GAIN_TOO_LOUD_S = 1.0   # this long in a row the limiter had to halve the sound: the gain learns from it even
                            # without the VAD (loud speech at x120 was never "speech" - the gain never came down)
MIC_MAX_S = 12.0            # longer ones are cut
MIC_UTTERANCE_PEAK = 0.5    # an utterance goes to the STT / voice recognition scaled to this peak (-6 dBFS), once
STT_BACKEND = "gemini"      # "gemini" (cloud) | "whisper" (on the Pi). Selim 2026-11-01: Gemini Flash-Lite - fast;
                            # Whisper missed a third of the names. Clips of people talking TO the robot leave it
STT_WHISPER_MODEL = "tiny"  # faster-whisper model in ~/.venvs/arcby-stt (tiny / base / small) - to be measured
STT_GEMINI_MODEL = "gemini-3.8-flash"   # Selim 2026-11-01: on his real voice 0 % WER (Flash-Lite 17 %, "Hayır" -> "hayırsız").
                            # Its default "thinking" (~110 tokens) took 3-4.5 s (2-9 s on the robot); Selim 2026-11-01:
                            # thinking off (STT_GEMINI_THINKING) - "we don't want to lose time there"
STT_DAILY_REQUESTS = 200    # speech-to-text calls a day at most (~$3 / month worst case); then the robot is deaf
STT_LANG = "tr"
STT_LIVE = True             # robot/live_stt.py: the sentence streams to Gemini while it is said - words ~0.4 s after it
                            # ends (clip STT: 2-5 s, measured 2026-11-01); the clip STT stays as the fallback
STT_LIVE_MODEL = "gemini-3.5-transcribe-live"
STT_LIVE_LANGS = ("tr-TR", "en-US")   # without them quiet Turkish came back as Hindi / English
STT_LIVE_DAILY = 1000       # live sessions a day (one per sentence said to the robot; 200 ran out by 17:11 on a test day)
STT_LIVE_WAIT_S = 3.0       # live words not there this long after the sentence ended: the clip STT instead
STT_GEMINI_THINKING = "off"  # "off" = as little as the model allows (3.x: thinkingLevel minimal, 2.5: thinkingBudget 0;
                            # a model that refuses the field is asked again without it, once - then never); None = the
                            # model's default (earlier tests: 3.8 Flash erred like Lite with it off - listen for that);
                            # or a generationConfig "thinkingConfig" dict, e.g. {"thinkingLevel": "low"}
CONV_ANSWER_S = 10.0        # robot/conversation.py: after a question it waits this long for the answer
CONV_SESSION_S = 20.0       # a conversation stays open this long after the last exchange: no need to look any more
CONV_GAZE_START_S = 1.0     # a sentence counts as said TO the robot when someone looked at it this close to its speech
                            # onset (+-; people turn to the robot as they start talking - 0.5 dropped sentences)
CONV_LOOK_KEEP_S = 30.0     # who looked when is kept this long (a 12 s sentence + the worker busy speaking)
CONV_WHOLE_LOOK = 0.8       # a voice print is learnt only when the face looked (alone) this share of the sentence
CONV_ASK_LOOK_S = 1.0       # an unnamed face is asked its name once it has looked at the robot this long
CONV_ASK_NAME_AGAIN_S = 120.0   # a new face that did not tell its name is asked again after this
# robot/bargein.py: someone talks while the robot speaks -> it stops and listens. Fitted on a recording
# 2026-11-01 (Selim over the robot, 1-2 m): no false alarm at 12 dB, one at 9; caught 0.1-1.5 s after he started
BARGE_IN = True
BARGE_THRESHOLD_DB = 10.0   # a band this far over the predicted echo is someone else's sound (3+ bands, 5 of 8 chunks)
BARGE_LAG_S = 0.108         # the echo reaches the mic this long after speakerd's playing time (measured, +-0.4 ms)
BARGE_DECAY_DB = 2.5        # the room's echo tail fades this much per 32 ms
BARGE_COUPLING_DB = (-42.0, -29.4, -15.0, -14.6, -16.6, -14.5, -6.4, -16.3, -15.8)   # speaker -> mic per band
BARGE_NOISE_DB = (-34.7, -39.5, -38.8, -39.3, -38.6, -41.6, -42.3, -42.8, -43.3)     # (learnt on, as it runs)
                            # ^ the defaults; the speaker's own values come from BARGE_TUNE_FILE once it exists
BARGE_TUNE_FILE = "~/.local/share/arcby/barge.json"  # coupling / noise / lag of THIS speaker: code/hwtest/bargecal.py
                            # (nobody talking, once per speaker) + every false alarm found (robot/bargein.py)
BARGE_FALSE_STEP_DB = 6.0   # a false alarm (its own words heard back) raises the bands that fired at most this much
BARGE_COUPLING_MAX_DB = 0.0  # ...never over this: the echo as loud as what is played (a barge-in still needs +10 dB)
CONV_PICTURE_WAIT_S = 0.8  # "bu ne?" / something shown: wait this long for robot/show.py's picture (next frames)
CONV_ECHO_END_S = 0.6      # ...and only when the sound ended this soon after the cut (its voice fades by ~0.5 s;
                            # someone who talked over it goes on): "Merhaba Selim." ended +0.3 s after its cut
CONV_ECHO_FUZZ = 0.75       # a heard word this alike (difflib ratio) to a word it just said counts as its own word
                            # after a barge-in ("çağırma" for its "çağıramam"); "Merhaba Selim." for its "Merhaba Selim!"
BARGE_PREROLL_S = 0.15      # the utterance after a barge-in starts this long before the other voice was first heard
                            # (1 s before the stop took the robot's own words, 0.3 s lost the first syllables)
BARGE_RING_S = 2.0          # the mic kept while the robot speaks; at a barge-in saved with what was played (tuning)
BARGE_SAVE_DIR = "~/scratch/alive/barge"     # those snippets (stereo WAV: left = mic, right = the robot's sound,
BARGE_SAVE_N = 60                            # shifted by BARGE_LAG_S); the newest kept
BARGE_LOOK_S = 1.0          # ...and only when someone looked at the robot this recently (or a conversation is open)
BARGE_LISTEN_S = 1.5        # after a barge-in the ears listen through the mute this long (its voice fading out)
CONV_BARGE_WINDOW_S = 6.0   # an utterance this soon after a barge-in: the robot's own words are taken out of it
CONV_SAVE_DIR = "~/scratch/alive/heard"   # the last CONV_SAVE_N utterances as WAV (heard + dropped, with what the
CONV_SAVE_N = 40                          # STT made of them) - to listen to what the robot hears. Local only.
CONV_FILLER_P = 1.0         # a mind reply that is not cached yet (first sound ~2.5 s after the mind) starts with a cached
                            # filler - always (Selim 2026-11-01: "so in conversation you dont wait too long"; was 0.7);
                            # 0 = never. Fillers are made at start-up, never at run time
VOICE_MODEL = "campplus_zh_en"  # robot/voices.py: speaker recognition (faces.MODELS, kind "voice"); 0.23 s per s of speech
VOICE_MATCH_SIM = 0.5       # a voice is someone's when its 3 closest prints are this alike on average (Selim's quiet
                            # robot-mic clips: 79 % recognised, 0 of 34 other voices taken for him - docs/ears.md)...
VOICE_MARGIN = 0.08         # ...and clearly more alike than the next person's
VOICE_SOLO_SIM = 0.6        # ...but while only ONE person has prints there is no "next person" (the margin means
                            # nothing): then it must be this alike (Selim 2026-11-01: Emre must not pass for Selim)
VOICE_LEARN_SIM = 0.4       # a seen speaker's sentence that is nobody's voice is learnt only this close to their own
VOICE_OTHER_MARGIN = 0.15   # a face looks, the voice is nobody's for sure: still not theirs when another known person's
                            # prints fit it >= VOICE_LEARN_SIM and this much better (Selim 0.47 / Emre 0.26 while Emre
                            # looked, 2026-11-01: 0.13-0.26 apart on the Pi's saved sentences)
VOICE_PARTNER_MIN_PRINTS = 8    # in a talk, a voice under VOICE_LEARN_SIM to the partner's prints (TikTok, the TV, a guest)
                                # is not theirs: ignored - only once the partner has this many prints
                            # prints (or when they have none yet): not from someone talking off camera
VOICE_ALIKE_SIM = 0.92      # a new print this like a kept one adds nothing
VOICE_KEEP = 40             # prints per person (the newest)
VOICE_MIN_S = 0.8           # shorter utterances are not learnt from (a print of "Evet" is mostly noise)
MIND_MODEL = "claude-haiku-5-5"  # robot/mind.py: Claude answers what it hears. Selim 2026-11-01: Haiku 5.5 - ~1 s,
                            # ~$0.0003 / reply ($0.10 / $0.50 per 1M tokens); key ANTHROPIC_API_KEY in SECRETS_FILE
MIND_EFFORT = "low"         # thinking off at low effort: small talk, the answer has to come fast
MIND_MAX_TOKENS = 100       # a reply is one or two spoken sentences (~60 tokens); long ones took 7-19 s to say
MIND_MAX_SENTENCES = 2      # ...and anything longer is cut to this many in code
MIND_DAILY_REQUESTS = 300   # Claude calls a day at most (~$0.10 a day worst case); then the rule-based answers
MIND_IMAGE_DAILY = 50       # pictures of things shown to it (robot/show.py) sent to the mind a day at most (~1000
                            # tokens each); after that it answers from what the object model saw, without the picture
MIND_TIMEOUT_S = 10.0       # no answer by then: the rule-based answer
MIND_FORGET_S = 300.0       # what was said is kept this long after the last exchange (one conversation), then dropped
MIND_MAX_TURNS = 20         # ...and at most this many exchanges of it
MIND_HOUSEHOLD_FILES = (    # the real household - private, never in git: [["Name", "who they are"], ...]
    __file__.rsplit("/code/robot/", 1)[0] + "/private/household.json",   # <repo>/private/ (git-ignored)
    "~/.config/arcby/household.json",
)                           # the first one found wins; without any the example household below is used
MIND_HOUSEHOLD = (          # who the people it knows are (the names come from the people store) - EXAMPLE names
    ("Selim", "built Arcby, designed and printed every part - your creator and boss; address him as Selim"),
    ("Emre", "Selim's younger brother"),
    ("Meltem", "Selim's mother"),
    ("Halil", "Selim's father"),
    ("Doğu", "Selim's cousin, visits now and then"),
)


def _load_household(paths=MIND_HOUSEHOLD_FILES, default=MIND_HOUSEHOLD):
    import json
    import os
    if os.environ.get("ARCBY_NO_PRIVATE"):      # the tests: always the example household
        return default
    for path in paths:
        try:
            with open(os.path.expanduser(path), encoding="utf-8") as f:
                people = tuple((str(n), str(d)) for n, d in json.load(f))
        except (OSError, ValueError, TypeError):
            continue
        if people:
            return people
    return default


MIND_HOUSEHOLD = _load_household()

# ---------- Motors (L298N) - docs/pinout.md ----------
MOTOR_LEFT = dict(forward=27, backward=22, enable=17)     # IN1 pin 13, IN2 pin 15, ENA pin 11
MOTOR_RIGHT = dict(forward=23, backward=24, enable=25)    # IN3 pin 16, IN4 pin 18, ENB pin 22
MOTOR_PINS = (17, 27, 22, 23, 24, 25)
DRIVE_RATE_HZ = 50
DRIVE_MIN = 0.35            # PWM where the wheels start on carpet (measured 0.15-0.35, 2026-10-29);
                            # any non-zero request gets at least this, so small commands still move
DRIVE_MAX = 0.8             # PWM for a full (1.0) request
DRIVE_ACCEL = 2.0           # PWM per second when speeding up - soft start (0 -> 0.5 at once lifted
                            # the nose 15 deg; with 2.0/s: 8 deg)
DRIVE_DECEL = 6.0           # PWM per second when slowing down (safety stops are instant)
DRIVE_START_RIGHT = 0.0     # velocity mode, starting forward from standstill: the right wheel gets this much more PWM
                            # (0 = off; 0.06 was the guess before the loose cable was found - try on the robot first)...
DRIVE_START_S = 0.4         # ...for this long - on the rug the left wheel broke free first and snapped the robot 8-30 deg
                            # right at every start (2026-11-05: -55..-83 deg/s within 0.3 s, base.py + nav recordings)
DRIVE_CMD_TIMEOUT_S = 0.3   # no fresh request for this long -> stop (lost connection, crashed sender)
HEADING_KP = 0.01           # straight driving: PWM steering per degree of heading error (route.py)
HEADING_MAX_STEER = 0.1

# Velocity mode (request_velocity: speed + turn rate in deg/s) - the gyro closes the loop on the turn
# rate: wheel difference = rate / TURN_K (feed-forward) + PI on the measured yaw rate.
TURN_K = 400.0              # deg/s per unit of (right - left) PWM, measured on carpet (384-406)
TURN_KP = 0.002             # PWM difference per deg/s of turn-rate error
TURN_KI = 0.01              # PWM difference per deg of accumulated turn-rate error
TURN_I_MAX = 0.5            # limit of the integral part (PWM difference) - room for floors that turn worse
TURN_I_ERR_MAX = 80.0       # only integrate while the error is below this (deg/s): no wind-up while the
                            # robot is still speeding up into a turn
TURN_MAX_DPS = 300.0        # largest turn rate a request may ask for
TURN_ACCEL_DPS2 = 600.0     # how fast the requested turn rate may change (smooth steering)
TURN_ON_SPOT = "spin"        # standing turns: "spin" = both wheels opposite, about the centre (chosen
                            # 2026-10-29: never slower than SPIN_MIN_DPS, but accurate turns come from
                            # stopping early, and fine aiming from the pan-tilt head); "pivot" = one
                            # wheel (slow turns possible, slight shuffle); "auto" = spin when fast, else pivot
# Spin on both wheels, measured on carpet 2026-10-29 (`hwtest/base.py --only spinramp`):
# 0.18-0.22 PWM stuck, 0.24 breaks loose, 0.26 = ~250 deg/s, then ~590 deg/s per PWM (0.60 = ~460).
# Friction makes it all-or-nothing: both wheels cannot turn slower than ~250 deg/s.
SPIN_START = 0.26           # PWM per wheel for the slowest steady spin
SPIN_MIN_DPS = 250.0        # turn rate at SPIN_START
SPIN_SLOPE = 590.0          # deg/s per extra PWM above SPIN_START
SPIN_RAMP_S = 0.15          # a spin from standstill (or reversing) rises to SPIN_START over this long, not at once: in case
                            # the 2026-11-08 power cut right after a 135 deg spin was the start current (0 = at once)
SPIN_AUTO_UP = 270.0        # "auto": switch pivot -> spin above this request (deg/s) ...
SPIN_AUTO_DOWN = 240.0      # ... and spin -> pivot below this (gap = no flicker)
PIVOT_CREEP_MAX = 0.15      # slow turns on the spot pivot on one wheel, which moves the centre a little;
                            # forward and backward pivots are mixed so the drift stays within this
                            # (PWM x seconds, ~4-6 cm). One-way pivots made A-D-A-D creep forward.
HEADING_RATE_KP = 3.0       # straight (turn rate 0, moving): deg/s of correction per deg of heading error
HEADING_MAX_RATE = 30.0

# ---------- Safety ----------
SAFETY_RATE_HZ = 50
SAFETY_IMU_STALE_S = 0.2    # IMU reading older than this -> stop
SAFETY_BUMP_BLANK_S = 0.3   # ignore bumps until this long after the motors reached their target speed:
                            # starting/stopping jerks read 0.5-1.1 g, steady driving <= 0.26 g
SAFETY_MAX_AGE_S = 0.2      # motors stop if the safety verdict is older than this (safety loop dead)
BUMP_CALM_S = 0.3           # a jolt only counts as a bump if the robot was calm this long before it:
BUMP_CALM_DPS = 40.0        # pitch rate (deg/s) below this = calm. At take-off the nose lifts and lands
                            # ~0.5 s later with a thump (2026-10-29) - not calm, so not a bump.
POWER_POLL_S = 0.5          # how often the Pi's under-voltage alarm is read (robot/power.py)
POWER_UV_HOLD_S = 1.0       # under-voltage this long -> latched "low battery" stop: the Pi buck sags
                            # when the 2S LiPo is down to ~7.0 V (3.5 V/cell) - charge it now
BLACKBOX_S = 3.0            # every safety stop saves the last 3 s of IMU + motor data
BLACKBOX_DIR = "~/scratch/blackbox"
# Power breadcrumb (robot/breadcrumb.py): once a second the drive programs and alive.py write one line - time, throttled
# bits, core volts, applied PWM, what the robot was doing - into a small ring file, each line fsync'ed (a power cut loses
# the page cache: nav.log ended in garbage on 2026-11-08). The next start logs the previous run's last line.
BREADCRUMB_FILE = "~/.local/share/arcby/breadcrumb.ring"
BREADCRUMB_S = 1.0
BREADCRUMB_SLOTS = 900      # 15 min of lines (fixed-size slots of BREADCRUMB_SLOT bytes: the file never grows)
BREADCRUMB_SLOT = 192

# ---------- HC-SR04 distance sensor (3.3 V, ECHO direct) ----------
SONAR_TRIG = 5              # pin 29
SONAR_ECHO = 6              # pin 31

# ---------- Body (code/body_mcp.py: Claude drives the robot) ----------
# No wheel encoders: distance = time x speed, from route.py on carpet (2026-10-29, PWM 0.45 = 45.7 cm/s).
BODY_SPEED = 0.22           # velocity request for move(): PWM 0.35 + 0.45 x 0.22 = 0.45
BODY_CM_PER_S = 45.7
BODY_MAX_CM = 100.0         # longest single move forward
BODY_MAX_BACK_CM = 30.0     # backward is blind (no sensor at the rear)
BODY_STOP_CM = 20.0         # front distance below this (2 pings in a row) -> stop the move
BODY_TURN_DPS = 250.0       # turns on the spot (the slowest steady spin, SPIN_MIN_DPS)
BODY_TURN_LEAD_S = 0.125    # stop this early: the robot coasts after a spin (base.py turn test)

# ---------- Navigation (code/nav.py, robot/floor.py, robot/localmap.py, robot/nav.py; docs/navigation.md) ----------
# Robot frame: origin = middle of the wheel axle on the floor, x forward, y left (cm). From the Fusion parameters
# (2026-11-04): plate 140 x 100 mm centred on the model origin, axle 52 mm from the rear (y = -18 mm), head at
# y = +45 mm, wheels 12 mm wide 7 mm outside the plate, HC-SR04 plate at the front edge.
NAV_FRONT_CM = 9.5          # axle -> front of the sonar plate
NAV_REAR_CM = 5.5           # axle -> rear edge of the plate
NAV_HALF_WIDTH_CM = 7.0     # centre line -> outside of a wheel
NAV_SPIN_RADIUS_CM = 10.5   # farthest corner from the axle middle: what turning on the spot sweeps
NAV_CAM_X_CM = 6.3          # lens ahead of the axle (pan-tilt centre; the tilt arm moves it ~1 cm) - measure
NAV_CAM_H_CM = 14.0         # lens above the floor (measured 2026-10-31 with the head level)
NAV_SONAR_X_CM = 9.0        # sonar face ahead of the axle
NAV_SONAR_HALF_DEG = 12.0   # half the HC-SR04 beam that reliably answers (datasheet 15 deg; thin legs only near 0)
NAV_SONAR_MAX_CM = 200.0    # farther echoes count as "nothing near"
NAV_SONAR_HZ = 12.0         # pings per second (>= 60 ms apart, HC-SR04)
# speeds: request (motors.request_velocity v) and what it is believed to give on carpet. No encoders: these are
# estimates until hwtest/navcal measures them (BODY_SPEED 0.22 = 45.7 cm/s measured 2026-10-29).
NAV_V_CRAWL = 0.04          # PWM 0.37 - just above the dead band
NAV_V_CRUISE = 0.12         # PWM 0.40
NAV_CRAWL_CM_S = 38.3       # navcal 2026-11-04, hard floor (sonar slope, tape-checked)
NAV_CRUISE_CM_S = 40.2      # navcal 2026-11-04 - barely faster than crawl (both just above the dead band)
NAV_V_BACK = -0.04          # backing up (blind behind: only along the way it came, NAV_BACK_CM at most)
NAV_BACK_CM = 10.0
NAV_TURN_DPS = 250.0        # turns on the spot (the slowest steady spin, SPIN_MIN_DPS)
NAV_TURN_SMALL_DEG = 60.0   # turns under this learn their own stop lead (they never reach full spin speed)
NAV_TURN_LEAD_S = 0.125     # stop the spin this early (it coasts), learnt per turn like body_mcp.py
NAV_STEER_DPS = 60.0        # most turn rate asked while driving (gentle curves)
NAV_STEER_K = 1.5           # deg/s of turn rate per deg of heading error while driving
# clearances (gap between the robot's outline and the obstacle)
NAV_STOP_CM = 18.0          # something this close ahead -> stop, look around
NAV_SLOW_CM = 55.0          # closer than this -> crawl
NAV_GO_CM = 40.0            # a direction needs at least this much free way to be chosen
NAV_MARGIN_CM = 4.0         # added to the half width for the corridor the robot drives through
NAV_SPIN_MARGIN_CM = 3.0    # nothing this close to the spin circle -> back up before turning
NAV_LEG_S = 8.0             # drive at most this long, then stop and look around (also refreshes the memory)
NAV_REPLAN_S = 0.5          # while driving, pick the best heading again this often (within NAV_REPLAN_DEG)
NAV_REPLAN_DEG = 30.0
NAV_SEEN_S = 1.0            # drive only through ground the camera saw free in the last second
# the head while navigating
NAV_TILT_DEG = -28.0        # looking down: the picture covers the floor from ~13 cm to ~3 m ahead
NAV_SCAN_PANS = (60.0, 30.0, 0.0, -30.0, -60.0)     # looking around before choosing (head angles, + = left)
NAV_SCAN_SETTLE_S = 0.25    # head still this long before a picture counts
NAV_SCAN_LOOK_S = 0.3       # then look this long at each angle (~4 pictures)
NAV_HEAD_DPS = 120.0        # head speed while navigating
NAV_HEAD_DOWN_S = 1.0       # the head board silent this long (I2C errors) -> navigation stops
# the floor (robot/floor.py): what the floor looks like, learnt from the patch just ahead while the sonar says free
NAV_SEE_CM = 150.0          # trust the floor picture this far (wide lens, 14 cm high: far rows are few pixels)
NAV_FLOOR_CELL = 4          # small-stream pixels per cell side (576 x 324 -> 144 x 81 cells)
NAV_FLOOR_COLS = 36         # rays across the picture
NAV_FLOOR_REF_CM = (5.0, 35.0)      # reference patch: this far beyond the robot's front...
NAV_FLOOR_REF_HALF_CM = 10.0        # ...and this wide each side of the centre line
NAV_FLOOR_RATE = 0.15       # how fast the floor model follows a new reference patch
NAV_FLOOR_MAX_NEW = 0.3     # learn a patch only if at most this share of it looks unlike the floor (else: object)
NAV_FLOOR_MIN = 0.01        # a cell is floor if its colour AND brightness bins hold at least this (of the peak); 0.04 called dark parquet planks obstacles (2026-11-04)
NAV_FLOOR_RUN = 2           # this many non-floor cells in a row (upwards) = an obstacle's foot
NAV_FLOOR_RUN_CM = 1.5      # ...and at least this long on the floor (near the robot 2 cells are ~0.6 cm: one
                            # dark tuft of a shaggy rug; a 1.6 cm cable still counts) - 2026-11-04 drive: 119 -> 14
NAV_EDGE_HALF_DEG = 20.0    # a floor EDGE is judged in the robot's path only: rays within this of straight ahead
NAV_EDGE_MIN_RAYS = 4       # ...at least this many of them...
NAV_EDGE_SHARE = 0.7        # a floor EDGE (rug, other floor): most rays hit...
NAV_EDGE_SPREAD_CM = 15.0   # ...along a straight line (80 % within half this of it)...
NAV_EDGE_FILL = 0.75        # ...and the ground beyond is non-floor nearly everywhere
NAV_EDGE_MAX_SLANT_DEG = 45.0   # ...and the line faces the robot (a wall's foot seen along the wall is no rug edge:
                                # sim, crossing "along" the end wall came 1.5 cm from it, 2026-11-04)
NAV_EDGE_SONAR_PAST_CM = 30.0   # an edge is crossed (crawling) only when the sonar's echo is this much past it (no wall
                                # or box standing at the edge)...
NAV_EDGE_SONAR_MIN_CM = 40.0    # ...and at least this far; was a flat 80 cm, which kept it on the rug with 50 cm of
                                # parquet free in front (2026-11-04)
NAV_CROSS_BAND_CM = 12.0    # while crossing, camera obstacles this close to the edge line are the edge itself
# memory (robot/localmap.py)
NAV_MEMORY_S = 30.0         # obstacle points are forgotten after this long...
NAV_MEMORY_CM = 200.0       # ...or this far away (dead reckoning without encoders drifts)
# trouble
NAV_STUCK_S = 1.5           # driving this long while the sonar distance to something ahead does not shrink...
NAV_STUCK_CM = 3.0          # ...by this much -> stuck (a wheel caught, a rug folded)
NAV_BUMP_RESETS = 3         # bump stops cleared by navigation itself within NAV_BUMP_WINDOW_S; more -> wait for help
NAV_BUMP_WINDOW_S = 300.0
NAV_STUCK_SCANS = 4         # this many looks around in a row without a way out -> wait for help
# going to people (people mode)
NAV_PERSON_SCORE = 0.5      # body detections at least this sure
NAV_PERSON_STOP_CM = 60.0   # stop this far from their feet
NAV_PERSON_GREET_S = 4.0    # look up at them this long
NAV_PERSON_COOLDOWN_S = 30.0    # then leave them alone this long (keep exploring)
NAV_PERSON_LOST_S = 2.0     # no body seen this long -> back to exploring
NAV_RATE_HZ = 20.0          # navigation decisions per second
# following someone who beckoned (follow mode: robot/nav.py FOLLOW, alive.py on a "gel gel" sign)
FOLLOW_STOP_CM = 60.0       # stand this far from them (their feet, or the sonar when they are straight ahead)...
FOLLOW_GO_CM = 25.0         # ...and go again when they are this much farther (no stop-go flicker)
FOLLOW_TURN_DEG = 25.0      # they are more than this to the side: turn on the spot first
FOLLOW_TILT_DEG = 10.0      # the head looks this high at them when the body gives no better idea
FOLLOW_LOST_S = 3.0         # not seen this long -> stop and look around once...
FOLLOW_REFIND_S = 5.0       # ...found again within this -> go on, else stop for good ("lost them")
FOLLOW_LOOK_PANS = (0.0, 40.0, -40.0, 0.0)  # that look around (head angles)
FOLLOW_REFIND_DEG = 25.0    # a new body track this close to where they were lost is them (the track broke)
FOLLOW_PERSON_CM = 165.0    # a whole person this tall: distance from the body's height in the picture
NAV_PORT = 8091             # code/nav.py web page
DRIVE_IN_ALIVE = True       # alive.py opens the motors + IMU + sonar when the battery is on (robot/legs.py; --no-drive)
DRIVE_STOP_HEAR_ALL = True  # while it drives, every sentence goes to speech-to-text - for a stop word ("dur") only, never
                            # answered: nobody looks at a robot whose head is down at the floor (an exception to the
                            # listening rule, robot/conversation.py; False = "dur" works only when said to it)
WANDER_MAX_S = 120.0        # one wander ("Arcby gez", the dashboard, bored) lasts at most this long
BORED_EXPLORE_S = 600.0     # nobody seen this long (faces or moving bodies) -> it wanders by itself (robot/legs.py
                            # Boredom), but only by day (not in TTS_NIGHT_HOURS: the household sleeps)...
BORED_AFTER_STOP_S = 1800.0 # ...and not this soon after a person stopped a drive (voice, stop sign, dashboard)
BORED_ENABLED = True
NAV_REC_FRAME_S = 1.0       # nav.py keeps a camera picture this often while driving (robot/nav_rec.py)

# ---------- Keyboard drive (code/drive.py) ----------
KEY_LEASE_S = 0.6           # a key press drives for this long; holding the key (auto-repeat) renews it
KEY_SPEED = 0.4             # start speed level (0.1-1.0, + / - keys)
KEY_TURN = 0.1              # spin-on-the-spot level (0.1 -> PWM 0.395, ~300 deg/s on carpet)
KEY_TURN_DPS = 90.0         # browser drive: turn rate for A / D (slider)

# ---------- Dashboard edits (robot/commands.py): the People / Heard tabs ----------
COMMAND_DIR = "~/scratch/alive/commands"    # requests from the dashboard, carried out by alive.py (or people_cli.py)
LABELS_FILE = "~/.local/share/arcby/labels.jsonl"   # every label Selim gave there: ground truth for the learning jobs

# ---------- Learning mode (robot/learning.py, code/learn.py; docs/vision.md) ----------
# While the Pi runs on the wall adapter (battery off: no head, no motors) learn.py improves the robot from its own
# recordings. Learnt values go ONLY into TUNED_FILE, ONLY for the keys below, ONLY within these limits - read here at
# start (the last lines of this file). `python3 code/learn.py show / undo KEY` - every change keeps the old value.
LEARN_ENABLED = True        # alive.py starts learn.py on wall power (alive.py --no-learn: never)
LEARN_STATE_FILE = "~/scratch/learn/state.json"   # the dashboard's "Learning" card
LEARN_CACHE_DIR = "/mnt/arcby-data/learn" if DATA_ON_DISK else "~/.local/share/arcby/learn"   # jobs' work (resumable)
TUNED_FILE = "~/.local/share/arcby/tuned.json"
TUNABLE = {                 # key: (lowest, highest) a job may ever set
    "FACE_MATCH_SIM": (0.30, 0.50),
}
LEARN_CHECK_S = 60.0        # alive.py: battery probe this often; off this many times in a row -> learning starts
LEARN_OFF_CHECKS = 2
LEARN_MAX_TEMP_C = 75.0     # learning pauses above this (the Pi throttles at 80)
LEARN_THREADS = 2           # onnxruntime threads for a job (alive.py keeps the rest)


def _tuned_overrides(path=TUNED_FILE, limits=None):
    """{key: value} from TUNED_FILE: tunable keys only, inside their limits. Never in the tests (ARCBY_NO_PRIVATE)."""
    import json
    import math
    import os
    limits = TUNABLE if limits is None else limits
    if os.environ.get("ARCBY_NO_PRIVATE"):
        return {}
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            values = json.load(f).get("values") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    out = {}
    for k, v in values.items():
        if k in limits and isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = min(max(float(v), limits[k][0]), limits[k][1])
    return out


TUNED = _tuned_overrides()  # what learning changed (logs: alive.py prints it at start)
globals().update(TUNED)
