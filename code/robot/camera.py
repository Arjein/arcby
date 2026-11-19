"""
Camera Module 3 Wide (imx708_wide) on the pan-tilt head, through picamera2.

Two streams from one sensor mode (2304x1296, full view):
  lores  576x324 YUV420 - small and fast: find faces here
  main   2304x1296 RGB  - 4x the pixels: cut faces out of this one to recognise them (a face 40 px wide in
         lores is 160 px here; real robot photos were too small to recognise well, docs/vision.md)

    cam = Camera()                      # FakePicamera2-compatible: Camera(picam=...) in tests
    cam.start()
    with cam.frame() as f:              # holds the camera buffer until the block ends - keep it short
        rgb = f.lores_rgb()             # 324 x 576 x 3
        crop, (ox, oy, s) = f.crop_main(box_in_lores, pad=0.4)   # main-stream pixels around a face
    cam.set_ev(1.0)                     # exposure compensation (FaceExposure decides)

FaceExposure keeps faces properly exposed: when a face is seen it steers ExposureValue so the face (not
the bright ceiling lamp behind it) gets the target brightness; when no face has been seen for a while it
"squints" - brightens for a moment every few seconds - to find faces too dark to be detected at all.

Geometry: angles_of() turns a pixel into degrees from the camera axis (pinhole model, CAMERA_HFOV_DEG).
"""
import math
import subprocess
import time
from contextlib import contextmanager

import numpy as np

from . import config


# ---------- geometry ----------

def focal_px(width, hfov_deg=config.CAMERA_HFOV_DEG):
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def angles_of(x, y, width, height, hfov_deg=config.CAMERA_HFOV_DEG):
    """Degrees from the camera axis to pixel (x, y): (right +, up +)."""
    f = focal_px(width, hfov_deg)
    return (math.degrees(math.atan((x - width / 2) / f)),
            math.degrees(math.atan((height / 2 - y) / f)))


def size_deg(width_px, frame_width, hfov_deg=config.CAMERA_HFOV_DEG):
    """Angular size of something width_px wide near the middle of the picture."""
    return math.degrees(2 * math.atan(width_px / 2 / focal_px(frame_width, hfov_deg)))


def head_target(x, y, width, height, pose):
    """Head angles (pan + = left, tilt + = up) that would centre pixel (x, y), given the head pose
    (pan, tilt) when the frame was taken."""
    ax, ay = angles_of(x, y, width, height)
    return pose[0] - ax, pose[1] + ay


# ---------- pixels ----------

def yuv420_to_rgb(buf, width, height):
    """picamera2 YUV420 (I420: Y plane, then U and V at quarter size; rows of `width`) -> H x W x 3 RGB.
    Rec.709 limited range (16-235), picamera2's default for video configurations."""
    buf = np.asarray(buf)
    y = buf[:height, :width].astype(np.float32)
    chroma = buf[height:height * 3 // 2].reshape(-1)
    q = (width // 2) * (height // 2)
    u = chroma[:q].reshape(height // 2, width // 2).astype(np.float32) - 128
    v = chroma[q:2 * q].reshape(height // 2, width // 2).astype(np.float32) - 128
    u = u.repeat(2, axis=0).repeat(2, axis=1)
    v = v.repeat(2, axis=0).repeat(2, axis=1)
    yy = (y - 16) * 1.164
    rgb = np.stack([yy + 1.793 * v, yy - 0.213 * u - 0.533 * v, yy + 2.112 * u], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mean_luma(y_plane, box):
    """Mean Y (16-235) inside box = (x1, y1, x2, y2) of the Y plane; the middle half only (skin, not hair
    or the background around the head)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a = y_plane[int(y1 + h * 0.25):int(y2 - h * 0.25) + 1, int(x1 + w * 0.25):int(x2 - w * 0.25) + 1]
    return float(a.mean()) if a.size else 0.0


_PEAK = {}


def lowpass(shape, width, cutoff=0.1):
    """Gaussian weights (sigma = cutoff cycles / pixel) for an rfft2 spectrum of a (h, width) image."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    return np.exp(-(fx ** 2 + fy ** 2) / (2 * cutoff ** 2))


def phase_shift(ref_spec, spec, weights=None):
    """How far the picture moved from the reference to now (phase correlation of two rfft2 spectra of windowed,
    same-size grey images, even width): ((dx, dy) pixels, + = content moved right / down; peak 0..1 = how sure).
    weights (lowpass()): whitened high frequencies are mostly noise - with them the error was 3x bigger."""
    cross = spec * np.conj(ref_spec)
    cross /= np.abs(cross) + 1e-9
    w = (ref_spec.shape[1] - 1) * 2
    if weights is not None:
        cross *= weights
    r = np.fft.irfft2(cross, s=(ref_spec.shape[0], w))
    h = r.shape[0]
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    best = 1.0                              # the peak of a perfect match (the weights' own transform at 0)
    if weights is not None:
        key = (id(weights), r.shape)
        if key not in _PEAK:
            _PEAK.clear()
            _PEAK[key] = float(np.fft.irfft2(weights, s=r.shape)[0, 0])
        best = _PEAK[key]
    peak = float(r[iy, ix])

    def sub(c, m, p):                       # parabola through the peak and its neighbours: sub-pixel
        d = m - 2 * c + p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (m - p) / d
    dx = ix + sub(peak, r[iy, ix - 1], r[iy, (ix + 1) % w])
    dy = iy + sub(peak, r[iy - 1, ix], r[(iy + 1) % h, ix])
    return ((dx + w / 2) % w - w / 2, (dy + h / 2) % h - h / 2), peak / best


class Background:
    """Where the camera REALLY pointed, from the picture itself. The servos' pose error makes everything in view
    wobble 1.1-1.6 deg while the head 'holds still' - a framed photo on the wall then moved like a sitting person
    (2026-11-01). The background moves exactly as the camera does: phase correlation of the (2x smaller) grey
    frame against a reference frame taken near the same commanded pose gives the true rotation since then, so

        corr = bg.update(frame.y_plane, pose, t)    # (dpan, dtilt) true pose - commanded pose, or None (unsure)
        face_pan + corr[0], face_tilt + corr[1]     # the face's position relative to the background

    stays put for a photo (within the detector's own noise) and moves for a person. Up to TRACK_BG_REFS references
    (one per TRACK_BG_REREF_DEG of commanded pose, so looking away and back returns to the same one and errors do
    not add up); a reference older than TRACK_BG_REF_S, or that no longer matches (lights, someone walked
    through), is taken again. Measures every TRACK_BG_EVERY-th call (None in between)."""

    def __init__(self, every=config.TRACK_BG_EVERY, reref_deg=config.TRACK_BG_REREF_DEG, refs=config.TRACK_BG_REFS,
                 ref_s=config.TRACK_BG_REF_S, min_peak=config.TRACK_BG_MIN_PEAK, max_deg=config.TRACK_BG_MAX_DEG):
        self.every, self.reref_deg, self.max_refs, self.ref_s = every, reref_deg, refs, ref_s
        self.min_peak, self.max_deg = min_peak, max_deg
        self.refs = []                      # [commanded pose, true pose, spectrum, t]
        self.corr = (0.0, 0.0)              # the last measured correction (a new reference starts from it)
        self.peak = 0.0
        self._n = 0
        self._win = self._weights = None

    def _spectrum(self, y_plane):
        y = np.asarray(y_plane, np.float32)
        h, w = (y.shape[0] // 2) * 2, (y.shape[1] // 2) * 2
        g = y[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        if self._win is None or self._win.shape != g.shape:
            self._win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
            self._weights = lowpass(g.shape, g.shape[1])
        g = (g - g.mean()) * self._win
        return np.fft.rfft2(g), g.shape[1]

    def update(self, y_plane, pose, t):
        self._n += 1
        if self.every > 1 and self._n % self.every:
            return None
        spec, width = self._spectrum(y_plane)
        refs = [r for r in self.refs if t - r[3] < self.ref_s]
        if not refs:                                    # the very first frame: the commanded pose is the baseline
            self._new_ref(pose, (0.0, 0.0), spec, t, refs)
            return self.corr
        ref = min(refs, key=lambda r: math.hypot(r[0][0] - pose[0], r[0][1] - pose[1]))
        corr = self._measure(ref, spec, width, pose)
        if math.hypot(ref[0][0] - pose[0], ref[0][1] - pose[1]) > self.reref_deg:
            # a new place: its reference gets its true pose measured against the nearest one (so references agree
            # with each other); too far / no match: from the last correction, and this frame is not trusted
            self._new_ref(pose, corr if corr is not None else self.corr, spec, t, refs)
        elif corr is None:                              # the scene changed (lights, someone walked through)
            self._new_ref(pose, self.corr, spec, t, [r for r in refs if r is not ref])
        if corr is not None:
            self.corr = corr
        return corr

    def _measure(self, ref, spec, width, pose):
        """True pose - commanded pose for this frame, measured against reference ref; None when unsure."""
        if math.hypot(ref[0][0] - pose[0], ref[0][1] - pose[1]) > config.TRACK_BG_LINK_DEG:
            return None
        (dx, dy), self.peak = phase_shift(ref[2], spec, self._weights)
        k = math.degrees(1.0 / focal_px(width))         # deg per pixel near the middle (shifts are small)
        corr = (ref[1][0] + dx * k - pose[0], ref[1][1] + dy * k - pose[1])
        if self.peak < self.min_peak or math.hypot(*corr) > self.max_deg:
            return None
        return corr

    def _new_ref(self, pose, corr, spec, t, refs):
        refs.append([tuple(pose), (pose[0] + corr[0], pose[1] + corr[1]), spec, t])
        self.refs = refs[-self.max_refs:]


# ---------- exposure ----------

def saturated(metadata, fps=config.CAMERA_FPS, max_gain=config.CAMERA_MAX_GAIN):
    """The sensor is at its longest exposure (the frame time) and its highest analogue gain: more light cannot
    be bought with the ExposureValue. False when the metadata does not say."""
    try:
        return (metadata["ExposureTime"] >= 0.97e6 / fps and metadata["AnalogueGain"] >= max_gain)
    except (TypeError, KeyError):
        return False


class FaceExposure:
    """Decides the camera's ExposureValue from the faces it sees. update() returns a new EV or None."""

    def __init__(self, base_ev=0.0, target=config.FACE_EXPOSURE_TARGET, ev_range=config.FACE_EXPOSURE_EV,
                 gain=config.FACE_EXPOSURE_GAIN, every_s=config.FACE_EXPOSURE_EVERY_S):
        self.base = base_ev
        self.ev = base_ev
        self.target, self.ev_range, self.gain, self.every_s = target, ev_range, gain, every_s
        self.last_face_t = None
        self.last_change_t = -math.inf
        self.squinting_since = None
        self.next_squint_t = None
        self.squint_ev = None       # EV of the frames taken while squinting (so a face found then counts)

    def update(self, face_luma, t, metadata=None):
        """face_luma = mean Y of the main face now, or None when no face is seen; metadata = the frame's
        (Frame.metadata), to know when the sensor has nothing left to give. Returns the EV to set (only when it
        should change), else None."""
        flat_out = saturated(metadata)
        if face_luma is not None:
            self.last_face_t = t
            self.squinting_since = self.next_squint_t = None
            if t - self.last_change_t < self.every_s:
                return None
            err = math.log2(self.target / max(face_luma, 4.0))      # stops of light missing
            if abs(err) < 0.15 or (flat_out and err > 0):
                return None                  # longest exposure + most gain already: a higher EV only winds up
            return self._set(self.ev + self.gain * err, t)
        if self.last_face_t is None:
            self.last_face_t = t                     # just started: wait the same delay before squinting
        since = t - self.last_face_t
        if since < config.FACE_SQUINT_AFTER_S:
            return None                              # face just lost: keep its exposure for a moment
        if self.next_squint_t is None:
            self.next_squint_t = t
        if flat_out and self.squinting_since is None:
            # dark room: brightening cannot find a face the sensor already gives everything for (it toggled
            # 0 <-> +2 every 3 s all night, 2026-11-01) - stay at the normal exposure
            if abs(self.ev - self.base) > 0.01 and t - self.last_change_t >= self.every_s:
                return self._set(self.base, t)
            return None
        if self.squinting_since is not None:
            if t - self.squinting_since >= config.FACE_SQUINT_S:
                self.squinting_since = None
                self.next_squint_t = t + config.FACE_SQUINT_EVERY_S
                return self._set(self.base, t)
            return None
        if t >= self.next_squint_t:
            self.squinting_since = t
            return self._set(self.base + config.FACE_SQUINT_EV, t)
        if abs(self.ev - self.base) > 0.01 and t - self.last_change_t >= self.every_s:
            return self._set(self.base, t)          # no face: back to normal exposure
        return None

    def _set(self, ev, t):
        ev = min(max(ev, self.ev_range[0]), self.ev_range[1])
        if abs(ev - self.ev) < 0.01:
            return None
        self.ev, self.last_change_t = ev, t
        return ev


# ---------- the camera ----------

def set_sensor_hdr(on):
    """Switch the imx708's own HDR (must be done while no program has the camera open). Returns True if
    the control was found and set."""
    for n in range(8):
        dev = f"/dev/v4l-subdev{n}"
        try:
            ctrls = subprocess.run(["v4l2-ctl", "-d", dev, "-l"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        if "wide_dynamic_range" in ctrls:
            r = subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"wide_dynamic_range={int(bool(on))}"],
                               capture_output=True, timeout=5)
            return r.returncode == 0
    return False


class Frame:
    """One camera frame: the small stream converted on demand, crops of the big one."""

    def __init__(self, request, t, lores_size, main_size):
        self.request, self.t = request, t
        self.lores_size, self.main_size = lores_size, main_size
        self._yuv = request.make_array("lores")
        self._rgb = None
        self._md = None

    @property
    def metadata(self):
        """libcamera's numbers for this frame: ExposureTime, AnalogueGain, Lux, LensPosition... ({} if none)."""
        if self._md is None:
            try:
                self._md = self.request.get_metadata()
            except Exception:  # noqa: BLE001 - fakes
                self._md = {}
        return self._md

    @property
    def y_plane(self):
        w, h = self.lores_size
        return self._yuv[:h, :w]

    def lores_rgb(self):
        if self._rgb is None:
            self._rgb = yuv420_to_rgb(self._yuv, *self.lores_size)
        return self._rgb

    def crop_main(self, box, pad=0.4):
        """Cut box (lores pixels) out of the big stream, grown by pad x its size on every side.
        Returns (crop RGB, (x0, y0, scale)): main pixel = (lores pixel x scale) - (x0, y0) inside the crop."""
        main = self.request.make_array("main")
        s = self.main_size[0] / self.lores_size[0]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        mx1 = max(0, int((x1 - pad * w) * s))
        my1 = max(0, int((y1 - pad * h) * s))
        mx2 = min(self.main_size[0], int((x2 + pad * w) * s) + 1)
        my2 = min(self.main_size[1], int((y2 + pad * h) * s) + 1)
        return np.ascontiguousarray(main[my1:my2, mx1:mx2, :3]), (mx1, my1, s)


def frame_age(request, now_ns=None):
    """Seconds between the middle of this frame's exposure and now. On the Pi SensorTimestamp (CLOCK_MONOTONIC
    ns) marks the end of exposure / start of readout: measured 30 ms before capture_request() returns with a
    66 ms exposure, so it cannot be the start. Falls back to config.CAMERA_LATENCY_S without metadata."""
    try:
        md = request.get_metadata()
        ts, exposure_us = md["SensorTimestamp"], md.get("ExposureTime", 0)
    except Exception:  # noqa: BLE001 - fake or missing metadata
        return config.CAMERA_LATENCY_S
    now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC) if now_ns is None else now_ns
    age = (now_ns - ts) / 1e9 + exposure_us / 2e6
    return age if 0.0 <= age <= 0.5 else config.CAMERA_LATENCY_S


class Camera:
    def __init__(self, picam=None, main=config.CAMERA_MAIN, lores=config.CAMERA_LORES, fps=config.CAMERA_FPS,
                 hdr=config.CAMERA_HDR, clock=time.monotonic):
        self._picam_arg = picam
        self.main_size, self.lores_size, self.fps, self.hdr = tuple(main), tuple(lores), fps, hdr
        self._clock = clock
        self.cam = None
        self.ev = 0.0

    def start(self):
        if self._picam_arg is None:
            # always set it: the sensor keeps the last value, also from a program that crashed
            if not set_sensor_hdr(self.hdr) and self.hdr:
                raise RuntimeError("camera HDR requested but wide_dynamic_range could not be set")
            from picamera2 import Picamera2
            self.cam = Picamera2()
        else:
            self.cam = self._picam_arg
        try:
            self._configure()
        except Exception:
            self.close()
            raise
        return self

    def _configure(self):
        frame_us = int(1e6 / self.fps)
        # "BGR888" in picamera2 means R, G, B byte order in the numpy array (its names follow libcamera's
        # little-endian pixel formats): main arrays come out RGB
        cfg = self.cam.create_video_configuration(
            main={"size": self.main_size, "format": "BGR888"},
            lores={"size": self.lores_size, "format": "YUV420"},
            sensor={"output_size": self.main_size}, buffer_count=4,
            controls={"FrameDurationLimits": (frame_us, frame_us)})
        self.cam.configure(cfg)
        self.cam.start()
        self.cam.set_controls({"AeMeteringMode": config.CAMERA_METERING,
                               "AeConstraintMode": config.CAMERA_CONSTRAINT, "ExposureValue": 0.0})
        try:
            self.cam.set_controls({"AfMode": 2})        # continuous autofocus
        except Exception:  # noqa: BLE001 - fixed focus is fine too
            pass

    @contextmanager
    def frame(self):
        req = self.cam.capture_request()
        t = self._clock() - frame_age(req)
        try:
            yield Frame(req, t, self.lores_size, self.main_size)
        finally:
            req.release()

    def set_ev(self, ev):
        self.cam.set_controls({"ExposureValue": float(ev)})
        self.ev = ev

    def close(self):
        if self.cam is not None:
            try:
                self.cam.stop()
            finally:
                self.cam.close()
                self.cam = None
        if self.hdr and self._picam_arg is None:
            set_sensor_hdr(False)
