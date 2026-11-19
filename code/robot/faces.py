"""
Faces: find them in a camera frame, line them up, and turn each into a fingerprint (embedding).

Two jobs, two kinds of ONNX model (onnxruntime + numpy + PIL - no OpenCV needed):
  detector  where are the faces? a box + 5 landmarks per face (eye, eye, nose tip, mouth corner x2;
            image-left one first)
  embedder  who is it? an aligned 112 x 112 face -> unit vector. Same person = vectors point the same
            way (cosine similarity near 1), two strangers ~0.

    det = load_detector("scrfd")                # or "yunet"
    emb = load_embedder("mbf")                  # or "edgeface_xs", "sface"
    for face in det.detect(rgb):                # rgb = H x W x 3 uint8 array, R-G-B order
        vec = emb.embed(align(rgb, face.kps))
    similarity = float(vec @ other_vec)

Model files live in config.MODELS_DIR; `python3 code/get_models.py` downloads them (MODELS below).
"""
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import config

# name -> file, where it comes from, sha256, what it is. "derived" files are made from a download
# by code/get_models.py --derive on a machine with the onnx / torch packages (the Mac), then copied.
SPK = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
MODELS = {
    "scrfd": dict(kind="detector", file="det_500m.onnx",
                  zip="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
                  sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
                  note="SCRFD-500M with landmarks (InsightFace buffalo_sc), non-commercial licence"),
    "yunet": dict(kind="detector", file="face_detection_yunet_2023mar_dyn.onnx",
                  derived_from="face_detection_yunet_2023mar.onnx",
                  url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
                      "face_detection_yunet_2023mar.onnx",
                  sha256_source="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                  sha256="29998b39987747d944988d8bd7000c787b9ef321968d39e6dca4302c7d9cad13",
                  note="YuNet (OpenCV Zoo, MIT); published with a fixed 640x640 input, derived copy = any size"),
    "mbf": dict(kind="embedder", file="w600k_mbf.onnx",
                zip="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
                sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
                note="MobileFaceNet trained on WebFace600K (InsightFace buffalo_sc), 512-d, non-commercial"),
    "r50": dict(kind="embedder", file="w600k_r50.onnx",
                zip="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
                sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
                note="ArcFace ResNet-50 trained on WebFace600K (InsightFace buffalo_l), 512-d, non-commercial"),
    "r50_int8": dict(kind="embedder", file="w600k_r50_int8.onnx",
                     derived_from="w600k_r50.onnx, static int8 (QDQ, per channel) calibrated on aligned LFW faces",
                     sha256="5978067788a3ec4f94dc76eb32e6aee343713a5bb2b226577bf1b955fbee9e5d",
                     note="r50 quantised to int8 for the Pi CPU"),
    "silero_vad": dict(kind="vad", file="silero_vad.onnx",
                       url="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
                       sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
                       note="Silero VAD v5 (MIT): is someone speaking, per 32 ms of 16 kHz audio (robot/vad.py)"),
    # speaker recognition (robot/voices.py), sherpa-onnx export; picked 2026-11-01 over WeSpeaker ResNet34 / CAM++ and
    # 3D-Speaker ERes2Net on Selim's robot-mic clips (docs/ears.md)
    "campplus_zh_en": dict(kind="voice", file="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
                           url=SPK + "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
                           sha256="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
                           note="CAM++ (3D-Speaker, Apache 2.0, ~200k Chinese + English speakers), 192-d, input = "
                                "80 log-mel fbank, mean-normalised"),
    # people's bodies (robot/body.py): Ultralytics YOLO26n-pose (AGPL-3.0, fine for a personal robot), 17 COCO
    # keypoints; exported on the Mac (get_models.py --derive --only yolo26n_pose, ultralytics 8.4.197)
    "yolo26n_pose": dict(kind="pose", file="yolo26n-pose-320x192.onnx", input=(320, 192),
                         derived_from="ultralytics yolo26n-pose.pt, ONNX opset 17, imgsz 192x320, simplified",
                         sha256="7559d0bcaaee2edb22bf907ef97d5a50f45fe91377a25299fdb31c0f26750219",
                         note="YOLO26n-pose, people + 17 keypoints, output 1 x 56 x 1260 (NMS done in robot/body.py)"),
    # objects held up to the robot (robot/show.py): YOLO26n detection, 80 COCO classes; exported on the Mac
    # (get_models.py --derive --only yolo26n, ultralytics 8.4.197) - optional: no file = no "showing" by itself
    "yolo26n": dict(kind="objects", file="yolo26n-320.onnx", input=(320, 320), optional=True,
                    derived_from="ultralytics yolo26n.pt, ONNX opset 17, imgsz 320, simplified (onnx 1.23.2, "
                                 "onnxslim 0.1.98; Mac 2026-11-02)",
                    sha256="16669b9fbb352a60b9b2dbeedac9d7904cf0703dc2a6fbeb14ca91e11361b5b3",
                    note="YOLO26n detection, 80 COCO classes, output 1 x 84 x 2100 (NMS in robot/show.py)"),
    # things held up to the robot, the winner of the showing session 2026-11-02 (docs/vision.md "Showing things"):
    # YOLOE-11s (open vocabulary) exported with this home word list - no wearables ("glasses" fired on Selim's own)
    "yoloe_home": dict(kind="objects", file="yoloe-11s-home-320.onnx", input=(320, 320), optional=True,
                       min_score=0.25,
                       classes=("phone", "sheet of paper", "book", "cup", "mug", "bottle", "glass", "remote control",
                                "toy", "stuffed animal", "card", "notebook", "tablet", "drawing", "box", "ball", "pen",
                                "keys", "wallet", "plate", "fruit", "game controller", "photo", "letter"),
                       derived_from="ultralytics yoloe-11s-seg.pt + set_classes(classes), ONNX opset 17, imgsz 320, "
                                    "simplified (ultralytics 8.4.197, onnx 1.23.2; Mac 2026-11-02)",
                       sha256="96755607ae8762331f2900b57383515c759681733cf3d91b63be3018450021b0",
                       note="YOLOE-11s seg, 24 home classes, output 1 x 60 x 2100 (4 box + 24 classes + 32 mask "
                            "coefficients, ignored) + mask protos (ignored); NMS in robot/show.py"),
    "sface": dict(kind="embedder", file="face_recognition_sface_2021dec.onnx",
                  url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
                      "face_recognition_sface_2021dec.onnx",
                  sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
                  note="SFace (OpenCV Zoo, Apache 2.0), 128-d"),
    "edgeface_xs": dict(kind="embedder", file="edgeface_xs_gamma_06.onnx",
                        derived_from="torch.hub otroshi/edgeface edgeface_xs_gamma_06",
                        note="EdgeFace-XS (Idiap, IJCB 2023), 512-d, CC BY-NC-SA 4.0"),
}

# How each embedder wants its 112 x 112 face: channel order and (x - mean) / std.
EMBED_INPUT = {
    "mbf": dict(bgr=False, mean=127.5, std=127.5),      # InsightFace ArcFaceONNX: RGB, (x - 127.5) / 127.5
    "r50": dict(bgr=False, mean=127.5, std=127.5),      # same InsightFace family
    "r50_int8": dict(bgr=False, mean=127.5, std=127.5),
    "edgeface_xs": dict(bgr=False, mean=127.5, std=127.5),  # torchvision ToTensor + Normalize(0.5, 0.5)
    "sface": dict(bgr=False, mean=0.0, std=1.0),        # OpenCV FaceRecognizerSF: RGB, raw 0-255
}

# Where the 5 landmarks of a well-aligned 112 x 112 face sit (InsightFace / ArcFace template).
ARCFACE_KPS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


@dataclass
class Face:
    box: tuple          # x1, y1, x2, y2 in frame pixels
    score: float        # detector confidence 0..1
    kps: np.ndarray     # (5, 2) landmarks in frame pixels

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def centre(self):
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def model_path(name, models_dir=None):
    return os.path.join(os.path.expanduser(models_dir or config.MODELS_DIR), MODELS[name]["file"])


def _session(path, threads):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)          # some exports warn about initializers on every load
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    # idle worker threads sleep instead of busy-waiting: with spinning on, one onnxruntime thread sat at 100 % of a
    # core between frames, starving the camera loop, the ears and the speaker (2026-11-01 review). Same results.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


# ---------- geometry ----------

def fit(rgb, size):
    """Shrink (or grow) the frame to fit inside size = (w, h), keeping its shape; pad right / bottom
    with black. Returns (padded h x w x 3 float32 array, scale frame -> input)."""
    w, h = size
    fh, fw = rgb.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(1, round(fw * scale)), max(1, round(fh * scale))
    img = rgb if (nw, nh) == (fw, fh) else np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((h, w, 3), np.float32)
    out[:nh, :nw] = img
    return out, scale


def nms(boxes, scores, iou=0.4):
    """Indices of the boxes to keep, best first: drop any box overlapping a better one by > iou."""
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= iou]
    return keep


def grid(w, h, stride, per_cell=1):
    """(x, y) of every cell's top-left corner on a stride grid, row by row; each repeated per_cell times."""
    ys, xs = np.mgrid[:h // stride, :w // stride]
    cells = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32) * stride
    return np.repeat(cells, per_cell, axis=0)


def similarity_transform(src, dst):
    """2 x 3 matrix M (rotation + uniform scale + shift) that best maps points src onto dst
    (least squares, Umeyama 1991): dst ~ M @ [x, y, 1]."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    fix = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:        # a mirror image is not allowed
        fix[1, 1] = -1
    rot = u @ fix @ vt
    scale = np.trace(np.diag(sig) @ fix) / s.var(0).sum()
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * rot @ mu_s
    return m


def align(rgb, kps, size=112):
    """Cut the face out so its landmarks land on the ArcFace template: size x size x 3 uint8."""
    m = similarity_transform(kps, ARCFACE_KPS * size / 112)
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]  # PIL wants output pixel -> input pixel
    img = Image.fromarray(rgb).transform((size, size), Image.AFFINE, inv.ravel().tolist(), Image.BILINEAR)
    return np.asarray(img)


def even_light(face, lo=0.01, hi=0.99, target=(20, 235)):
    """Stretch a dark or washed-out aligned face so its brightness spans target (same gain on all three
    colours, so skin tones stay). Percentiles come from the middle of the crop - the face, not the corners."""
    luma = face[16:-16, 16:-16] @ np.array([0.299, 0.587, 0.114])
    p_lo, p_hi = np.quantile(luma, [lo, hi])
    if p_hi - p_lo < 8:                 # flat (blank) crop: nothing to stretch
        return face
    gain = (target[1] - target[0]) / (p_hi - p_lo)
    return np.clip((face.astype(np.float32) - p_lo) * gain + target[0], 0, 255).astype(np.uint8)


# ---------- detectors ----------

class Detector:
    """Common part: fit the frame into the model input, run it, map boxes back to frame pixels."""
    strides = (8, 16, 32)

    def __init__(self, session, size=config.FACE_DET_SIZE, min_score=config.FACE_MIN_SCORE,
                 min_px=config.FACE_MIN_PX):
        if size[0] % 32 or size[1] % 32:
            raise ValueError(f"input size {size} must be a multiple of 32")
        self.session = session
        self.size = tuple(size)
        self.min_score = min_score
        self.min_px = min_px
        self.input_name = session.get_inputs()[0].name

    def detect(self, rgb):
        """All faces in an H x W x 3 uint8 RGB frame, best first."""
        x, scale = fit(rgb, self.size)
        outs = self.session.run(None, {self.input_name: self.prepare(x)[np.newaxis]})
        boxes, scores, kps = self.decode([o.reshape(o.shape[-2:]) for o in outs])
        ok = scores >= self.min_score
        boxes, scores, kps = boxes[ok] / scale, scores[ok], kps[ok] / scale
        faces = []
        fh, fw = rgb.shape[:2]
        for i in nms(boxes, scores):
            x1, y1, x2, y2 = np.clip(boxes[i], 0, [fw, fh, fw, fh])
            if x2 - x1 >= self.min_px and y2 - y1 >= self.min_px:
                faces.append(Face((float(x1), float(y1), float(x2), float(y2)), float(scores[i]), kps[i]))
        return faces


class ScrfdDetector(Detector):
    """SCRFD (InsightFace): 2 anchors per grid cell; outputs per stride: scores, box distances, landmarks."""

    def prepare(self, x):
        return ((x - 127.5) / 128.0).transpose(2, 0, 1)            # RGB

    def decode(self, outs):
        n = len(self.strides)
        boxes, scores, kps = [], [], []
        w, h = self.size
        for k, s in enumerate(self.strides):
            c = grid(w, h, s, per_cell=2)
            d = outs[n + k] * s
            boxes.append(np.concatenate([c - d[:, :2], c + d[:, 2:]], axis=1))
            scores.append(outs[k][:, 0])
            kps.append(c[:, None, :] + outs[2 * n + k].reshape(-1, 5, 2) * s)
        return np.concatenate(boxes), np.concatenate(scores), np.concatenate(kps)


class YunetDetector(Detector):
    """YuNet (OpenCV Zoo, 2023mar): 1 anchor per cell; outputs cls x3, obj x3, bbox x3, kps x3."""

    def prepare(self, x):
        return x[:, :, ::-1].transpose(2, 0, 1)                    # BGR, raw 0-255

    def decode(self, outs):
        n = len(self.strides)
        boxes, scores, kps = [], [], []
        w, h = self.size
        for k, s in enumerate(self.strides):
            c = grid(w, h, s)
            cls, obj = np.clip(outs[k][:, 0], 0, 1), np.clip(outs[n + k][:, 0], 0, 1)
            b = outs[2 * n + k]
            centre = c + b[:, :2] * s
            half = np.exp(b[:, 2:4]) * s / 2
            boxes.append(np.concatenate([centre - half, centre + half], axis=1))
            scores.append(np.sqrt(cls * obj))
            kps.append(c[:, None, :] + outs[3 * n + k].reshape(-1, 5, 2) * s)
        return np.concatenate(boxes), np.concatenate(scores), np.concatenate(kps)


DETECTORS = {"scrfd": ScrfdDetector, "yunet": YunetDetector}


def load_detector(name=config.FACE_DETECTOR, size=config.FACE_DET_SIZE, threads=config.FACE_THREADS,
                  models_dir=None, **kw):
    return DETECTORS[name](_session(model_path(name, models_dir), threads), size=size, **kw)


# ---------- embedders ----------

class Embedder:
    def __init__(self, session, bgr=False, mean=127.5, std=127.5):
        self.session = session
        self.bgr, self.mean, self.std = bgr, mean, std
        self.input_name = session.get_inputs()[0].name

    def embed(self, face112):
        """Aligned 112 x 112 x 3 uint8 RGB face -> unit vector (float32)."""
        x = face112.astype(np.float32)
        if self.bgr:
            x = x[:, :, ::-1]
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[np.newaxis]
        v = self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0].astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def load_embedder(name=config.FACE_EMBEDDER, threads=config.FACE_EMBED_THREADS, models_dir=None):
    return Embedder(_session(model_path(name, models_dir), threads), **EMBED_INPUT[name])
