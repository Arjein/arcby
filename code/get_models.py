#!/usr/bin/env python3
"""
Fetch the face models (robot/faces.py MODELS) into config.MODELS_DIR and check their sha256.

    python3 code/get_models.py              # on the Pi: download what is missing, verify everything
    python3 code/get_models.py --derive OUT # on the Mac (needs: pip install onnx onnxruntime torch timm):
                                            # make the derived files into OUT, then copy them to the Pi:
                                            # scp OUT/*.onnx arjein@raspberrypi.local:.local/share/arcby/models/

Derived files: YuNet with a free input size (the published file is fixed at 640x640), EdgeFace-XS
exported from PyTorch to ONNX, r50 quantised to int8, YOLO26n-pose exported by Ultralytics
(--only yolo26n_pose). Licences differ per model - see the "note" of each entry.
"""
import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import config                                  # noqa: E402
from robot.faces import MODELS                            # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r:
        return r.read()


def download(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    zips = {}
    ok = True
    for name, m in MODELS.items():
        path = os.path.join(models_dir, m["file"])
        if not os.path.exists(path):
            if "zip" in m:
                if m["zip"] not in zips:
                    zips[m["zip"]] = zipfile.ZipFile(io.BytesIO(fetch(m["zip"])))
                z = zips[m["zip"]]
                member = next(n for n in z.namelist() if n.endswith(m["file"]))
                with open(path, "wb") as f:
                    f.write(z.read(member))
            elif "url" in m and "derived_from" not in m:
                with open(path, "wb") as f:
                    f.write(fetch(m["url"]))
            else:
                print(f"{name}: missing {m['file']} - make it with --derive on the Mac and copy it here"
                      + (" (optional)" if m.get("optional") else ""))
                ok = ok and bool(m.get("optional"))
                continue
        want = m.get("sha256")
        got = sha256(path)
        if want and got != want:
            print(f"{name}: CHECKSUM WRONG for {path}\n  want {want}\n  got  {got}")
            ok = False
        else:
            print(f"{name}: ok ({m['file']}, {os.path.getsize(path) // 1024} KB)")
    return ok


def derive_yunet(out):
    """Published YuNet has input 1x3x640x640; give it free height / width so any multiple of 32 runs."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    m = MODELS["yunet"]
    src = fetch(m["url"])
    got = hashlib.sha256(src).hexdigest()
    if got != m["sha256_source"]:
        raise SystemExit(f"yunet source checksum wrong: {got}")
    model = onnx.load_from_string(src)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param, dims[3].dim_param = "h", "w"
    for value in model.graph.output:                     # cls_8 etc.: [1, cells, k]
        value.type.tensor_type.shape.dim[1].dim_param = "cells" + value.name.rsplit("_", 1)[1]
    del model.graph.value_info[:]                        # inner shapes were worked out for 640x640
    path = os.path.join(out, m["file"])
    onnx.save(model, path)
    # same picture through both: the 640x640 original and the derived file at 320x192 + padding
    rng = np.random.default_rng(0)
    x = np.zeros((1, 3, 640, 640), np.float32)
    x[:, :, :192, :320] = rng.random((1, 3, 192, 320), np.float32) * 255
    a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, {"input": x})
    b = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x[:, :, :192, :320]})
    # cells far from the picture's right / bottom edge see the same pixels in both (stride 8: 80 vs 40
    # cells per row); near the edge the padding differs, so compare the top-left 8 x 16 cells only
    ia = [r * 80 + c for r in range(8) for c in range(16)]
    ib = [r * 40 + c for r in range(8) for c in range(16)]
    diff = float(np.abs(a[0][0, ia, 0] - b[0][0, ib, 0]).max())
    print(f"yunet: {path} (top-left cells max diff {diff:.2e}), sha256 {sha256(path)}")


def derive_edgeface(out):
    import numpy as np
    import onnxruntime as ort
    import torch
    m = MODELS["edgeface_xs"]
    model = torch.hub.load("otroshi/edgeface", "edgeface_xs_gamma_06", source="github", pretrained=True,
                           trust_repo=True)
    model.eval()
    x = torch.randn(1, 3, 112, 112)
    path = os.path.join(out, m["file"])
    torch.onnx.export(model, x, path, input_names=["input"], output_names=["embedding"], opset_version=17,
                      dynamo=False)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    print(f"edgeface_xs: {path} (max diff vs torch {np.abs(ref - got).max():.2e}), sha256 {sha256(path)}")


def derive_r50_int8(out, lfw, models_dir):
    """Static int8 (QDQ, per channel) copy of the ArcFace R50 for the Pi CPU, calibrated on 200 aligned LFW faces
    (people 1000+ by name - hwtest/faceshots.py compare uses the first 600 as strangers). The published model is
    opset 11, which has no per-channel DequantizeLinear: converted to opset 13 first. BatchNorms stay float."""
    import tempfile

    import numpy as np
    import onnx
    from onnx import version_converter
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from hwtest.faceshots import lfw_faces
    from robot import faces
    src = os.path.join(models_dir, MODELS["r50"]["file"])
    if not os.path.exists(src):
        z = zipfile.ZipFile(io.BytesIO(fetch(MODELS["r50"]["zip"])))
        with open(src, "wb") as f:
            f.write(z.read(MODELS["r50"]["file"]))
    det = faces.load_detector("scrfd", size=(256, 256), models_dir=models_dir)
    calib = lfw_faces(Path(lfw).expanduser(), 200, det, skip=1000)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(((a.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None] for a in calib)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input.1": x}

    path = os.path.join(out, MODELS["r50_int8"]["file"])
    with tempfile.TemporaryDirectory() as tmp:
        op13, pre = os.path.join(tmp, "op13.onnx"), os.path.join(tmp, "pre.onnx")
        onnx.save(version_converter.convert_version(onnx.load(src), 13), op13)
        quant_pre_process(op13, pre)
        quantize_static(pre, path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                        op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    print(f"r50_int8: {path} ({len(calib)} calibration faces), sha256 {sha256(path)}")


def derive_yolo_pose(out):
    """YOLO26n-pose (people + 17 keypoints, robot/body.py) to ONNX at the robot's input size. Needs
    pip install ultralytics==8.4.197 onnx onnxslim (AGPL-3.0: fine for a personal robot)."""
    import shutil
    from ultralytics import YOLO
    m = MODELS["yolo26n_pose"]
    w, h = m["input"]
    path = YOLO("yolo26n-pose.pt").export(format="onnx", imgsz=[h, w], simplify=True, dynamic=False, opset=17)
    dst = os.path.join(out, m["file"])
    shutil.move(path, dst)
    got = sha256(dst)
    print(f"yolo26n_pose: {dst} sha256 {got}" + ("" if got == m["sha256"] else
          f" (differs from the recorded {m['sha256'][:12]}... - other ultralytics / onnx version? check, then update)"))


def derive_yolo_objects(out):
    """YOLO26n detection (things held up to the robot, robot/show.py) to ONNX 320 x 320. Needs
    pip install ultralytics==8.4.197 onnx onnxslim (AGPL-3.0: fine for a personal robot)."""
    import shutil
    from ultralytics import YOLO
    m = MODELS["yolo26n"]
    w, h = m["input"]
    path = YOLO("yolo26n.pt").export(format="onnx", imgsz=[h, w], simplify=True, dynamic=False, opset=17)
    dst = os.path.join(out, m["file"])
    shutil.move(path, dst)
    got = sha256(dst)
    print(f"yolo26n: {dst} sha256 {got}" + ("" if got == m["sha256"] else
          " - put this sha256 into robot/faces.py MODELS['yolo26n'], then copy the file to the Pi's models folder"))


def derive_yoloe_home(out):
    """YOLOE-11s with the home word list (robot/show.py, the showing session winner) to ONNX 320 x 320. Needs
    pip install ultralytics==8.4.197 onnx onnxslim (it also fetches the text encoder once)."""
    import shutil
    from ultralytics import YOLOE
    m = MODELS["yoloe_home"]
    w, h = m["input"]
    model = YOLOE("yoloe-11s-seg.pt")
    names = list(m["classes"])
    model.set_classes(names, model.get_text_pe(names))
    path = model.export(format="onnx", imgsz=[h, w], simplify=True, dynamic=False, opset=17)
    dst = os.path.join(out, m["file"])
    shutil.move(path, dst)
    got = sha256(dst)
    print(f"yoloe_home: {dst} sha256 {got}" + ("" if got == m["sha256"] else
          " (differs from the recorded one - another ultralytics / onnx version? check, then update)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.expanduser(config.MODELS_DIR), help="models folder")
    ap.add_argument("--derive", metavar="OUT", help="make the derived models into OUT (Mac)")
    ap.add_argument("--lfw", default="~/scratch/datasets/lfw", help="with --derive: LFW folder (r50_int8 calibration)")
    ap.add_argument("--only", choices=("yunet", "edgeface_xs", "r50_int8", "yolo26n_pose", "yolo26n", "yoloe_home"),
                    help="with --derive: just this one "
                    "(edgeface_xs runs the model code of github.com/otroshi/edgeface through torch.hub)")
    args = ap.parse_args()
    if args.derive:
        os.makedirs(args.derive, exist_ok=True)
        if args.only in (None, "yunet"):
            derive_yunet(args.derive)
        if args.only in (None, "edgeface_xs"):
            derive_edgeface(args.derive)
        if args.only in (None, "r50_int8"):
            derive_r50_int8(args.derive, args.lfw, args.dir)
        if args.only == "yolo26n_pose":         # only on request: needs ultralytics (external code)
            derive_yolo_pose(args.derive)
        if args.only == "yolo26n":              # the same
            derive_yolo_objects(args.derive)
        if args.only == "yoloe_home":           # the same
            derive_yoloe_home(args.derive)
        return
    sys.exit(0 if download(args.dir) else 1)


if __name__ == "__main__":
    main()
