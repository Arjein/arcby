#!/usr/bin/env python3
"""
Make phone photos safe and small for the public repo.

Phone photos carry EXIF metadata: GPS position (= the home address), phone model, date and
time. This script writes a clean copy of each photo into docs/images/:
  - every bit of metadata dropped (EXIF, GPS, XMP, ICC comments...), the picture turned upright first
  - longest side at most 1800 px, JPEG quality 85 (a few hundred KB each instead of 3-5 MB)

    python3 tools/clean_photos.py ~/Desktop/arcby-photos/*.jpg
    python3 tools/clean_photos.py IMG_1234.HEIC --name hero      # -> docs/images/hero.jpg

HEIC (iPhone) files are converted first with macOS `sips`. Needs Pillow (`pip install pillow`).
Then check: python3 tools/clean_photos.py --check docs/images/*.jpg
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
MAX_SIDE = 1800


def load(path):
    if path.suffix.lower() in (".heic", ".heif"):
        tmp = Path(tempfile.mkdtemp()) / (path.stem + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)],
                       check=True, capture_output=True)
        path = tmp
    return Image.open(path)


def clean(src, name=None):
    img = ImageOps.exif_transpose(load(src))          # upright, before the orientation tag is dropped
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    bare = Image.new("RGB", img.size)                 # a fresh image: no info dict, no EXIF, nothing carried over
    bare.paste(img)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{name or src.stem.lower().replace(' ', '-')}.jpg"
    bare.save(dst, "JPEG", quality=85, optimize=True, progressive=True)
    return dst


def has_metadata(path):
    img = Image.open(path)
    found = [k for k in ("exif", "xmp", "icc_profile", "comment", "photoshop") if img.info.get(k)]
    if img.getexif():
        found.append("exif-tags")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--name", help="output name (one photo only), e.g. hero")
    ap.add_argument("--check", action="store_true", help="only report metadata left in the given files")
    a = ap.parse_args()
    if a.name and len(a.photos) != 1:
        sys.exit("--name works with one photo at a time")
    bad = 0
    for p in a.photos:
        if a.check:
            left = has_metadata(p)
            bad += bool(left)
            print(f"{'METADATA ' + ','.join(left) if left else 'clean   '}  {p}")
            continue
        dst = clean(p, a.name)
        left = has_metadata(dst)
        bad += bool(left)
        print(f"{p.name} -> {dst.relative_to(OUT.parent.parent)}  {dst.stat().st_size // 1024} KB"
              f"{'  (metadata left: ' + ','.join(left) + ')' if left else '  clean'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
