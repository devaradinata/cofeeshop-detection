"""Ekstrak frame dari video CCTV untuk anotasi.
Output: data/raw/<nama_video>/images/<nama_video>_<frame>.jpg  (1 video = 1 "session").

  python scripts/extract_frames.py --videos data/videos --every-sec 2
"""
import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder berisi video")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--every-sec", type=float, default=2.0)
    a = ap.parse_args()

    for vp in sorted(Path(a.videos).glob("*")):
        if vp.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(int(fps * a.every_sec), 1)
        out = Path(a.out) / vp.stem / "images"
        out.mkdir(parents=True, exist_ok=True)
        i = n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                cv2.imwrite(str(out / f"{vp.stem}_{i:07d}.jpg"), frame)
                n += 1
            i += 1
        cap.release()
        print(f"{vp.name}: {n} frame")


if __name__ == "__main__":
    main()
