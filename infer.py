"""Inferensi dasar: visualisasi bounding box semua kelas (Phase 1).

  python infer.py --source data/video.mp4 --weights models/best_yolo.pt --show
"""
import argparse
from pathlib import Path

import cv2

from src.detector import Detector

COLORS = {"person": (0, 200, 0), "chair": (255, 150, 0), "table": (0, 150, 255),
          "belongings": (0, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default="models/best_yolo.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    det = Detector(a.weights, conf=a.conf)
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets, _ = det.detect(frame)
        for d in dets:
            x1, y1, x2, y2 = map(int, d.xyxy)
            c = COLORS.get(d.cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{d.cls_name} {d.conf:.2f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        if a.out and writer is None:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer:
            writer.write(frame)
        if a.show:
            cv2.imshow("detections", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
