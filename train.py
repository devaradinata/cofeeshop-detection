"""Fine-tuning YOLO untuk coffee shop (Phase 1).

Contoh:
  python train.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
Bobot COCO dipakai sebagai inisialisasi; head otomatis disesuaikan ke 4 kelas.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from src.utils import ROOT, resolved_data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8s.pt / yolov8m.pt / yolo11s.pt ...")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0, 0,1, cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="coffee_yolo")
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=resolved_data_yaml(), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, workers=a.workers, patience=a.patience,
        project=str(ROOT / "runs" / "train"), name=a.name, exist_ok=True,
        # augmentasi yang masuk akal untuk CCTV kafe
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, close_mosaic=10, seed=42,
    )

    best = ROOT / "runs" / "train" / a.name / "weights" / "best.pt"
    dst = ROOT / "models" / "best_yolo.pt"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy(best, dst)
    print(f"Bobot terbaik disalin ke {dst}")


if __name__ == "__main__":
    main()
