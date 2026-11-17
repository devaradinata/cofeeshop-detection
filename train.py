"""Fine-tuning YOLO untuk coffee shop (Phase 1).

Mendukung otomatis GPU NVIDIA CUDA (RTX 5050 / RTX Series) atau CPU.

Contoh:
  python train.py --model yolov8s.pt --epochs 25 --imgsz 640 --batch 16
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

# Hindari error permission matplotlib dan kestabilan CPU threading pada Windows
temp_dir = Path(tempfile.gettempdir()) / "mpl_temp"
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(temp_dir)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import torch
from ultralytics import YOLO

from src.utils import ROOT, resolved_data_yaml


def main():
    cuda_available = torch.cuda.is_available()
    default_device = "0" if cuda_available else "cpu"
    default_imgsz = 640 if cuda_available else 416
    default_batch = 16 if cuda_available else 4

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8s.pt / yolov8m.pt / yolo11s.pt ...")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=default_imgsz)
    ap.add_argument("--batch", type=int, default=default_batch)
    ap.add_argument("--device", default=default_device, help="0, cpu, cuda")
    ap.add_argument("--workers", type=int, default=2 if cuda_available else 0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--name", default="coffee_yolo")
    a = ap.parse_args()

    project_dir = (ROOT / "runs" / "train").resolve()
    
    device_info = f"GPU [{torch.cuda.get_device_name(0)}]" if cuda_available and a.device != "cpu" else "CPU"

    print(f"============================================================")
    print(f"Memulai Fine-Tuning YOLO: {a.model}")
    print(f"Akselerasi Komputasi: {device_info} (device={a.device})")
    print(f"Target Kelas: 0: person, 1: chair, 2: table, 3: belongings")
    print(f"Epochs: {a.epochs} | ImgSz: {a.imgsz} | Batch: {a.batch}")
    print(f"Data configuration: {resolved_data_yaml()}")
    print(f"============================================================")
    
    model = YOLO(a.model)
    model.train(
        data=resolved_data_yaml(),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        lr0=a.lr0,
        device=a.device,
        workers=a.workers,
        patience=a.patience,
        project=str(project_dir),
        name=a.name,
        exist_ok=True,
        # Augmentasi untuk variasi pencahayaan dan posisi objek meja CCTV
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=min(3, max(1, a.epochs // 4)),
        seed=42,
    )

    best = project_dir / a.name / "weights" / "best.pt"
    if not best.exists():
        best = project_dir / a.name / "weights" / "last.pt"

    dst = (ROOT / "models" / "best_yolo.pt").resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, dst)
        print(f"\n============================================================")
        print(f"SUCCESS: Bobot terbaik berhasil disimpan ke {dst}")
        print(f"============================================================\n")
    else:
        print(f"WARNING: File {best} tidak ditemukan.")


if __name__ == "__main__":
    main()
