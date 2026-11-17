"""Script untuk mengekstrak dan menyiapkan dataset YOLO 4-kelas (Phase 1)
dari seluruh video di data/videos/ menggunakan akselerasi GPU CUDA.

Kelas:
  0: person
  1: chair
  2: table
  3: belongings (laptop, tas/ransel, buku, botol/gelas, mouse, keyboard)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import cv2
import torch
from ultralytics import YOLO

COCO_MAP = {
    "person": (0, 0.25),
    "chair": (1, 0.25), "couch": (1, 0.25), "bench": (1, 0.25),
    "dining table": (2, 0.20),
    "backpack": (3, 0.10), "handbag": (3, 0.10), "suitcase": (3, 0.10),
    "bottle": (3, 0.10), "cup": (3, 0.10), "bowl": (3, 0.10), "plate": (3, 0.10),
    "laptop": (3, 0.15), "mouse": (3, 0.10), "keyboard": (3, 0.10),
    "cell phone": (3, 0.10), "book": (3, 0.10)
}


def auto_annotate_image(model: YOLO, img_bgr, device="0"):
    h, w = img_bgr.shape[:2]
    res = model.predict(img_bgr, conf=0.10, imgsz=640, device=device, verbose=False)[0]
    lines = []
    for b in res.boxes:
        cname = model.names[int(b.cls.item())]
        conf = float(b.conf.item())
        if cname in COCO_MAP:
            target_cls, min_conf = COCO_MAP[cname]
            if conf >= min_conf:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                # Convert to normalized xywh
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                # Clamp to [0, 1]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    return lines


def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Memuat base model yolov8s.pt pada device={device}...")
    base_model = YOLO("yolov8s.pt")
    
    video_dir = ROOT / "data" / "videos"
    videos = sorted(list(video_dir.glob("*.mp4")))
    print(f"Ditemukan {len(videos)} video untuk diekstraksi:")
    for v in videos:
        print(f" - {v.name}")
    
    samples = []  # list of (img, label_lines, session_id, name)
    
    for vpath in videos:
        session_name = vpath.stem
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ambil ~8-12 frame per video secara proporsional
        num_target = 10
        step = max(total_f // num_target, 1)
        
        idx = 0
        extracted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0 and extracted < num_target:
                name = f"{session_name}_{idx:05d}"
                lbl_lines = auto_annotate_image(base_model, frame, device=device)
                if len(lbl_lines) > 0:
                    samples.append((frame, lbl_lines, session_name, name))
                    extracted += 1
            idx += 1
        cap.release()
        print(f"Ekstraksi dari {vpath.name}: {extracted} frame")

    # Tambahkan still images jika ada
    still_imgs = ["coffee_shop.jpg", "coffeshop.jpeg"]
    for s_name in still_imgs:
        sp = ROOT / s_name
        if sp.exists():
            img = cv2.imread(str(sp))
            if img is not None:
                lbl_lines = auto_annotate_image(base_model, img, device=device)
                samples.append((img, lbl_lines, "still_images", sp.stem))
                print(f"Menambahkan {s_name}: {len(lbl_lines)} anotasi")

    print(f"\nTotal seluruh gambar untuk dataset baru: {len(samples)}")
    
    # 2. Split dataset ke train (75%), val (15%), test (10%)
    random.seed(42)
    random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * 0.75)
    n_val = int(n_total * 0.15)
    
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:]
    }
    
    dataset_dir = ROOT / "dataset"
    for split, items in splits.items():
        img_out = dataset_dir / split / "images"
        lbl_out = dataset_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        # Bersihkan file lama di split jika perlu
        for f in img_out.glob("*.jpg"):
            f.unlink()
        for f in lbl_out.glob("*.txt"):
            f.unlink()
            
        for frame, lbl_lines, sess, name in items:
            cv2.imwrite(str(img_out / f"{name}.jpg"), frame)
            with open(lbl_out / f"{name}.txt", "w", encoding="utf-8") as f:
                f.writelines(lbl_lines)
                
        print(f"Split {split}: {len(items)} gambar & label disimpan di dataset/{split}/")

    print("Dataset baru berhasil disiapkan dan beranotasi lengkap!")


if __name__ == "__main__":
    main()
