# Coffee Shop Occupancy — Phase 1 & 2

## Setup
    pip install -r requirements.txt

## Phase 1 — Dataset & Deteksi
1. `python scripts/extract_frames.py --videos data/videos --every-sec 2`
2. Anotasi frame di `data/raw/<sesi>/images` (CVAT / Roboflow / Label Studio) -> export YOLO ke `data/raw/<sesi>/labels`. Kelas & urutan: lihat `config/classes.yaml`.
3. `python scripts/split_by_session.py` -> `dataset/{train,val,test}` (dipisah per sesi video).
4. `python train.py --model yolov8s.pt --epochs 100` -> `models/best_yolo.pt`
5. `python eval.py --split test` -> Precision, Recall, mAP@50, mAP@50-95
6. `python infer.py --source data/videos/x.mp4 --show`

## Phase 2 — Tracking
    python -m src.tracker --source data/videos/x.mp4 --tracker bytetrack \
        --out runs/track/out.mp4 --mot-out runs/track/pred.txt
    python evaluate_tracking.py --gt data/gt/x.txt --pred runs/track/pred.txt
