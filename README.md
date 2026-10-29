# Coffee Shop Seat Availability, Occupancy & Seating Pressure Monitoring System

A Computer Vision system for coffee shop environments processing CCTV/IP camera video feeds to:
1. **Detect Seat & Table Availability:** Classify regions as `AVAILABLE`, `OCCUPIED`, `RESERVED_BY_BELONGINGS`, or `UNKNOWN`.
2. **Monitor Occupancy & Seating Duration:** Track customers across frames using ByteTrack/BoT-SORT, distinguish person visibility duration from actual seating duration, and flag long-stay sessions (`NORMAL`, `APPROACHING_LIMIT`, `OVERTIME`).
3. **Assess Seating Pressure:** Compute capacity metrics, occupancy rates, and evaluate real-time seating pressure (`LOW`, `MEDIUM`, `HIGH`) with explainable rules.

---

## 1. System Architecture & Processing Pipeline

```
CCTV / Video (`kopisop.mp4`)
    │
    ▼
Frame Preprocessing
    │
    ▼
YOLOv8s Object Detection (Person, Chair, Table, Belongings: Laptop, Backpack, Handbag, Cup, Book)
    │
    ▼
Multi-Object Tracking (ByteTrack / BoT-SORT for Person Class)
    │
    ▼
Seat and Table ROI Mapping (`config/roi_config.json`)
    │
    ▼
Spatial Association (IoA, Centroid Containment, Occlusion Handling)
    │
    ▼
Temporal State Machine & Grace Period Buffering (`src/state_machine.py`)
    │
    ▼
Occupancy Metrics & Seating Pressure Assessment (`src/metrics.py`)
    │
    ▼
Interactive Dashboard (`app.py` via Streamlit)
```

---

## 2. Environment Setup (Windows PowerShell)

Ensure Python 3.10+ is installed on Windows:

```powershell
# 1. Clone or navigate to the repository
cd "d:\0_KULIYAH\TI SEMESTER 3\KECERDASAN TIRUAN\coffee-shop-occupancy"

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 3. Running the Streamlit Dashboard

To launch the web-based interactive monitoring dashboard:

```powershell
streamlit run app.py
```

### Dashboard Features:
* **Prerecorded Video Feed Playback:** Process `data/videos/kopisop.mp4` with colored polygon overlays (Green = Available, Red = Occupied, Amber = Reserved by Belongings, Gray = Unknown).
* **Live KPI Cards:** Real-time Seat Occupancy Rate (%), Seating Pressure Gauge, Available vs Occupied Seats/Tables, Reserved Count, and Overtime Count.
* **Table-by-Table Status Table:** Real-time table status, assigned seat IDs, active tracked customer IDs, seating duration, belongings duration, and duration warning level.
* **Explainable Seating Pressure Box:** Shows the exact rule triggered (e.g. *Occupancy rate in moderate range (50-80%) | 1 table overtime*).
* **Configurable Sidebar:** Change video source, YOLO model weights, detection confidence, duration limit thresholds, and grace period buffers.

---

## 4. Running System Evaluation & Experiments

The system supports three experimental configurations as outlined in the PRD:

### Run All Experiments (A, B, and C):
```powershell
python evaluate_system.py --source data/videos/kopisop.mp4 --weights yolov8s.pt --max-frames 60
```

### Run Experiment A (Detection Baseline):
Evaluates raw YOLO object detection on frames without tracking or temporal state machines.
```powershell
python evaluate_system.py --source data/videos/kopisop.mp4 --exp a --max-frames 60
```

### Run Experiment B (Detection & Tracking):
Evaluates ByteTrack person tracking stability, identity retention, and track durations.
```powershell
python evaluate_system.py --source data/videos/kopisop.mp4 --exp b --max-frames 60
```

### Run Experiment C (Full Context-Aware Seating Pipeline):
Evaluates the complete end-to-end proposed system including spatial ROI mapping, belongings association, temporal confirmation, and seating pressure scoring.
```powershell
python evaluate_system.py --source data/videos/kopisop.mp4 --exp c --max-frames 60
```

Results are printed to console and saved as JSON to `runs/eval_system_results.json`.

---

## 5. Running the Comprehensive Test Suite

Run the full automated test suite covering all 14 required capabilities:

```powershell
python tests/test_all_modules.py
```

Tests include:
* Video input validation
* Model loading & COCO category mapping
* Multi-object tracking
* Seat and table ROI polygon configuration loading/saving
* Spatial association (IoA, centroid containment, foot-point)
* Belongings-based reservation logic
* Availability state transitions (`AVAILABLE` ↔ `OCCUPIED` ↔ `RESERVED_BY_BELONGINGS` ↔ `UNKNOWN`)
* Duration calculations and warning limits (`NORMAL`, `APPROACHING_LIMIT`, `OVERTIME`)
* Customer grace periods during temporary absence
* Occupancy rate metrics
* Rule-based seating pressure scoring (`LOW`, `MEDIUM`, `HIGH`)
* End-to-end pipeline execution on `data/videos/kopisop.mp4`

---

## 6. Project Structure

```
coffee-shop-occupancy/
├── config/
│   ├── classes.yaml            # YOLO class definitions
│   ├── bytetrack.yaml          # ByteTrack configuration
│   ├── botsort.yaml            # BoT-SORT configuration
│   └── roi_config.json         # Table & seat polygon coordinates
├── data/
│   ├── raw/                    # Raw frame storage
│   └── videos/
│       ├── kopisop.mp4         # Primary coffee shop video feed
│       └── coffee_shop.mp4     # Secondary test video
├── dataset/                    # Custom annotated dataset splits (train/val/test)
├── models/                     # Custom trained model weights
├── runs/                       # Output artifacts, evaluations & benchmark JSONs
├── scripts/
│   ├── extract_frames.py       # Frame extraction script
│   └── split_by_session.py     # Session-based dataset split script
├── src/
│   ├── __init__.py
│   ├── detector.py             # YOLO detector wrapper & category mapping
│   ├── tracker.py              # ByteTrack / BoT-SORT multi-object tracker
│   ├── spatial.py              # Table & seat ROI geometric association (IoA/Centroid)
│   ├── state_machine.py        # Temporal state machine, grace periods & duration timers
│   ├── metrics.py              # Capacity metrics, occupancy rates & seating pressure
│   ├── pipeline.py             # Full end-to-end processing pipeline orchestrator
│   └── utils.py                # File & path utilities
├── tests/
│   ├── __init__.py
│   └── test_all_modules.py     # 18-test comprehensive test suite
├── app.py                      # Streamlit interactive dashboard UI
├── evaluate_system.py          # Benchmark runner for Experiments A, B, and C
├── yolov8s.pt                  # Pretrained YOLOv8s model weights
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## 7. Dataset Status & Ground Truth Note

* The folders `dataset/train`, `dataset/val`, and `dataset/test` currently do not contain annotated image frames.
* The system utilizes the verified pre-trained `yolov8s.pt` model baseline with COCO class mapping for general object detection (people, chairs, tables, laptops, backpacks, handbags, cups, books).
* If fine-tuning on custom coffee shop data is desired, frames can be extracted using `scripts/extract_frames.py`, annotated in YOLO format, placed in `data/raw/<session>`, and split using `scripts/split_by_session.py`.
