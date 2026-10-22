# Product Requirement Document (PRD)

## Project Title

**Computer Vision-Based Coffee Shop Seat Availability, Occupancy, & Duration Monitoring System**

## 1. Project Overview & Objectives

This system is designed to process CCTV/IP camera video feeds from a coffee shop environment to automatically:

1. **Detect Seat Availability:** Determine whether seats/tables are `Available`, `Occupied`, or `Reserved by Belongings` (items left behind without people present).

2. **Monitor Occupancy & Duration:** Track customer presence, identify unique visitors across frames, and calculate real-time seating duration.

3. **Assess Seating Pressure:** Compute capacity metrics, occupancy rates, and evaluate seating pressure to provide actionable operational insights for coffee shop management.

## 2. Technical Stack

* **Language:** Python 3.10+

* **Object Detection Model:** Ultralytics YOLOv8 / YOLOv11

* **Multi-Object Tracking (MOT):** ByteTrack / BoT-SORT

* **Spatial & Geometry Analysis:** OpenCV, Shapely (Polygon/ROI overlap logic)

* **Temporal Analysis:** Custom State Machine with Timestamps & Buffering

* **Dashboard / UI:** Streamlit (or Gradio / FastAPI + React)

## 3. Phased Roadmap & Workflows

To ensure structured development, implementation is divided into four distinct phases. **Each phase must be completed and tested before moving to the next.**

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│     PHASE 1     │ ──> │     PHASE 2     │ ──> │     PHASE 3      │ ──> │     PHASE 4      │
│ Dataset & Model │     │ Object Tracking │     │ Spatial-Temporal │     │ Metrics & UI/UX  │
└─────────────────┘     └─────────────────┘     └──────────────────┘     └──────────────────┘

```

### 🟢 PHASE 1: Dataset Preparation & Custom Object Detection Model

> **Focus:** Gather/annotate dataset and train/fine-tune the foundational YOLO object detection model.

#### Tasks & Technical Requirements:

1. **Target Object Classes:**

   * `person` (Customers)

   * `chair` / `seat` (Seating items)

   * `table` (Tables)

   * `belongings` / `items` (Laptops, backpacks/bags, books, bottles/cups)

2. **Dataset Configuration:**

   * Utilize COCO pre-trained weights for baseline initialization.

   * Collect & annotate custom coffee shop video frames (1,000–2,000 annotated images in standard YOLO format).

   * Structure dataset directory following Ultralytics format (`train/`, `val/`, `test/`). Ensure train/test splits are divided by video sessions rather than random consecutive frames to prevent data leakage.

3. **Training & Model Evaluation:**

   * Train YOLO models (e.g., YOLOv8s or YOLOv8m).

   * Evaluate detection performance using: **Precision, Recall, mAP@50, and mAP@50-95**.

#### Deliverables Phase 1:

* Model configuration file (`config/classes.yaml`).

* Training and evaluation scripts (`train.py`, `eval.py`).

* Best model weight file (`best.pt`).

* Basic inference script visualizing object bounding boxes.

### 🟡 PHASE 2: Multi-Object Tracking (MOT) & Person Identification

> **Focus:** Integrate multi-object tracking algorithms to maintain persistent customer IDs across video frames.

#### Tasks & Technical Requirements:

1. **MOT Integration:**

   * Integrate **ByteTrack** or **BoT-SORT** into the YOLO inference pipeline.

   * Target ID tracking specifically for the `person` class.

2. **Occlusion & ID Switch Management:**

   * Implement trajectory retention and low-confidence association logic to handle short-term occlusions.

3. **Tracking Metrics Evaluation:**

   * Measure tracking performance using **IDF1, HOTA, and ID Switches**.

#### Deliverables Phase 2:

* Tracking module (`src/tracker.py`) reading video input and outputting consistent `Person Unique IDs` across consecutive frames.

### 🟠 PHASE 3: Seat ROI Mapping, Spatial-Temporal Rules, & State Machine

> **Focus:** Connect object detections and trajectories to predefined seat/table regions to determine status and duration.

#### Tasks & Technical Requirements:

1. **Interactive/Configurable ROI Mapping:**

   * Build a module to define multi-point polygons (ROI) for each table/seat in the video frame (`config/roi_config.json`).

2. **Geometric Overlap & Spatial Analysis:**

   * Calculate intersection over area / centroid overlap between detected objects (`person`, `belongings`) and defined table/seat polygons.

3. **Table/Seat State Machine Implementation:**
   Implement seating status classification:

   * 🟢 **Available:** No `person` and no `belongings` detected within the table ROI.

   * 🔴 **Occupied:** `person` detected within the table ROI (regardless of belongings).

   * 🟡 **Reserved by Belongings:** No `person` detected, but `belongings` (laptop, bag, book) are present within the table ROI.

   * ⚪ **Unknown / Needs Review:** Occluded views or low-confidence detection states.

4. **Temporal Analysis & Buffer (Smoothing):**

   * Apply temporal smoothing / grace periods (e.g., require status consistency for N consecutive frames or allow a 2-minute buffer for temporary absences like restroom visits) to prevent flickering.

   * Track session timestamps and calculate total seating duration:
     

     $$
     D = t_{\text{end}} - t_{\text{start}}
     $$

   * Categorize seating duration states: **Normal**, **Approaching Limit** (e.g., >90% of threshold), and **Overtime**.

#### Deliverables Phase 3:

* Spatial overlap engine (`src/spatial.py`).

* State Machine engine (`src/state_machine.py`).

* Real-time metadata output per Table ID (Status, Active Person ID, Duration, Warning Level).

### 🔵 PHASE 4: Operational Metrics, Seating Pressure Assessment, & Dashboard UI

> **Focus:** Aggregate metrics across all seats/tables and present insights via a web-based dashboard.

#### Tasks & Technical Requirements:

1. **Operational Metrics Calculation:**

   * **Occupancy Rate (%):**
     

     $$
     \text{Occupancy Rate} = \left( \frac{\text{Occupied Seats}}{\text{Total Capacity}} \right) \times 100\%
     $$

   * **Seating Pressure Level (Rule-based Scoring):**

     * **High:** Occupancy $\ge 90\%$ and minimal available seats.

     * **Medium:** High occupancy with limited remaining seats.

     * **Low:** Sufficient capacity available.

2. **System Evaluation Metrics:**

   * Evaluate table status classification accuracy (Accuracy, F1-Score).

   * Calculate duration tracking error using **Mean Absolute Error (MAE)**:
     

     $$
     \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \vert{}D_i - \hat{D}_i\vert{}
     $$

   * Measure overall system processing latency and **FPS (Frames Per Second)**.

3. **Interactive Dashboard UI (Streamlit):**

   * Real-time video stream overlay with polygon boundary colors representing status.

   * Executive summary panels: Total Capacity, Occupied Count, Available Count, Reserved Count, Seating Pressure Indicator, and Overtime Table Alerts.

#### Deliverables Phase 4:

* Web Dashboard Application (`app.py`).

* System evaluation runner (`evaluate_system.py`).

## 4. Recommended Directory Structure

```
coffee-shop-occupancy/
│
├── config/
│   ├── classes.yaml           # YOLO class definitions
│   └── roi_config.json        # Polygon coordinates for tables/seats
│
├── dataset/                   # Custom annotated dataset
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── best_yolo.pt           # Trained weights file
│
├── src/
│   ├── __init__.py
│   ├── detector.py            # YOLO detector wrapper (Phase 1)
│   ├── tracker.py             # ByteTrack tracker wrapper (Phase 2)
│   ├── spatial.py             # ROI polygon overlap analysis (Phase 3)
│   ├── state_machine.py       # State transitions & duration timers (Phase 3)
│   └── metrics.py             # Occupancy, Seating Pressure, & MAE (Phase 4)
│
├── app.py                     # Streamlit Dashboard UI (Phase 4)
├── train.py                   # Model training script (Phase 1)
├── evaluate.py                # System accuracy & benchmark runner
└── requirements.txt

```