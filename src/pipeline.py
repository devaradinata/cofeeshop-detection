"""End-to-End Processing Pipeline Module (Phases 1-4).

Coordinates:
CCTV/IP Camera Video
→ Frame Preprocessing
→ YOLO Object Detection
→ Multi-Object Tracking
→ Seat and Table Mapping
→ Customer and Belongings Association
→ Spatial and Temporal Analysis
→ Seat Availability Classification
→ Occupancy and Duration Calculation
→ Seating Pressure Assessment
→ Dashboard / Visual Output
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from src.detector import Detection, Detector, resolve_model_weights
from src.metrics import MetricsCalculator, OccupancyMetrics
from src.spatial import SeatingLayout, SpatialAssociationResult
from src.state_machine import (
    AvailabilityState,
    DurationWarningState,
    RegionState,
    SeatingStateMachine,
)
from src.tracker import PersonTracker, TrackedPerson

STATUS_COLORS = {
    AvailabilityState.AVAILABLE: (40, 200, 40),          # Green
    AvailabilityState.OCCUPIED: (40, 40, 230),           # Red
    AvailabilityState.RESERVED_BY_BELONGINGS: (0, 165, 255), # Amber / Orange
    AvailabilityState.UNKNOWN: (130, 130, 130),          # Gray
}


@dataclass
class PipelineConfig:
    video_source: str = "data/videos/kopisop.mp4"
    model_weights: str = "yolov8s.pt"
    roi_config_path: str = "config/roi_config.json"
    tracker_type: str = "bytetrack"
    tracker_cfg: Optional[str] = None
    conf_threshold: float = 0.25
    imgsz: int = 640
    device: Optional[str] = None
    duration_limit_seconds: float = 60.0
    grace_period_seconds: float = 3.0
    confirmation_seconds: float = 0.5
    high_occupancy_threshold: float = 80.0
    medium_occupancy_threshold: float = 50.0


@dataclass
class ProcessedFrame:
    frame_idx: int
    timestamp: float
    raw_frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[Detection]
    tracked_persons: List[TrackedPerson]
    spatial_association: SpatialAssociationResult
    table_states: Dict[str, RegionState]
    seat_states: Dict[str, RegionState]
    metrics: OccupancyMetrics
    person_visibility_durations: Dict[int, float]


class CoffeeShopPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # 1. Detector
        self.detector = Detector(
            weights=self.config.model_weights,
            conf=self.config.conf_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device
        )
        person_cls_id = self.detector.class_id("person")
        if person_cls_id is None:
            raise ValueError("Target class 'person' was not found in the model classes.")

        # 2. Layout
        self.layout = SeatingLayout(self.config.roi_config_path)

        # 3. Tracker
        self.tracker = PersonTracker(
            person_cls_id=person_cls_id,
            tracker_type=self.config.tracker_type,
            cfg_path=self.config.tracker_cfg,
            frame_rate=24
        )

        # 4. State Machine
        self.state_machine = SeatingStateMachine(
            table_ids=list(self.layout.tables.keys()),
            seat_ids=list(self.layout.seats.keys()),
            duration_limit_seconds=self.config.duration_limit_seconds,
            grace_period_seconds=self.config.grace_period_seconds,
            confirmation_seconds=self.config.confirmation_seconds
        )

        # 5. Metrics Calculator
        self.metrics_calc = MetricsCalculator(
            high_occupancy_threshold=self.config.high_occupancy_threshold,
            medium_occupancy_threshold=self.config.medium_occupancy_threshold
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        draw_annotations: bool = True
    ) -> ProcessedFrame:
        # Preprocessing & Detection
        detections, raw_boxes = self.detector.detect(frame)

        # Multi-Object Tracking for person class
        tracked_persons = self.tracker.update(raw_boxes, frame)
        visible_person_ids = [p.track_id for p in tracked_persons]

        # Spatial Association (Tables & Seats)
        association = self.layout.associate_spatial(
            tracked_persons=tracked_persons,
            detections=detections
        )

        # State Machine update (Temporal confirmation & Duration calculation)
        self.state_machine.update(
            association=association,
            current_time=timestamp,
            visible_person_ids=visible_person_ids
        )

        # Operational Metrics & Seating Pressure
        metrics = self.metrics_calc.compute(
            table_states=self.state_machine.table_states,
            seat_states=self.state_machine.seat_states
        )

        person_vis_durations = self.state_machine.get_person_visibility_durations(timestamp)

        # Visualization
        if draw_annotations:
            annotated_frame = self.annotate_frame(
                frame=frame,
                detections=detections,
                tracked_persons=tracked_persons,
                metrics=metrics,
                timestamp=timestamp
            )
        else:
            annotated_frame = frame

        return ProcessedFrame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            raw_frame=frame,
            annotated_frame=annotated_frame,
            detections=detections,
            tracked_persons=tracked_persons,
            spatial_association=association,
            table_states=self.state_machine.table_states,
            seat_states=self.state_machine.seat_states,
            metrics=metrics,
            person_visibility_durations=person_vis_durations
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracked_persons: List[TrackedPerson],
        metrics: OccupancyMetrics,
        timestamp: float
    ) -> np.ndarray:
        vis = frame.copy()

        # 1. Draw Seat ROIs
        for s_id, seat in self.layout.seats.items():
            s_state = self.state_machine.seat_states.get(s_id)
            color = STATUS_COLORS.get(
                s_state.availability_state if s_state else AvailabilityState.AVAILABLE,
                (128, 128, 128)
            )
            pts = np.array(seat.polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=1)
            # Seat ID label
            cx = int(sum(p[0] for p in seat.polygon) / len(seat.polygon))
            cy = int(sum(p[1] for p in seat.polygon) / len(seat.polygon))
            cv2.putText(vis, s_id, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 2. Draw Table ROIs with semi-transparent overlay
        overlay = vis.copy()
        for t_id, table in self.layout.tables.items():
            t_state = self.state_machine.table_states.get(t_id)
            state = t_state.availability_state if t_state else AvailabilityState.AVAILABLE
            color = STATUS_COLORS.get(state, (128, 128, 128))

            pts = np.array(table.polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color=color)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            # Table status and duration badge
            min_x = min(p[0] for p in table.polygon)
            min_y = min(p[1] for p in table.polygon)
            badge_text = f"{t_id}: {state.value}"
            if t_state and t_state.occupancy_duration > 0:
                badge_text += f" ({t_state.occupancy_duration:.1f}s)"
            if t_state and t_state.duration_warning_state != DurationWarningState.NORMAL:
                badge_text += f" [{t_state.duration_warning_state.value}]"

            # Badge background
            cv2.rectangle(vis, (int(min_x), int(max(min_y - 20, 10))),
                          (int(min_x + len(badge_text) * 7.5), int(max(min_y, 25))),
                          (20, 20, 20), -1)
            cv2.putText(vis, badge_text, (int(min_x + 3), int(max(min_y - 5, 20))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Blend table fill
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # 3. Draw Tracked Persons (with ID and tracking duration)
        for p in tracked_persons:
            x1, y1, x2, y2 = map(int, p.xyxy)
            vis_dur = self.state_machine.person_last_seen.get(p.track_id, timestamp) - \
                      self.state_machine.person_first_seen.get(p.track_id, timestamp)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(vis, f"Person #{p.track_id} ({vis_dur:.1f}s)", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # 4. Draw Belongings
        for d in detections:
            if d.category == "belongings":
                x1, y1, x2, y2 = map(int, d.xyxy)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 1)
                cv2.putText(vis, f"{d.cls_name}", (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)

        # 5. Top HUD Display (Capacity, Occupancy, Seating Pressure)
        hud_h, hud_w = 42, 600
        hud_bg = vis[10:10+hud_h, 10:10+hud_w].copy()
        cv2.rectangle(vis, (10, 10), (10 + hud_w, 10 + hud_h), (25, 25, 25), -1)
        cv2.rectangle(vis, (10, 10), (10 + hud_w, 10 + hud_h), (100, 100, 100), 1)

        pressure_col = (0, 220, 0) if metrics.seating_pressure == "LOW" else \
                       (0, 165, 255) if metrics.seating_pressure == "MEDIUM" else (0, 0, 255)

        hud_text1 = f"Seats: {metrics.occupied_seats}/{metrics.total_seats} Occ | " \
                    f"Avail: {metrics.available_seats} | Belongings: {metrics.belongings_seats} | " \
                    f"Occ Rate: {metrics.seat_occupancy_rate:.1f}%"
        hud_text2 = f"Tables: {metrics.occupied_tables}/{metrics.total_tables} Occ | " \
                    f"Pressure: {metrics.seating_pressure} | Overtime: {metrics.overtime_tables}"

        cv2.putText(vis, hud_text1, (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
        cv2.putText(vis, hud_text2, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pressure_col, 1)

        return vis

    def process_video(
        self,
        video_source: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> Iterator[ProcessedFrame]:
        source = video_source or self.config.video_source
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        idx = 0
        try:
            while True:
                if max_frames and idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = idx / fps
                processed = self.process_frame(frame, idx, timestamp)
                yield processed
                idx += 1
        finally:
            cap.release()

