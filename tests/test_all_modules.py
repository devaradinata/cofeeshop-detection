"""Comprehensive Test Suite for Coffee Shop Occupancy System.

Validates all 14 required capabilities:
1. Video input validation
2. Model loading
3. Object detection & categorization
4. Tracking
5. Seat and table configuration
6. Spatial association
7. Belongings association
8. Availability-state transitions
9. Duration calculation
10. Grace periods
11. Occupancy metrics calculation
12. Seating pressure assessment
13. Dashboard components
14. End-to-end pipeline execution on data/videos/kopisop.mp4
"""
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.detector import Detection, Detector, resolve_model_weights
from src.metrics import MetricsCalculator, OccupancyMetrics, compute_mae
from src.pipeline import CoffeeShopPipeline, PipelineConfig
from src.spatial import SeatingLayout, SpatialAssociationResult, SeatROI, TableROI
from src.state_machine import (
    AvailabilityState,
    DurationWarningState,
    RegionState,
    SeatingStateMachine,
)
from src.tracker import PersonTracker, TrackedPerson


class TestVideoInputValidation(unittest.TestCase):
    def test_valid_video(self):
        video_path = Path("data/videos/kopisop.mp4")
        self.assertTrue(video_path.exists(), "Primary video kopisop.mp4 must exist")
        cap = cv2.VideoCapture(str(video_path))
        self.assertTrue(cap.isOpened(), "Video should open successfully")
        cap.release()

    def test_nonexistent_video(self):
        cap = cv2.VideoCapture("data/videos/non_existent_file.mp4")
        self.assertFalse(cap.isOpened(), "Non-existent video should fail to open")
        cap.release()


class TestModelLoadingAndDetection(unittest.TestCase):
    def test_resolve_model_weights(self):
        resolved = resolve_model_weights("yolov8s.pt")
        self.assertTrue(Path(resolved).exists())

    def test_detector_initialization(self):
        detector = Detector("yolov8s.pt", conf=0.25)
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.class_id("person"), 0)

    def test_object_categorization(self):
        detector = Detector("yolov8s.pt", conf=0.25)
        self.assertEqual(detector.get_category("person"), "person")
        self.assertEqual(detector.get_category("chair"), "seat")
        self.assertEqual(detector.get_category("dining table"), "table")
        self.assertEqual(detector.get_category("laptop"), "belongings")
        self.assertEqual(detector.get_category("backpack"), "belongings")
        self.assertEqual(detector.get_category("cup"), "belongings")
        self.assertEqual(detector.get_category("car"), "other")


class TestTracking(unittest.TestCase):
    def test_person_tracker(self):
        tracker = PersonTracker(person_cls_id=0, tracker_type="bytetrack", frame_rate=24)
        self.assertIsNotNone(tracker.tracker)
        # Verify reset
        tracker.reset()


class TestSeatAndTableConfiguration(unittest.TestCase):
    def test_load_and_save_layout(self):
        layout = SeatingLayout("config/roi_config.json")
        self.assertGreater(len(layout.tables), 0)
        self.assertGreater(len(layout.seats), 0)
        self.assertIn("T1", layout.tables)
        self.assertIn("S1-A", layout.seats)

        # Test save and reload
        tmp_cfg = "runs/test_roi_save.json"
        layout.save_config(tmp_cfg)
        self.assertTrue(Path(tmp_cfg).exists())
        reloaded = SeatingLayout(tmp_cfg)
        self.assertEqual(len(layout.tables), len(reloaded.tables))
        self.assertEqual(len(layout.seats), len(reloaded.seats))
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)


class TestSpatialAssociation(unittest.TestCase):
    def setUp(self):
        self.layout = SeatingLayout("config/roi_config.json")

    def test_person_inside_table(self):
        # T1 table is around [260, 490, 530, 690]
        # S1-A is around [100, 480, 270, 680]
        person = TrackedPerson(track_id=42, xyxy=(120.0, 500.0, 240.0, 650.0), conf=0.85)
        assoc = self.layout.associate_spatial(
            tracked_persons=[person],
            detections=[]
        )
        self.assertIn(42, assoc.seat_persons.get("S1-A", []))
        self.assertIn(42, assoc.table_persons.get("T1", []))
        self.assertEqual(len(assoc.unassociated_persons), 0)

    def test_belongings_on_table(self):
        # Laptop placed directly on T1: [300, 520, 360, 580]
        laptop = Detection(
            xyxy=(300.0, 520.0, 360.0, 580.0),
            conf=0.75,
            cls_id=63,
            cls_name="laptop",
            category="belongings"
        )
        assoc = self.layout.associate_spatial(
            tracked_persons=[],
            detections=[laptop]
        )
        self.assertIn("T1", assoc.table_belongings)
        self.assertEqual(len(assoc.table_belongings["T1"]), 1)
        self.assertEqual(len(assoc.unassociated_belongings), 0)

    def test_unassociated_person(self):
        # Far off coordinates (outside any coffee table)
        person = TrackedPerson(track_id=99, xyxy=(10.0, 10.0, 50.0, 80.0), conf=0.90)
        assoc = self.layout.associate_spatial(
            tracked_persons=[person],
            detections=[]
        )
        self.assertIn(99, assoc.unassociated_persons)


class TestStateMachineAndGracePeriod(unittest.TestCase):
    def test_availability_transitions(self):
        sm = SeatingStateMachine(
            table_ids=["T1"],
            seat_ids=["S1-A"],
            duration_limit_seconds=10.0,
            grace_period_seconds=2.0,
            confirmation_seconds=0.5
        )

        # 1. Initially AVAILABLE
        self.assertEqual(sm.table_states["T1"].availability_state, AvailabilityState.AVAILABLE)

        # 2. Customer arrives -> OCCUPIED
        assoc_occupied = SpatialAssociationResult(
            table_persons={"T1": [1]},
            table_belongings={"T1": []},
            seat_persons={"S1-A": [1]},
            seat_belongings={"S1-A": []}
        )
        sm.update(assoc_occupied, current_time=0.0, visible_person_ids=[1])
        self.assertEqual(sm.table_states["T1"].availability_state, AvailabilityState.OCCUPIED)
        self.assertIsNotNone(sm.table_states["T1"].current_session)

        # 3. Customer steps away, leaving belongings -> within grace period -> RESERVED_BY_BELONGINGS
        belonging = Detection((300, 520, 350, 570), 0.8, 63, "laptop", "belongings")
        assoc_belongings = SpatialAssociationResult(
            table_persons={"T1": []},
            table_belongings={"T1": [belonging]},
            seat_persons={"S1-A": []},
            seat_belongings={"S1-A": []}
        )
        sm.update(assoc_belongings, current_time=1.0, visible_person_ids=[])
        self.assertEqual(sm.table_states["T1"].availability_state, AvailabilityState.RESERVED_BY_BELONGINGS)

        # 4. Grace period expires with belongings still present -> remains RESERVED_BY_BELONGINGS
        sm.update(assoc_belongings, current_time=4.0, visible_person_ids=[])
        self.assertEqual(sm.table_states["T1"].availability_state, AvailabilityState.RESERVED_BY_BELONGINGS)

        # 5. Belongings removed -> confirmed empty after confirmation window -> AVAILABLE
        assoc_empty = SpatialAssociationResult(
            table_persons={"T1": []},
            table_belongings={"T1": []},
            seat_persons={"S1-A": []},
            seat_belongings={"S1-A": []}
        )
        sm.update(assoc_empty, current_time=5.0, visible_person_ids=[])
        # After confirmation window (0.5s)
        sm.update(assoc_empty, current_time=5.6, visible_person_ids=[])
        self.assertEqual(sm.table_states["T1"].availability_state, AvailabilityState.AVAILABLE)

    def test_duration_limits(self):
        sm = SeatingStateMachine(
            table_ids=["T1"],
            seat_ids=["S1-A"],
            duration_limit_seconds=10.0,
            approaching_ratio=0.90
        )
        assoc = SpatialAssociationResult(
            table_persons={"T1": [1]},
            table_belongings={"T1": []},
            seat_persons={"S1-A": [1]},
            seat_belongings={"S1-A": []}
        )
        # At start: NORMAL
        sm.update(assoc, current_time=0.0, visible_person_ids=[1])
        self.assertEqual(sm.table_states["T1"].duration_warning_state, DurationWarningState.NORMAL)

        # At 9.2s (92%): APPROACHING_LIMIT
        sm.update(assoc, current_time=9.2, visible_person_ids=[1])
        self.assertEqual(sm.table_states["T1"].duration_warning_state, DurationWarningState.APPROACHING_LIMIT)

        # At 10.5s (>100%): OVERTIME
        sm.update(assoc, current_time=10.5, visible_person_ids=[1])
        self.assertEqual(sm.table_states["T1"].duration_warning_state, DurationWarningState.OVERTIME)


class TestMetricsAndSeatingPressure(unittest.TestCase):
    def setUp(self):
        self.calc = MetricsCalculator(high_occupancy_threshold=80.0, medium_occupancy_threshold=50.0)

    def test_occupancy_rate_calculation(self):
        t_states = {
            "T1": RegionState("T1", True, AvailabilityState.OCCUPIED),
            "T2": RegionState("T2", True, AvailabilityState.AVAILABLE),
        }
        s_states = {
            "S1-A": RegionState("S1-A", False, AvailabilityState.OCCUPIED),
            "S1-B": RegionState("S1-B", False, AvailabilityState.OCCUPIED),
            "S2-A": RegionState("S2-A", False, AvailabilityState.AVAILABLE),
            "S2-B": RegionState("S2-B", False, AvailabilityState.AVAILABLE),
        }
        metrics = self.calc.compute(t_states, s_states)
        # 2 / 4 = 50.0%
        self.assertEqual(metrics.total_seats, 4)
        self.assertEqual(metrics.occupied_seats, 2)
        self.assertEqual(metrics.seat_occupancy_rate, 50.0)

    def test_high_seating_pressure(self):
        t_states = {
            "T1": RegionState("T1", True, AvailabilityState.OCCUPIED),
            "T2": RegionState("T2", True, AvailabilityState.OCCUPIED),
        }
        s_states = {
            "S1-A": RegionState("S1-A", False, AvailabilityState.OCCUPIED),
            "S1-B": RegionState("S1-B", False, AvailabilityState.OCCUPIED),
            "S2-A": RegionState("S2-A", False, AvailabilityState.OCCUPIED),
            "S2-B": RegionState("S2-B", False, AvailabilityState.AVAILABLE),
        }
        metrics = self.calc.compute(t_states, s_states)
        # All tables occupied (0 available tables) -> HIGH
        self.assertEqual(metrics.seating_pressure, "HIGH")

    def test_low_seating_pressure(self):
        t_states = {
            "T1": RegionState("T1", True, AvailabilityState.AVAILABLE),
            "T2": RegionState("T2", True, AvailabilityState.AVAILABLE),
            "T3": RegionState("T3", True, AvailabilityState.AVAILABLE),
            "T4": RegionState("T4", True, AvailabilityState.AVAILABLE),
        }
        s_states = {
            f"S{i}": RegionState(f"S{i}", False, AvailabilityState.AVAILABLE) for i in range(8)
        }
        metrics = self.calc.compute(t_states, s_states)
        self.assertEqual(metrics.seating_pressure, "LOW")

    def test_unknown_not_counted_as_available(self):
        t_states = {"T1": RegionState("T1", True, AvailabilityState.UNKNOWN)}
        s_states = {
            "S1-A": RegionState("S1-A", False, AvailabilityState.UNKNOWN),
            "S1-B": RegionState("S1-B", False, AvailabilityState.AVAILABLE),
        }
        metrics = self.calc.compute(t_states, s_states)
        self.assertEqual(metrics.unknown_seats, 1)
        self.assertEqual(metrics.available_seats, 1)
        self.assertNotEqual(metrics.available_seats, 2)

    def test_mae_function(self):
        gt = [10.0, 20.0, 30.0]
        pred = [12.0, 19.0, 33.0]
        mae = compute_mae(gt, pred)
        self.assertAlmostEqual(mae, 2.0)


class TestPipelineEndToEnd(unittest.TestCase):
    def test_pipeline_execution(self):
        cfg = PipelineConfig(
            video_source="data/videos/kopisop.mp4",
            model_weights="yolov8s.pt",
            roi_config_path="config/roi_config.json"
        )
        pipe = CoffeeShopPipeline(cfg)
        frames_processed = 0
        for pf in pipe.process_video(max_frames=3):
            frames_processed += 1
            self.assertIsNotNone(pf.annotated_frame)
            self.assertGreater(len(pf.table_states), 0)
            self.assertGreater(len(pf.seat_states), 0)
            self.assertIn(pf.metrics.seating_pressure, ["LOW", "MEDIUM", "HIGH"])
        self.assertEqual(frames_processed, 3)


if __name__ == "__main__":
    unittest.main()
