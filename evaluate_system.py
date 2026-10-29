"""System Evaluation and Benchmark Runner (Phase 4).

Supports three research experiments:
  - Experiment A: Detection Baseline (Raw YOLO detections, single-frame inference)
  - Experiment B: Detection and Tracking (YOLO + ByteTrack/BoT-SORT, temporal persistence)
  - Experiment C: Full Context-Aware Seating Pipeline (Spatial ROI, Belongings association,
                  State Machine, Duration Limits, Seating Pressure)
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.detector import Detector, resolve_model_weights
from src.metrics import MetricsCalculator, OccupancyMetrics, compute_mae
from src.pipeline import CoffeeShopPipeline, PipelineConfig
from src.spatial import SeatingLayout
from src.tracker import PersonTracker


def run_experiment_a(video_path: str, weights: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
    """Experiment A: Detection Baseline (no tracking, one-frame decisions)."""
    print("\n--- Running Experiment A: Detection Baseline ---")
    detector = Detector(weights=weights, conf=0.25)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_latencies: List[float] = []
    category_counts: Dict[str, int] = {"person": 0, "seat": 0, "table": 0, "belongings": 0, "other": 0}
    total_frames = 0

    try:
        while True:
            if max_frames and total_frames >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            dets, _ = detector.detect(frame)
            latency = time.perf_counter() - t0
            frame_latencies.append(latency)

            for d in dets:
                category_counts[d.category] = category_counts.get(d.category, 0) + 1
            total_frames += 1
    finally:
        cap.release()

    avg_latency = float(np.mean(frame_latencies)) if frame_latencies else 0.0
    proc_fps = 1.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "experiment": "A - Detection Baseline",
        "processed_frames": total_frames,
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "processing_fps": round(proc_fps, 2),
        "total_detections": sum(category_counts.values()),
        "category_counts": category_counts,
        "notes": "Raw single-frame YOLOv8 detections without temporal tracking or spatial state machines."
    }


def run_experiment_b(video_path: str, weights: str, tracker_type: str = "bytetrack", max_frames: Optional[int] = None) -> Dict[str, Any]:
    """Experiment B: Detection and Tracking (person tracking, ID switches, trajectory persistence)."""
    print(f"\n--- Running Experiment B: Detection and Tracking ({tracker_type}) ---")
    detector = Detector(weights=weights, conf=0.25)
    person_cls_id = detector.class_id("person")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    tracker = PersonTracker(person_cls_id=person_cls_id, tracker_type=tracker_type, frame_rate=int(round(fps)))
    frame_latencies: List[float] = []
    total_frames = 0
    unique_track_ids = set()
    track_lengths: Dict[int, int] = {}

    try:
        while True:
            if max_frames and total_frames >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            dets, boxes = detector.detect(frame)
            tracked = tracker.update(boxes, frame)
            latency = time.perf_counter() - t0
            frame_latencies.append(latency)

            for p in tracked:
                unique_track_ids.add(p.track_id)
                track_lengths[p.track_id] = track_lengths.get(p.track_id, 0) + 1
            total_frames += 1
    finally:
        cap.release()

    avg_latency = float(np.mean(frame_latencies)) if frame_latencies else 0.0
    proc_fps = 1.0 / avg_latency if avg_latency > 0 else 0.0

    durations = [cnt / fps for cnt in track_lengths.values()]

    return {
        "experiment": f"B - Detection & Tracking ({tracker_type})",
        "processed_frames": total_frames,
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "processing_fps": round(proc_fps, 2),
        "unique_person_tracks": len(unique_track_ids),
        "avg_person_track_duration_sec": round(float(np.mean(durations)), 2) if durations else 0.0,
        "max_person_track_duration_sec": round(float(np.max(durations)), 2) if durations else 0.0,
        "notes": "ByteTrack/BoT-SORT maintains persistent person identities and measures visible person presence."
    }


def run_experiment_c(video_path: str, weights: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
    """Experiment C: Full Context-Aware Seating Pipeline."""
    print("\n--- Running Experiment C: Full Context-Aware Seating Pipeline ---")
    cfg = PipelineConfig(
        video_source=video_path,
        model_weights=weights,
        duration_limit_seconds=6.0,  # Illustrative 6s threshold for 10s video demonstration
        grace_period_seconds=2.0
    )
    pipeline = CoffeeShopPipeline(cfg)
    frame_latencies: List[float] = []
    total_frames = 0

    state_samples: Dict[str, int] = {"AVAILABLE": 0, "OCCUPIED": 0, "RESERVED_BY_BELONGINGS": 0, "UNKNOWN": 0}
    pressure_samples: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    last_metrics: Optional[OccupancyMetrics] = None

    for pf in pipeline.process_video(video_path, max_frames=max_frames):
        t0 = time.perf_counter()
        last_metrics = pf.metrics
        pressure_samples[pf.metrics.seating_pressure] = pressure_samples.get(pf.metrics.seating_pressure, 0) + 1
        for ts in pf.table_states.values():
            st_val = ts.availability_state.value
            state_samples[st_val] = state_samples.get(st_val, 0) + 1
        total_frames += 1

    final_table_summary = {}
    for tid, ts in pipeline.state_machine.table_states.items():
        final_table_summary[tid] = {
            "status": ts.availability_state.value,
            "occupancy_duration_sec": round(ts.occupancy_duration, 2),
            "warning": ts.duration_warning_state.value,
            "active_persons": ts.active_person_ids,
            "belongings_count": ts.active_belongings_count
        }

    return {
        "experiment": "C - Full Context-Aware Seating Pipeline",
        "processed_frames": total_frames,
        "final_occupancy_rate_pct": last_metrics.seat_occupancy_rate if last_metrics else 0.0,
        "final_occupied_seats": last_metrics.occupied_seats if last_metrics else 0,
        "final_total_seats": last_metrics.total_seats if last_metrics else 0,
        "final_seating_pressure": last_metrics.seating_pressure if last_metrics else "UNKNOWN",
        "seating_pressure_distribution": pressure_samples,
        "table_state_sample_distribution": state_samples,
        "tables_approaching_or_overtime": (last_metrics.approaching_limit_tables + last_metrics.overtime_tables) if last_metrics else 0,
        "tables_detail": final_table_summary,
        "notes": "Full end-to-end context-aware pipeline with spatial mapping, belongings association, and duration monitoring."
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Coffee Shop Occupancy System across Experiments A, B, and C.")
    parser.add_argument("--source", default="data/videos/kopisop.mp4", help="Video path")
    parser.add_argument("--weights", default="yolov8s.pt", help="YOLO weights path")
    parser.add_argument("--max-frames", type=int, default=60, help="Frames to process for evaluation (or None for all)")
    parser.add_argument("--exp", choices=["all", "a", "b", "c"], default="all")
    parser.add_argument("--out", default="runs/eval_system_results.json", help="Save benchmark results")
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_source": args.source,
        "model_weights": args.weights,
        "dataset_ground_truth_status": "Custom annotations in dataset/train/val/test are currently empty. Evaluation was conducted on video data with rule verification and latency benchmarking."
    }

    if args.exp in ["all", "a"]:
        results["exp_a"] = run_experiment_a(args.source, args.weights, args.max_frames)
    if args.exp in ["all", "b"]:
        results["exp_b"] = run_experiment_b(args.source, args.weights, "bytetrack", args.max_frames)
    if args.exp in ["all", "c"]:
        results["exp_c"] = run_experiment_c(args.source, args.weights, args.max_frames)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n================ SYSTEM EVALUATION REPORT ================")
    print(json.dumps(results, indent=2))
    print(f"\nResults successfully saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

