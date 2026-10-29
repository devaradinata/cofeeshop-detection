"""Run Coffee Shop Occupancy Pipeline on Video and Save Annotated Output."""
import argparse
import time
from pathlib import Path
import cv2

from src.pipeline import CoffeeShopPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Process video with Coffee Shop Occupancy Pipeline.")
    parser.add_argument("--source", default="data/videos/kopisop.mp4", help="Video source path")
    parser.add_argument("--weights", default="yolov8s.pt", help="YOLO model weights")
    parser.add_argument("--out", default="runs/annotated_kopisop.mp4", help="Output annotated video path")
    parser.add_argument("--max-frames", type=int, default=120, help="Max frames to process (None for full video)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--limit", type=float, default=60.0, help="Duration limit in seconds")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path.resolve()}")

    print(f"==================================================")
    print(f"☕ COFFEE SHOP OCCUPANCY SYSTEM: PROCESSING VIDEO")
    print(f"==================================================")
    print(f"Source Video : {source_path}")
    print(f"Model Weights: {args.weights}")
    print(f"Output Video : {args.out}")
    print(f"Duration Limit: {args.limit}s")
    print(f"Max Frames   : {args.max_frames if args.max_frames else 'Full Video'}")
    print(f"--------------------------------------------------")

    cfg = PipelineConfig(
        video_source=str(source_path),
        model_weights=args.weights,
        roi_config_path="config/roi_config.json",
        conf_threshold=args.conf,
        duration_limit_seconds=args.limit
    )

    pipeline = CoffeeShopPipeline(cfg)
    cap = cv2.VideoCapture(str(source_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    t_start = time.perf_counter()
    processed_count = 0

    print("Processing frames...")
    for pf in pipeline.process_video(str(source_path), max_frames=args.max_frames):
        writer.write(pf.annotated_frame)
        processed_count += 1
        m = pf.metrics

        if processed_count % 12 == 0 or processed_count == 1:
            print(
                f"[Frame {pf.frame_idx:03d} | t={pf.timestamp:05.2f}s] "
                f"Seats: {m.occupied_seats}/{m.total_seats} ({m.seat_occupancy_rate:4.1f}%) | "
                f"Tables: {m.occupied_tables}/{m.total_tables} Occ | "
                f"Pressure: {m.seating_pressure:6s} | "
                f"Tracked Persons: {len(pf.tracked_persons)}"
            )

    writer.release()
    total_time = time.perf_counter() - t_start
    avg_fps = processed_count / total_time if total_time > 0 else 0.0

    print(f"--------------------------------------------------")
    print(f"✅ Processing Complete!")
    print(f"Processed Frames : {processed_count}")
    print(f"Total Time       : {total_time:.2f} seconds")
    print(f"Average Speed    : {avg_fps:.2f} FPS")
    print(f"Annotated Video  : {out_path.resolve()}")
    print(f"==================================================")


if __name__ == "__main__":
    main()

