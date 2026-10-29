"""Multi-object tracking untuk kelas `person` (Phase 2).

Detektor mendeteksi SEMUA kelas (person, chair, table, belongings) -- Phase 3 butuh semuanya --
tetapi hanya `person` yang dilacak dan diberi ID unik persisten via ByteTrack / BoT-SORT.

Penanganan oklusi / ID switch:
  * asosiasi dua tahap ByteTrack: deteksi confidence rendah (track_low_thresh) tetap dipakai
    untuk mengaitkan track yang tertutup sebagian;
  * `track_buffer` mempertahankan track yang hilang selama N frame sebelum dihapus;
  * BoT-SORT: kompensasi gerak kamera (GMC) + opsional Re-ID (with_reid: True).

Pemakaian CLI:
  python -m src.tracker --source data/video.mp4 --weights models/best_yolo.pt \
      --tracker bytetrack --out runs/track/out.mp4 --mot-out runs/track/pred.txt
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from src.detector import Detection, Detector, resolve_model_weights
from src.utils import ROOT

DEFAULT_CFG = {"bytetrack": ROOT / "config" / "bytetrack.yaml",
               "botsort": ROOT / "config" / "botsort.yaml"}


@dataclass
class TrackedPerson:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    conf: float


@dataclass
class FrameResult:
    frame_idx: int          # 0-based
    timestamp: float        # detik sejak awal video
    frame: np.ndarray
    detections: List[Detection]   # semua kelas (untuk Phase 3)
    persons: List[TrackedPerson]  # person + ID unik


class PersonTracker:
    def __init__(self, person_cls_id: int, tracker_type: str = "bytetrack",
                 cfg_path: Optional[str] = None, frame_rate: int = 30):
        self.person_cls_id = person_cls_id
        cfg_file = Path(cfg_path or DEFAULT_CFG[tracker_type])
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        cfg = IterableSimpleNamespace(**cfg_dict)
        cls = {"bytetrack": BYTETracker, "botsort": BOTSORT}[tracker_type]
        self.tracker = cls(args=cfg, frame_rate=frame_rate)

    def reset(self):
        self.tracker.reset()

    def update(self, boxes, frame: np.ndarray) -> List[TrackedPerson]:
        # Tetap dipanggil walau kosong supaya track yang hilang menua (buffer berjalan).
        person_boxes = boxes[boxes.cls == self.person_cls_id]
        tracks = self.tracker.update(person_boxes.cpu().numpy(), frame)
        # kolom: x1, y1, x2, y2, id, score, cls, idx
        return [TrackedPerson(int(t[4]), tuple(map(float, t[:4])), float(t[5])) for t in tracks]


def track_video(source: str, weights: Optional[str] = None, tracker_type: str = "bytetrack",
                conf: float = 0.25, imgsz: int = 640, device: Optional[str] = None,
                tracker_cfg: Optional[str] = None) -> Iterator[FrameResult]:
    """Generator utama: dipakai CLI ini dan nanti oleh pipeline Phase 3/4."""
    detector = Detector(weights, conf=conf, imgsz=imgsz, device=device)
    person_id = detector.class_id("person")
    if person_id is None:
        raise ValueError("Kelas 'person' tidak ada di model.")
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise IOError(f"Tidak bisa membuka sumber video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = PersonTracker(person_id, tracker_type, tracker_cfg, int(round(fps)))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets, boxes = detector.detect(frame)
            persons = tracker.update(boxes, frame)
            yield FrameResult(idx, idx / fps, frame, dets, persons)
            idx += 1
    finally:
        cap.release()


def _color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.RandomState(track_id * 9973 % (2**31))
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw(res: FrameResult, person_name: str = "person") -> np.ndarray:
    img = res.frame.copy()
    for d in res.detections:
        if d.cls_name == person_name:
            continue
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
        cv2.putText(img, d.cls_name, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (160, 160, 160), 1)
    for p in res.persons:
        x1, y1, x2, y2 = map(int, p.xyxy)
        c = _color(p.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"ID {p.track_id}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img


def main():
    ap = argparse.ArgumentParser(description="Person tracking (ByteTrack / BoT-SORT)")
    ap.add_argument("--source", required=True, help="path video / index webcam / URL RTSP")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    ap.add_argument("--tracker-cfg", default=None, help="override YAML tracker")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="simpan video hasil (mp4)")
    ap.add_argument("--mot-out", default=None, help="simpan hasil format MOTChallenge (.txt)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    writer, mot = None, None
    if a.mot_out:
        Path(a.mot_out).parent.mkdir(parents=True, exist_ok=True)
        mot = open(a.mot_out, "w")
    try:
        for res in track_video(a.source, a.weights, a.tracker, a.conf, a.imgsz,
                               a.device, a.tracker_cfg):
            vis = draw(res)
            if a.out and writer is None:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            if writer:
                writer.write(vis)
            if mot:
                for p in res.persons:  # frame, id, x, y, w, h, conf, -1, -1, -1 (frame 1-based)
                    x1, y1, x2, y2 = p.xyxy
                    mot.write(f"{res.frame_idx + 1},{p.track_id},{x1:.2f},{y1:.2f},"
                              f"{x2 - x1:.2f},{y2 - y1:.2f},{p.conf:.3f},-1,-1,-1\n")
            if a.show:
                cv2.imshow("tracking", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer:
            writer.release()
        if mot:
            mot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
