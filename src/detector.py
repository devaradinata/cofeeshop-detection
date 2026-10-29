"""Wrapper detektor YOLO (Phase 1)."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Set

import numpy as np
from ultralytics import YOLO

from src.utils import ROOT

BELONGINGS_COCO: Set[str] = {
    "laptop", "backpack", "handbag", "suitcase", "book", "bottle", "cup", "cell phone"
}
SEAT_COCO: Set[str] = {"chair", "couch", "bench"}
TABLE_COCO: Set[str] = {"dining table", "table"}


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str
    category: str = "other"  # 'person', 'seat', 'table', 'belongings', 'other'


def resolve_model_weights(weights: Optional[str] = None) -> str:
    """Resolve model weights with graceful fallbacks."""
    candidates = []
    if weights:
        candidates.append(Path(weights))
        candidates.append(ROOT / weights)
    candidates.extend([
        ROOT / "models" / "best_yolo.pt",
        ROOT / "yolov8s.pt",
        Path("yolov8s.pt")
    ])
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    return weights or "yolov8s.pt"


class Detector:
    def __init__(self, weights: Optional[str] = None, conf: float = 0.25,
                 iou: float = 0.45, imgsz: int = 640, device: Optional[str] = None,
                 half: bool = False):
        resolved_weights = resolve_model_weights(weights)
        self.weights_path = resolved_weights
        self.model = YOLO(resolved_weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device, self.half = device, half
        self.names = self.model.names  # {id: name}

    def class_id(self, name: str) -> Optional[int]:
        for k, v in self.names.items():
            if v == name:
                return int(k)
        return None

    def get_category(self, class_name: str) -> str:
        name_lower = class_name.lower()
        if name_lower == "person":
            return "person"
        elif name_lower in SEAT_COCO or "chair" in name_lower or "seat" in name_lower:
            return "seat"
        elif name_lower in TABLE_COCO or "table" in name_lower:
            return "table"
        elif name_lower in BELONGINGS_COCO or "belonging" in name_lower:
            return "belongings"
        return "other"

    def detect(self, frame: np.ndarray):
        """Return (list[Detection], ultralytics Boxes). Boxes dipakai langsung oleh tracker."""
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, half=self.half, verbose=False)[0]
        dets: List[Detection] = []
        for b in res.boxes:
            cid = int(b.cls.item())
            cname = self.names[cid]
            cat = self.get_category(cname)
            dets.append(Detection(
                xyxy=tuple(b.xyxy[0].tolist()),
                conf=float(b.conf.item()),
                cls_id=cid,
                cls_name=cname,
                category=cat
            ))
        return dets, res.boxes
