"""Wrapper detektor YOLO (Phase 1)."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str


class Detector:
    def __init__(self, weights: str = "models/best_yolo.pt", conf: float = 0.25,
                 iou: float = 0.45, imgsz: int = 640, device: Optional[str] = None,
                 half: bool = False):
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device, self.half = device, half
        self.names = self.model.names  # {id: name}

    def class_id(self, name: str) -> Optional[int]:
        for k, v in self.names.items():
            if v == name:
                return int(k)
        return None

    def detect(self, frame: np.ndarray):
        """Return (list[Detection], ultralytics Boxes). Boxes dipakai langsung oleh tracker."""
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, half=self.half, verbose=False)[0]
        dets: List[Detection] = []
        for b in res.boxes:
            cid = int(b.cls.item())
            dets.append(Detection(tuple(b.xyxy[0].tolist()), float(b.conf.item()),
                                  cid, self.names[cid]))
        return dets, res.boxes
