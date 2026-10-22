"""Evaluasi detektor: Precision, Recall, mAP@50, mAP@50-95 (Phase 1).

  python eval.py --weights models/best_yolo.pt --split test
"""
import argparse
import json

from ultralytics import YOLO

from src.utils import ROOT, load_classes, resolved_data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best_yolo.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    names = load_classes()["names"]
    m = YOLO(a.weights).val(data=resolved_data_yaml(), split=a.split, imgsz=a.imgsz,
                            conf=a.conf, device=a.device, plots=True,
                            project=str(ROOT / "runs" / "eval"), name=a.split, exist_ok=True)

    out = {
        "overall": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                    "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)},
        "per_class": {},
    }
    for i, c in enumerate(m.box.ap_class_index):
        out["per_class"][names[int(c)]] = {
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "mAP50": float(m.box.ap50[i]), "mAP50-95": float(m.box.ap[i]),
        }
    print(json.dumps(out, indent=2))
    path = ROOT / "runs" / "eval" / f"metrics_{a.split}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Disimpan: {path}")


if __name__ == "__main__":
    main()
