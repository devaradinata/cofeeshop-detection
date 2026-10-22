"""Evaluasi tracking (Phase 2): IDF1, ID Switches, MOTA via py-motmetrics.
Format file: MOTChallenge  ->  frame,id,x,y,w,h,conf,-1,-1,-1 (frame 1-based)

  python evaluate_tracking.py --gt data/gt/session1.txt --pred runs/track/pred.txt

HOTA: motmetrics tidak menghitung HOTA. Untuk HOTA gunakan TrackEval
(https://github.com/JonathonLuiten/TrackEval) dengan file gt/pred yang sama.
"""
import argparse
from collections import defaultdict

import motmetrics as mm
import numpy as np


def load_mot(path):
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    frames = defaultdict(lambda: ([], []))
    for r in data:
        ids, boxes = frames[int(r[0])]
        ids.append(int(r[1]))
        boxes.append(r[2:6])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    gt, pr = load_mot(a.gt), load_mot(a.pred)
    acc = mm.MOTAccumulator(auto_id=False)
    for f in sorted(set(gt) | set(pr)):
        g_ids, g_boxes = gt[f] if f in gt else ([], [])
        p_ids, p_boxes = pr[f] if f in pr else ([], [])
        g_boxes = np.array(g_boxes).reshape(-1, 4)
        p_boxes = np.array(p_boxes).reshape(-1, 4)
        d = mm.distances.iou_matrix(g_boxes, p_boxes, max_iou=1 - a.iou)
        acc.update(g_ids, p_ids, d, frameid=f)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["idf1", "idp", "idr", "num_switches",
                                       "num_fragmentations", "mota", "motp"], name="tracking")
    print(mm.io.render_summary(summary, formatters=mh.formatters,
                               namemap=mm.io.motchallenge_metric_names))


if __name__ == "__main__":
    main()
