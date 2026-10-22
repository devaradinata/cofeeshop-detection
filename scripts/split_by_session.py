"""Bagi dataset ke train/val/test PER SESI VIDEO (bukan per frame acak) agar tidak ada data leakage.

Struktur input (setelah anotasi, format YOLO .txt):
  data/raw/<sesi>/images/*.jpg
  data/raw/<sesi>/labels/*.txt

  python scripts/split_by_session.py --raw data/raw --out dataset --ratios 0.7 0.15 0.15
"""
import argparse
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sessions = sorted(p for p in Path(a.raw).iterdir() if (p / "images").is_dir())
    if len(sessions) < 3:
        raise SystemExit("Butuh minimal 3 sesi video agar split train/val/test tanpa leakage.")
    random.Random(a.seed).shuffle(sessions)
    n = len(sessions)
    n_tr = max(1, round(n * a.ratios[0]))
    n_va = max(1, round(n * a.ratios[1]))
    n_tr = min(n_tr, n - 2)
    n_va = min(n_va, n - n_tr - 1)
    groups = {"train": sessions[:n_tr], "val": sessions[n_tr:n_tr + n_va],
              "test": sessions[n_tr + n_va:]}

    for split, sess in groups.items():
        (Path(a.out) / split / "images").mkdir(parents=True, exist_ok=True)
        (Path(a.out) / split / "labels").mkdir(parents=True, exist_ok=True)
        cnt = 0
        for s in sess:
            for img in (s / "images").glob("*"):
                lbl = s / "labels" / (img.stem + ".txt")
                if not lbl.exists():
                    continue  # lewati frame yang belum dianotasi
                shutil.copy(img, Path(a.out) / split / "images" / img.name)
                shutil.copy(lbl, Path(a.out) / split / "labels" / lbl.name)
                cnt += 1
        print(f"{split}: {len(sess)} sesi ({', '.join(s.name for s in sess)}) -> {cnt} gambar")


if __name__ == "__main__":
    main()
