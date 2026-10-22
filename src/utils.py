"""Utilitas bersama."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES_YAML = ROOT / "config" / "classes.yaml"


def load_classes(path: Path = CLASSES_YAML) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolved_data_yaml(path: Path = CLASSES_YAML) -> str:
    """Tulis salinan classes.yaml dengan `path` absolut agar Ultralytics tidak salah resolve."""
    cfg = load_classes(path)
    cfg["path"] = str((ROOT / cfg.get("path", "dataset")).resolve())
    out = ROOT / "runs" / "_data_resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out)
