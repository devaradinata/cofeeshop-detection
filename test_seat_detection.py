from pathlib import Path
from ultralytics import YOLO

from src.detector import resolve_model_weights

# Lokasi video kafe (default ke kopisop.mp4)
video_path = Path("data/videos/kopisop.mp4") if Path("data/videos/kopisop.mp4").exists() else Path("data/videos/coffee_shop.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        f"Video tidak ditemukan: {video_path.resolve()}"
    )

# Memuat model YOLO terbaik yang sudah di-train (atau fallback ke yolov8s.pt)
model_path = resolve_model_weights("models/best_yolo.pt")
print(f"Menggunakan model: {model_path}")
model = YOLO(model_path)

# Deteksi objek pada video
results = model.predict(
    source=str(video_path),
    conf=0.20,
    save=False,
    stream=True,
    verbose=False
)

# Menghitung objek yang terdeteksi pada setiap frame
for frame_number, result in enumerate(results, start=1):
    counts = {}

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1

    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"Frame {frame_number}: {summary}")
    if frame_number >= 30:  # Batasi preview terminal agar responsif
        break

print("\nDeteksi selesai.")
