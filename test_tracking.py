
from ultralytics import YOLO
from pathlib import Path

# Lokasi video
video_path = Path("data/videos/kopisop.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        f"Video tidak ditemukan: {video_path.resolve()}"
    )

# Memuat model YOLO
model = YOLO("yolov8s.pt")

# Menjalankan object tracking menggunakan ByteTrack
results = model.track(
    source=str(video_path),
    tracker="bytetrack.yaml",
    conf=0.25,
    persist=True,
    show=False,
    save=True,
    stream=True
)

# Membaca hasil tracking
for result in results:
    if result.boxes is None:
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        track_id = (
            int(box.id[0])
            if box.id is not None
            else None
        )

        if model.names[class_id] == "person":
            print(
                f"ID: {track_id} | "
                f"Objek: person | "
                f"Confidence: {confidence:.2f}"
            )

print("Tracking selesai. Periksa folder runs/detect/track.")
