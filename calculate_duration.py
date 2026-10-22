import cv2
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# Lokasi video
video_path = Path("data/videos/coffee_shop.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        f"Video tidak ditemukan: {video_path.resolve()}"
    )

# Memuat model
model = YOLO("yolov8s.pt")

# Menyimpan jumlah frame ketika setiap ID terlihat
person_frames = defaultdict(int)

# Membaca FPS video agar durasi dapat dihitung
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if fps <= 0:
    raise ValueError("FPS video tidak dapat dibaca.")

# Menjalankan tracking
results = model.track(
    source=str(video_path),
    tracker="bytetrack.yaml",
    conf=0.25,
    stream=True,
    persist=True,
    verbose=False
)

for result in results:
    if result.boxes is None or result.boxes.id is None:
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])

        # Hanya menghitung objek orang
        if model.names[class_id] != "person":
            continue

        track_id = int(box.id[0])
        person_frames[track_id] += 1

# Menampilkan perkiraan durasi terlihat
print("\n=== DURASI KEMUNCULAN ORANG ===")

for track_id, frame_count in sorted(person_frames.items()):
    duration = frame_count / fps
    print(
        f"ID {track_id}: "
        f"{duration:.2f} detik terdeteksi"
    )

print("\nPenghitungan selesai.")
