
from ultralytics import YOLO
from pathlib import Path

# Lokasi video kafe
video_path = Path("data/videos/coffee_shop.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        f"Video tidak ditemukan: {video_path.resolve()}"
    )

# Memuat model YOLO
model = YOLO("yolov8s.pt")

# Deteksi objek pada video
results = model.predict(
    source=str(video_path),
    conf=0.25,
    classes=[0, 56, 60],
    save=True,
    stream=True,
    verbose=False
)

# Menghitung objek yang terdeteksi pada setiap frame
for frame_number, result in enumerate(results, start=1):
    counts = {
        "person": 0,
        "chair": 0,
        "dining table": 0
    }

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            counts[class_name] += 1

    print(
        f"Frame {frame_number}: "
        f"orang={counts['person']}, "
        f"kursi={counts['chair']}, "
        f"meja={counts['dining table']}"
    )

print("\nDeteksi selesai. Periksa video hasil deteksi.")
