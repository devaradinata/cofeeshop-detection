
from ultralytics import YOLO
from pathlib import Path

# 1. Memuat model YOLO
model = YOLO("yolov8s.pt")

# 2. Menentukan gambar yang akan diuji
image_path = Path("coffee_shop.jpg")

if not image_path.exists():
    raise FileNotFoundError(
        "Gambar coffee_shop.jpg belum ada di folder proyek."
    )

# 3. Mendeteksi objek pada gambar
results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True
)

# 4. Menampilkan hasil deteksi
for result in results:
    print("Objek yang terdeteksi:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"- {model.names[class_id]} "
            f"(confidence: {confidence:.2f})"
        )

print("\nSelesai! Periksa lokasi hasil deteksi yang tercetak di terminal.")
