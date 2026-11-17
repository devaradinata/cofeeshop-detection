from pathlib import Path
from ultralytics import YOLO

from src.detector import resolve_model_weights

# 1. Memuat model YOLO (otomatis memprioritaskan model hasil fine-tuning)
model_path = resolve_model_weights("models/best_yolo.pt")
print(f"Menggunakan model: {model_path}")
model = YOLO(model_path)

# 2. Menentukan gambar yang akan diuji
image_path = Path("coffee_shop.jpg")

if not image_path.exists():
    raise FileNotFoundError(
        "Gambar coffee_shop.jpg belum ada di folder proyek."
    )

# 3. Mendeteksi objek pada gambar
results = model.predict(
    source=str(image_path),
    conf=0.20,
    save=True
)

# 4. Menampilkan hasil deteksi
for result in results:
    print("\nObjek yang terdeteksi:")

    counts = {}
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = model.names[class_id]
        counts[name] = counts.get(name, 0) + 1

        print(
            f"- {name} "
            f"(confidence: {confidence:.2f})"
        )

    print("\nRingkasan total deteksi:")
    for k, v in counts.items():
        print(f"  * {k}: {v} objek")

print("\nSelesai! Periksa lokasi hasil deteksi yang tercetak di terminal.")
