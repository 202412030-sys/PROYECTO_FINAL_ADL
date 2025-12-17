import os
import json
import shutil
import random
from pathlib import Path

# =====================================================
# CONFIGURACIÓN
# =====================================================
DATA_INPUT = "DATA_ETIQUETADA_PLANTAS"
DATA_OUTPUT = "DATA_PROCESADA_PLANTAS"

SPLIT_RATIO = 0.8
CLASS_ID = 0          # hojas (clase única YOLO)
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)

# =====================================================
# CREAR ESTRUCTURA YOLO
# =====================================================
for split in ["train", "val"]:
    os.makedirs(os.path.join(DATA_OUTPUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT, "labels", split), exist_ok=True)

# =====================================================
# UTILIDADES
# =====================================================
def convert_bbox_to_yolo(points, img_w, img_h):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    return x_center, y_center, w, h

# =====================================================
# RECOLECTAR MUESTRAS
# =====================================================
samples = []

for cultivo in os.listdir(DATA_INPUT):
    cultivo_path = os.path.join(DATA_INPUT, cultivo)
    if not os.path.isdir(cultivo_path):
        continue

    for file in os.listdir(cultivo_path):
        if file.lower().endswith(IMG_EXTENSIONS):
            img_path = os.path.join(cultivo_path, file)
            json_path = img_path.replace(Path(img_path).suffix, ".json")

            if os.path.exists(json_path):
                samples.append((img_path, json_path))

print(f"📦 Total de imágenes encontradas: {len(samples)}")

if len(samples) == 0:
    raise RuntimeError("❌ No se encontraron imágenes con JSON asociados")

random.shuffle(samples)
split_idx = int(len(samples) * SPLIT_RATIO)

train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

# =====================================================
# PROCESAR SPLITS
# =====================================================
def process_split(samples, split_name):
    processed = 0

    for img_path, json_path in samples:
        img_name = os.path.basename(img_path)
        label_name = img_name.replace(Path(img_name).suffix, ".txt")

        shutil.copy(
            img_path,
            os.path.join(DATA_OUTPUT, "images", split_name, img_name)
        )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data["imageWidth"]
        img_h = data["imageHeight"]

        yolo_lines = []

        for shape in data.get("shapes", []):
            points = shape["points"]

            x_c, y_c, w, h = convert_bbox_to_yolo(points, img_w, img_h)

            yolo_lines.append(
                f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            )

        if yolo_lines:
            with open(
                os.path.join(DATA_OUTPUT, "labels", split_name, label_name),
                "w"
            ) as f:
                f.write("\n".join(yolo_lines))
            processed += 1

    print(f"✅ {split_name}: {processed} imágenes procesadas")

# =====================================================
# EJECUCIÓN
# =====================================================
process_split(train_samples, "train")
process_split(val_samples, "val")

print("🎯 Dataset YOLO-HOJAS generado correctamente")
