import os
import json
import shutil
import random
import cv2
import unicodedata

# =========================
# RUTAS
# =========================
DATA_ETIQUETADA = './DATA_ETIQUETADA'
DATA_PROCESADA = './DATA_PROCESADA'
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# =========================
# NORMALIZAR NOMBRES
# =========================
def normalize_filename(name):
    return unicodedata.normalize('NFKD', name) \
        .encode('ascii', 'ignore') \
        .decode('ascii')

# =========================
# SAFE JSON LOAD
# =========================
def safe_load_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print(f"⚠️ JSON inválido o vacío (omitido): {json_path}")
        return None

# =========================
# CATEGORÍAS
# =========================
categorias = {
    'Papa': {
        'Sana': 'papa_sana',
        'TTAR': 'papa_ttar',
        'TTEM': 'papa_ttem',
    },
    'Pimiento_Morron': {
        'Sano': 'pimiento_sano',
        'MBAC': 'pimiento_mbac',
    },
    'Tomate': {
        'ARAC': 'tomate_arac',
        'MFSEP': 'tomate_mfsep',
        'Sano': 'tomate_sano',
        'MBAC': 'tomate_mbac',
        'MFOL': 'tomate_mfol',
        'TTAR': 'tomate_ttar',
        'MDI': 'tomate_mdi',
        'VMO': 'tomate_vmo',
        'VRALH': 'tomate_vralh',
        'TTEM': 'tomate_ttem',
    }
}

# =========================
# CLASS MAP
# =========================
class_map = {
    'papa_sana': 0,
    'papa_ttar': 1,
    'papa_ttem': 2,
    'pimiento_sano': 3,
    'pimiento_mbac': 4,
    'tomate_arac': 5,
    'tomate_mfsep': 6,
    'tomate_sano': 7,
    'tomate_mbac': 8,
    'tomate_mfol': 9,
    'tomate_ttar': 10,
    'tomate_mdi': 11,
    'tomate_vmo': 12,
    'tomate_vralh': 13,
    'tomate_ttem': 14,
}

# =========================
# CREAR ESTRUCTURA YOLO
# =========================
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DATA_PROCESADA, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DATA_PROCESADA, split, 'labels'), exist_ok=True)

# =========================
# CARGAR IMÁGENES
# =========================
def load_images(data_path):
    data = []

    for planta in os.listdir(data_path):
        planta_path = os.path.join(data_path, planta)
        if not os.path.isdir(planta_path):
            continue

        for enfermedad in os.listdir(planta_path):
            enf_path = os.path.join(planta_path, enfermedad)
            if not os.path.isdir(enf_path):
                continue

            if planta not in categorias or enfermedad not in categorias[planta]:
                continue

            clase_yolo = categorias[planta][enfermedad]

            for file in os.listdir(enf_path):
                if file.lower().endswith(IMG_EXTENSIONS):
                    clean_name = normalize_filename(file)
                    src_img = os.path.join(enf_path, file)
                    img_path = os.path.join(enf_path, clean_name)

                    if file != clean_name:
                        os.rename(src_img, img_path)

                    label_path = os.path.splitext(img_path)[0] + ".json"

                    data.append({
                        "image": img_path,
                        "label": label_path if os.path.exists(label_path) else None,
                        "class": clase_yolo
                    })

    return data

# =========================
# JSON → YOLO SEGMENTATION
# =========================
def convert_to_yolo_seg(image_path, label_path, class_id):
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w, _ = img.shape
    data = safe_load_json(label_path)
    if data is None:
        return []

    yolo_lines = []

    for obj in data.get("objects", []):
        polygon = obj.get("polygon")
        if not polygon or len(polygon) < 3:
            continue

        coords = []
        for x, y in polygon:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")

        yolo_lines.append(f"{class_id} " + " ".join(coords))

    return yolo_lines

# =========================
# SPLIT Y GUARDADO
# =========================
def process_and_save(data):
    for item in data:
        split = random.choices(['train', 'val', 'test'], weights=[70, 15, 15])[0]

        img_dst = os.path.join(
            DATA_PROCESADA, split, 'images',
            os.path.basename(item["image"])
        )
        shutil.copy(item["image"], img_dst)

        if item["label"]:
            class_id = class_map[item["class"]]
            yolo_lines = convert_to_yolo_seg(
                item["image"], item["label"], class_id
            )

            if not yolo_lines:
                continue

            label_name = os.path.splitext(os.path.basename(item["image"]))[0] + ".txt"
            label_dst = os.path.join(DATA_PROCESADA, split, 'labels', label_name)

            with open(label_dst, "w") as f:
                f.write("\n".join(yolo_lines))

# =========================
# EJECUCIÓN
# =========================
data = load_images(DATA_ETIQUETADA)
process_and_save(data)

print("✅ Preprocesamiento completado. Dataset YOLOv8-SEG generado correctamente.")
