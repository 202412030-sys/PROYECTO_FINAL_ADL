import os
import json
import cv2
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent

IMG_DEMO_DIR = BASE_DIR / "IMAGENES_DEMOSTRACION"
VIDEO_DEMO_DIR = BASE_DIR / "VIDEO_DEMOSTRACION"

OUTPUT_DIR = BASE_DIR / "RESULTADOS_INFERENCIA"
IMG_OUT_DIR = OUTPUT_DIR / "imagenes"
VIDEO_OUT_DIR = OUTPUT_DIR / "videos"
METADATA_DIR = OUTPUT_DIR / "metadata"

for d in [IMG_OUT_DIR, VIDEO_OUT_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================
# DISPOSITIVO
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Inferencia usando dispositivo: {DEVICE}")

# =====================================================
# RUTAS DE MODELOS
# =====================================================
YOLO_MODEL_PATH = BASE_DIR / "runs" / "detect" / "yolo_hojas_v1" / "weights" / "best.pt"
CLASSIFIER_PATH = BASE_DIR / "runs" / "classifier" / "best_classifier.pt"
CLASSES_PATH = BASE_DIR / "runs" / "classifier" / "classes.json"

# =====================================================
# CARGAR CLASES (ROBUSTO)
# =====================================================
if CLASSES_PATH.exists():
    with open(CLASSES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    print(f"🏷 Clases cargadas desde classes.json ({len(CLASS_NAMES)})")
else:
    CLASS_NAMES = [
        "papa_sana", "papa_ttar",
        "pimiento_morron_mbac", "pimiento_morron_sano",
        "tomate_arac", "tomate_mbac", "tomate_mdi", "tomate_mfol",
        "tomate_mfsep", "tomate_sano", "tomate_ttar",
        "tomate_ttem", "tomate_vmo", "tomate_vralh"
    ]
    print("⚠️ classes.json no encontrado, usando fallback seguro")

print(f"🏷 Clases finales: {CLASS_NAMES}")

# =====================================================
# CARGAR MODELOS
# =====================================================
yolo_model = YOLO(str(YOLO_MODEL_PATH))
print(f"✅ YOLO cargado: {YOLO_MODEL_PATH.name}")

classifier = models.efficientnet_b0(weights=None)
classifier.classifier[1] = nn.Linear(
    classifier.classifier[1].in_features,
    len(CLASS_NAMES)
)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier.to(DEVICE)
classifier.eval()

print(f"✅ Clasificador cargado: {CLASSIFIER_PATH.name}")

# =====================================================
# TRANSFORMACIÓN PARA CLASIFICADOR
# =====================================================
clf_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CONF_THRESHOLD = 0.60

# =====================================================
# UTILIDAD: VALIDAR CLASE
# =====================================================
def es_clase_valida(nombre):
    if nombre is None:
        return False
    if nombre == "indefinido":
        return False
    if " " in nombre:           # filtra errores tipo "tomat e_arac"
        return False
    return True

# =====================================================
# CLASIFICAR HOJA
# =====================================================
def classify_leaf(crop):
    img = clf_transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = classifier(img)
        probs = torch.softmax(outputs, dim=1)
        conf, cls_id = torch.max(probs, dim=1)

    if conf.item() < CONF_THRESHOLD:
        return "indefinido", float(conf)

    return CLASS_NAMES[int(cls_id)], float(conf)

# =====================================================
# INFERENCIA EN IMÁGENES
# =====================================================
def infer_images():
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "tipo": "imagenes",
        "resultados": []
    }

    for cultivo in os.listdir(IMG_DEMO_DIR):
        cultivo_path = IMG_DEMO_DIR / cultivo
        if not cultivo_path.is_dir():
            continue

        print(f"🌱 Procesando imágenes de: {cultivo}")

        for img_name in os.listdir(cultivo_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = cultivo_path / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            results = yolo_model.predict(
                source=img,
                conf=0.25,
                device=DEVICE,
                save=False
            )

            detecciones = []

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    clase, prob = classify_leaf(crop)

                    if es_clase_valida(clase):
                        detecciones.append({
                            "bbox": [x1, y1, x2, y2],
                            "clase": clase,
                            "confianza": prob
                        })

                    # dibujo siempre
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"{clase} ({prob:.2f})",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            total = len(detecciones)
            sanas = sum(1 for d in detecciones if "sano" in d["clase"])
            enfermas = total - sanas

            pct_sanas = (sanas / total * 100) if total > 0 else 0
            pct_enfermas = (enfermas / total * 100) if total > 0 else 0

            out_name = f"{cultivo}_{img_name}"
            cv2.imwrite(str(IMG_OUT_DIR / out_name), img)

            metadata["resultados"].append({
                "archivo": out_name,
                "cultivo": cultivo,
                "total_hojas_validas": total,
                "hojas_sanas": sanas,
                "hojas_enfermas": enfermas,
                "pct_sanas": round(pct_sanas, 2),
                "pct_enfermas": round(pct_enfermas, 2),
                "detecciones": detecciones
            })

    with open(METADATA_DIR / "inferencia_imagenes.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Inferencia en imágenes completada")

# =====================================================
# INFERENCIA EN VIDEOS
# =====================================================
def infer_videos():
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "tipo": "videos",
        "resultados": []
    }

    for vid_name in os.listdir(VIDEO_DEMO_DIR):
        if not vid_name.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        vid_path = VIDEO_DEMO_DIR / vid_name
        cap = cv2.VideoCapture(str(vid_path))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = VIDEO_OUT_DIR / vid_name
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(str(out_vid), fourcc, fps, (w, h))

        total_frames, total_valid = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model.predict(
                source=frame,
                conf=0.25,
                device=DEVICE,
                save=False
            )

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    clase, prob = classify_leaf(crop)
                    if es_clase_valida(clase):
                        total_valid += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"{clase} ({prob:.2f})",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )

            writer.write(frame)
            total_frames += 1

        cap.release()
        writer.release()

        metadata["resultados"].append({
            "archivo": vid_name,
            "total_frames": total_frames,
            "hojas_validas_detectadas": total_valid
        })

    with open(METADATA_DIR / "inferencia_videos.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Inferencia en videos completada")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    infer_images()
    infer_videos()
    print("🎯 Inferencia completa. Resultados en RESULTADOS_INFERENCIA/")
