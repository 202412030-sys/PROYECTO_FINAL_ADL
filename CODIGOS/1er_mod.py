from ultralytics import YOLO
from pathlib import Path

# =====================================================
# RESOLUCIÓN DE RUTAS (ROBUSTA)
# =====================================================
# Ruta del archivo actual: PROYECTO_FINAL/CODIGOS/1er_mod.py
BASE_DIR = Path(__file__).resolve().parent.parent  # PROYECTO_FINAL

DATA_YAML = BASE_DIR / "data_hojas.yaml"
MODEL_BASE = BASE_DIR / "yolov8n.pt"

# =====================================================
# CONFIGURACIÓN DE EXPERIMENTO
# =====================================================
PROJECT_NAME = "yolo_hojas"
EXPERIMENT_NAME = "v1"

EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8

# =====================================================
# ENTRENAMIENTO
# =====================================================
def train_yolo_hojas():
    print("🚀 Iniciando entrenamiento YOLOv8 - Detección de Hojas")
    print(f"📄 Usando dataset: {DATA_YAML}")
    print(f"🧠 Modelo base: {MODEL_BASE}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"❌ No se encontró data.yaml en {DATA_YAML}")

    if not MODEL_BASE.exists():
        raise FileNotFoundError(f"❌ No se encontró el modelo base en {MODEL_BASE}")

    model = YOLO(str(MODEL_BASE))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(BASE_DIR / "runs" / "detect"),
        name=f"{PROJECT_NAME}_{EXPERIMENT_NAME}",
        exist_ok=True
    )

    print("✅ Entrenamiento finalizado correctamente")

# =====================================================
# EJECUCIÓN
# =====================================================
if __name__ == "__main__":
    train_yolo_hojas()
