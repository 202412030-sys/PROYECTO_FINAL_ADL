import os
import shutil
import random
from pathlib import Path

# =====================================================
# CONFIGURACIÓN
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_INPUT = BASE_DIR / "DATA_ETIQUETADA_HOJAS"
DATA_OUTPUT = BASE_DIR / "DATA_PROCESADA_HOJAS"

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

SPLITS = ["train", "val", "test"]
SPLIT_WEIGHTS_REST = [0.6, 0.2, 0.2]  # para imágenes restantes

random.seed(42)

# =====================================================
# CREAR ESTRUCTURA BASE
# =====================================================
for split in SPLITS:
    os.makedirs(DATA_OUTPUT / split, exist_ok=True)

# =====================================================
# PROCESAMIENTO CON GARANTÍA DE TRAIN
# =====================================================
def process_hojas():
    total = 0
    skipped_classes = []

    for cultivo in os.listdir(DATA_INPUT):
        cultivo_path = DATA_INPUT / cultivo
        if not cultivo_path.is_dir():
            continue

        for enfermedad in os.listdir(cultivo_path):
            enf_path = cultivo_path / enfermedad
            if not enf_path.is_dir():
                continue

            class_name = f"{cultivo.lower()}_{enfermedad.lower()}"

            images = [
                f for f in os.listdir(enf_path)
                if f.lower().endswith(IMG_EXTENSIONS)
            ]

            # ⚠️ Si no hay imágenes, se omite la clase
            if len(images) == 0:
                skipped_classes.append(class_name)
                continue

            # Crear carpetas de salida
            for split in SPLITS:
                os.makedirs(DATA_OUTPUT / split / class_name, exist_ok=True)

            random.shuffle(images)

            # 🔒 Garantizar al menos 1 imagen en train
            first_img = images[0]
            shutil.copy(
                enf_path / first_img,
                DATA_OUTPUT / "train" / class_name / first_img
            )
            total += 1

            # Resto de imágenes
            for file in images[1:]:
                split = random.choices(
                    SPLITS,
                    weights=SPLIT_WEIGHTS_REST
                )[0]

                shutil.copy(
                    enf_path / file,
                    DATA_OUTPUT / split / class_name / file
                )
                total += 1

    print(f"✅ Total de hojas procesadas: {total}")

    if skipped_classes:
        print("⚠️ Clases omitidas por no tener imágenes:")
        for c in skipped_classes:
            print(f"   - {c}")

# =====================================================
# EJECUCIÓN
# =====================================================
if __name__ == "__main__":
    process_hojas()
