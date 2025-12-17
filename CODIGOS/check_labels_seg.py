import os

LABEL_DIR = "./DATA_PROCESADA/train/labels"

def check_labels():
    for f in os.listdir(LABEL_DIR):
        if not f.endswith(".txt"):
            continue

        with open(os.path.join(LABEL_DIR, f)) as file:
            line = file.readline().strip().split()

            if len(line) <= 5:
                print(f"❌ NO ES SEGMENTACIÓN: {f}")
                return

    print("✅ Labels correctos: formato YOLOv8-SEG confirmado")

if __name__ == "__main__":
    check_labels()
