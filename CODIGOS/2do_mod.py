import os
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# =====================================================
# FUNCIÓN: ANÁLISIS SANIDAD DE HOJAS
# =====================================================
def analizar_sanidad(dataset):
    total = 0
    sanas = 0
    enfermas = 0

    for class_idx, class_name in enumerate(dataset.classes):
        num_imgs = sum(1 for _, label in dataset.samples if label == class_idx)
        total += num_imgs

        if "sano" in class_name.lower():
            sanas += num_imgs
        else:
            enfermas += num_imgs

    pct_sanas = (sanas / total) * 100
    pct_enfermas = (enfermas / total) * 100

    print("\n📊 DISTRIBUCIÓN DE SANIDAD DE HOJAS (TRAIN)")
    print(f"🍃 Total de hojas: {total}")
    print(f"✅ Hojas sanas: {sanas} ({pct_sanas:.2f}%)")
    print(f"🦠 Hojas con enfermedad: {enfermas} ({pct_enfermas:.2f}%)")

    return {
        "total": total,
        "sanas": sanas,
        "enfermas": enfermas,
        "pct_sanas": round(pct_sanas, 2),
        "pct_enfermas": round(pct_enfermas, 2)
    }

# =====================================================
# MAIN
# =====================================================
def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "DATA_PROCESADA_HOJAS"
    OUTPUT_DIR = BASE_DIR / "runs" / "classifier"

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 1e-3
    IMG_SIZE = 224
    NUM_WORKERS = 4
    PIN_MEMORY = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =====================================================
    # DISPOSITIVO
    # =====================================================
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️ Usando CPU")

    # =====================================================
    # TRANSFORMACIONES
    # =====================================================
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # =====================================================
    # DATASETS
    # =====================================================
    print("📥 Cargando datasets de clasificación...")

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
    val_dataset   = datasets.ImageFolder(DATA_DIR / "val", transform=val_transforms)
    test_dataset  = datasets.ImageFolder(DATA_DIR / "test", transform=val_transforms)

    num_classes = len(train_dataset.classes)

    print(f"📊 Número de clases: {num_classes}")
    print(f"🏷 Clases: {train_dataset.classes}")

    # =====================================================
    # 🔐 CONSISTENCIA ENTRENAMIENTO–INFERENCIA
    # =====================================================
    with open(OUTPUT_DIR / "classes.json", "w") as f:
        json.dump(train_dataset.classes, f, indent=4)

    print(f"🔒 classes.json guardado en {OUTPUT_DIR}")

    # =====================================================
    # ANÁLISIS DE SANIDAD
    # =====================================================
    sanidad_stats = analizar_sanidad(train_dataset)

    with open(OUTPUT_DIR / "sanidad_dataset.json", "w") as f:
        json.dump(sanidad_stats, f, indent=4)

    # =====================================================
    # DATALOADERS
    # =====================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # =====================================================
    # MODELO
    # =====================================================
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    # =====================================================
    # ENTRENAMIENTO
    # =====================================================
    print("\n🚀 Iniciando entrenamiento del clasificador")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n🟦 Epoch {epoch+1}/{EPOCHS}")

        model.train()
        correct, total, loss_sum = 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc="Entrenando"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_classifier.pt")
            print("💾 Modelo guardado (mejor validación)")

    # =====================================================
    # TEST FINAL
    # =====================================================
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_classifier.pt"))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"\n✅ Accuracy final en TEST: {correct / total:.4f}")

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    main()
