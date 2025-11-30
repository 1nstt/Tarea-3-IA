import os
import shutil
import random

DATASET_DIR = "dataset"      
OUTPUT_DIR = "dataset_split"

# detectar clases automáticamente
CLASSES = [d for d in os.listdir(DATASET_DIR)
           if os.path.isdir(os.path.join(DATASET_DIR, d))]

print("Clases detectadas:", CLASSES)

# extensiones permitidas
EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# porcentajes
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# crear carpetas destino
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# función para obtener todas las imágenes, incluso en subcarpetas
def collect_images(path):
    imgs = []
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f.lower())[1] in EXTS:
                imgs.append(os.path.join(root, f))
    return imgs

# dividir por clase
for cls in CLASSES:
    class_path = os.path.join(DATASET_DIR, cls)

    images = collect_images(class_path)

    print(f"Encontradas en {cls}: {len(images)} imágenes")

    random.shuffle(images)
    n = len(images)

    if n == 0:
        print(f"ERROR: No se encontraron imágenes en {cls}")
        continue

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # copiar
    for f in train_imgs:
        shutil.copy(f, os.path.join(OUTPUT_DIR, "train", cls))

    for f in val_imgs:
        shutil.copy(f, os.path.join(OUTPUT_DIR, "val", cls))

    for f in test_imgs:
        shutil.copy(f, os.path.join(OUTPUT_DIR, "test", cls))

    print(f"{cls}: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

print("\nDIVISIÓN COMPLETA")