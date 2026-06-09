import os
import random
import shutil

SOURCE = "../datasets/asimetria/normal"   # carpeta normal con subcarpetas
TARGET_SIZE = 3500

random.seed(42)

# 1. CREAR LISTA DE TODAS LAS IMÁGENES (incluye subcarpetas)
all_images = []

for root, dirs, files in os.walk(SOURCE):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, file)
            all_images.append(full_path)

print(f"Total imágenes encontradas: {len(all_images)}")

# 2. SI HAY MÁS DE 3500, SELECCIONAR SOLO 3500
if len(all_images) > TARGET_SIZE:
    selected = set(random.sample(all_images, TARGET_SIZE))
else:
    selected = set(all_images)

# 3. CREAR CARPETA PLANA (sin subcarpetas)
FLAT_DIR = os.path.join(SOURCE, "_flat")
os.makedirs(FLAT_DIR, exist_ok=True)

# 4. COPIAR SOLO LAS SELECCIONADAS A UNA SOLA CARPETA
for i, img_path in enumerate(selected):
    ext = os.path.splitext(img_path)[1]
    new_name = f"normal_{i}{ext}"
    dst = os.path.join(FLAT_DIR, new_name)
    shutil.copy2(img_path, dst)

print(f"✅ Reducido a {len(selected)} imágenes en carpeta plana: {FLAT_DIR}")