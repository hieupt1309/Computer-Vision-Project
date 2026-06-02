import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
loader = DataLoader(BASE_DIR)

INPUT_TRAIN_DIR = os.path.join(loader.processed_dir, "train")
OUTPUT_DIR = loader.split_dir

TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)

train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for person_name in sorted(os.listdir(INPUT_TRAIN_DIR)):
    person_path = os.path.join(INPUT_TRAIN_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)
    random.shuffle(images)
    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    train_imgs = images[:train_end]
    val_imgs = images[train_end:]

    train_person_dir = os.path.join(train_dir, person_name)
    val_person_dir = os.path.join(val_dir, person_name)
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(val_person_dir, exist_ok=True)

    for img_name in train_imgs:
        shutil.copy2(os.path.join(person_path, img_name), os.path.join(train_person_dir, img_name))
    for img_name in val_imgs:
        shutil.copy2(os.path.join(person_path, img_name), os.path.join(val_person_dir, img_name))

    print(f"{person_name}: {len(train_imgs)} train | {len(val_imgs)} val")

print("\n====================")
print("SPLIT DONE")
print("====================")
