import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
loader = DataLoader(BASE_DIR)

INPUT_DIR = loader.raw_dir
OUTPUT_DIR = loader.processed_dir

IMAGE_SIZE = 112

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
saved_count = 0
skipped_count = 0

for split in ["train", "test"]:
    input_split = os.path.join(INPUT_DIR, split)
    output_split = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_split, exist_ok=True)

    print(f"\nPROCESSING {split.upper()} SET")

    for person_name in sorted(os.listdir(input_split)):
        person_path = os.path.join(input_split, person_name)
        if not os.path.isdir(person_path):
            continue

        save_person = os.path.join(output_split, person_name)
        os.makedirs(save_person, exist_ok=True)

        count = 0
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                skipped_count += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            if len(faces) == 0:
                skipped_count += 1
                continue

            x, y, w, h = faces[0]
            face = cv2.resize(img[y:y+h, x:x+w], (IMAGE_SIZE, IMAGE_SIZE))
            cv2.imwrite(os.path.join(save_person, f"{count}.jpg"), face)
            count += 1
            saved_count += 1

        print(f"  {person_name}: {count}")

print("\n======================")
print("PREPROCESS DONE")
print("======================")
print(f"Saved: {saved_count}")
print(f"Skipped: {skipped_count}")
