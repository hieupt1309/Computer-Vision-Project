import os
import cv2
import pickle
import numpy as np


# ==================================================
# PATH
# ==================================================

TRAIN_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_traditional\processed_dataset\train"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lbph_model.yml"
)

LABEL_MAP_PATH = os.path.join(
    MODEL_DIR,
    "label_map.pkl"
)


# ==================================================
# LOAD DATA
# ==================================================

faces = []
labels = []

label_map = {}

current_label = 0


for person_name in sorted(
    os.listdir(TRAIN_DIR)
):

    person_dir = os.path.join(
        TRAIN_DIR,
        person_name
    )

    if not os.path.isdir(
        person_dir
    ):
        continue

    label_map[current_label] = (
        person_name
    )

    image_count = 0

    for img_name in os.listdir(
        person_dir
    ):

        img_path = os.path.join(
            person_dir,
            img_name
        )

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        faces.append(image)
        labels.append(
            current_label
        )

        image_count += 1

    print(
        f"{person_name}: "
        f"{image_count} images"
    )

    current_label += 1


labels = np.array(
    labels
)


print("\n====================")
print("TRAIN IMAGES:", len(faces))
print("IDENTITIES:", len(label_map))
print("====================")


# ==================================================
# LBPH MODEL
# ==================================================
# giữ gần baseline cũ
# nhưng tune nhẹ
# ==================================================

model = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)


# ==================================================
# TRAIN
# ==================================================

print("\nTraining LBPH...")

model.train(
    faces,
    labels
)

print("Training Done!")


# ==================================================
# SAVE
# ==================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(
    MODEL_PATH
)

with open(
    LABEL_MAP_PATH,
    "wb"
) as f:

    pickle.dump(
        label_map,
        f
    )

print("\nModel saved!")
print(MODEL_PATH)