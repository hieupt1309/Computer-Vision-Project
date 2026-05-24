import os
import cv2
import pickle
import numpy as np


# ==================================================
# PATH
# ==================================================

TEST_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_traditional\processed_dataset\test"

MODEL_PATH = "models/lbph_model.yml"

LABEL_MAP_PATH = "models/label_map.pkl"


# ==================================================
# LOAD MODEL
# ==================================================

model = cv2.face.LBPHFaceRecognizer_create()

model.read(
    MODEL_PATH
)

with open(
    LABEL_MAP_PATH,
    "rb"
) as f:

    label_map = pickle.load(f)


# ==================================================
# EVALUATION
# ==================================================

correct = 0
total = 0

correct_confidences = []
wrong_confidences = []


for person_name in sorted(
    os.listdir(TEST_DIR)
):

    person_dir = os.path.join(
        TEST_DIR,
        person_name
    )

    if not os.path.isdir(
        person_dir
    ):
        continue

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

        # ==================================
        # PREDICT
        # ==================================

        pred_label, confidence = (
            model.predict(image)
        )

        pred_name = label_map[
            pred_label
        ]

        # ==================================
        # CHECK CORRECT
        # ==================================

        if pred_name == person_name:

            correct += 1

            correct_confidences.append(
                confidence
            )

        else:

            wrong_confidences.append(
                confidence
            )

        total += 1


# ==================================================
# RESULT
# ==================================================

accuracy = correct / total


print("\n========================")
print("TRADITIONAL METHOD")
print("========================")

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Correct: "
    f"{correct}"
)

print(
    f"Total: "
    f"{total}"
)

if len(correct_confidences) > 0:

    print(
        f"Avg Correct Confidence: "
        f"{np.mean(correct_confidences):.2f}"
    )

if len(wrong_confidences) > 0:

    print(
        f"Avg Wrong Confidence: "
        f"{np.mean(wrong_confidences):.2f}"
    )

print("========================")