import os
import cv2
import pickle
import numpy as np


# ==================================================
# PATH
# ==================================================

TEST_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_traditional\dataset_split\test"

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
# COLLECT SCORES
# ==================================================

genuine_scores = []
impostor_scores = []


print(
    "Collecting scores..."
)

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

        # ==========================
        # MODEL PREDICT
        # ==========================

        pred_label, confidence = (
            model.predict(image)
        )

        pred_name = label_map[
            pred_label
        ]

        # genuine
        if pred_name == person_name:

            genuine_scores.append(
                confidence
            )

        # impostor
        else:

            impostor_scores.append(
                confidence
            )


# ==================================================
# THRESHOLD SEARCH
# ==================================================

all_scores = np.concatenate([
    genuine_scores,
    impostor_scores
])

thresholds = np.linspace(
    all_scores.min(),
    all_scores.max(),
    300
)


best_acc = 0

best_threshold = 0
best_far = 0
best_frr = 0
best_eer_gap = 999


for threshold in thresholds:

    # genuine
    TP = sum(
        s < threshold
        for s in genuine_scores
    )

    FN = sum(
        s >= threshold
        for s in genuine_scores
    )

    # impostor
    FP = sum(
        s < threshold
        for s in impostor_scores
    )

    TN = sum(
        s >= threshold
        for s in impostor_scores
    )

    FAR = FP / (
        FP + TN + 1e-6
    )

    FRR = FN / (
        TP + FN + 1e-6
    )

    ACC = (
        TP + TN
    ) / (
        TP + TN
        + FP + FN
    )

    eer_gap = abs(
        FAR - FRR
    )

    # maximize accuracy
    # tie break by EER

    if (
        ACC > best_acc
        or
        (
            abs(
                ACC
                - best_acc
            ) < 1e-4
            and
            eer_gap
            < best_eer_gap
        )
    ):

        best_acc = ACC

        best_threshold = (
            threshold
        )

        best_far = FAR
        best_frr = FRR

        best_eer_gap = (
            eer_gap
        )


# ==================================================
# RESULT
# ==================================================

print("\n========================")
print("AUTHENTICATION RESULT")
print("========================")

print(
    f"Best Threshold: "
    f"{best_threshold:.4f}"
)

print(
    f"Authentication Accuracy: "
    f"{best_acc:.4f}"
)

print(
    f"FAR: "
    f"{best_far:.4f}"
)

print(
    f"FRR: "
    f"{best_frr:.4f}"
)

print("========================")