import os
import pickle
import numpy as np

from tqdm import tqdm
from deepface import DeepFace

from sklearn.metrics import (
    accuracy_score
)


# ===================================================
# PATH
# ===================================================

TEST_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_arcface\processed_dataset\test"

MODEL_PATH = "arcface_classifier.pkl"


# ===================================================
# SETTINGS
# ===================================================

MODEL_NAME = "ArcFace"

DETECTOR = "retinaface"


# ===================================================
# LOAD MODEL
# ===================================================

with open(
    MODEL_PATH,
    "rb"
) as f:

    clf, encoder = pickle.load(f)


# ===================================================
# EXTRACT EMBEDDING
# ===================================================

def extract_embedding(img_path):

    embedding = DeepFace.represent(

        img_path=img_path,

        model_name=MODEL_NAME,

        detector_backend=DETECTOR,

        enforce_detection=False
    )

    emb = np.array(
        embedding[0]["embedding"],
        dtype=np.float32
    )

    emb = emb / np.linalg.norm(emb)

    return emb


# ===================================================
# EVALUATE
# ===================================================

y_true = []
y_pred = []

people = sorted(
    os.listdir(TEST_DIR)
)

for person in people:

    person_path = os.path.join(
        TEST_DIR,
        person
    )

    if not os.path.isdir(
        person_path
    ):
        continue

    images = os.listdir(
        person_path
    )

    for img_name in tqdm(
        images,
        desc=person
    ):

        img_path = os.path.join(
            person_path,
            img_name
        )

        try:

            emb = extract_embedding(
                img_path
            )

            pred = clf.predict(
                [emb]
            )[0]

            pred_name = encoder.inverse_transform(
                [pred]
            )[0]

            y_true.append(
                person
            )

            y_pred.append(
                pred_name
            )

        except Exception as e:

            print(
                "Error:",
                img_name
            )


# ===================================================
# RESULT
# ===================================================

acc = accuracy_score(
    y_true,
    y_pred
)

print("\n======================")
print("ARCFACE IDENTIFICATION")
print("======================")
print(f"Top-1 Accuracy: {acc:.4f}")
print(f"Correct: {(np.array(y_true)==np.array(y_pred)).sum()}")
print(f"Total: {len(y_true)}")
print("======================")