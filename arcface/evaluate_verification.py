import os
import random
import numpy as np

from deepface import DeepFace
from sklearn.metrics import accuracy_score


# ===================================================
# PATH
# ===================================================

TEST_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_arcface\processed_dataset\test"


# ===================================================
# SETTINGS
# ===================================================

MODEL_NAME = "ArcFace"

THRESHOLD = 0.65


# ===================================================
# EMBEDDING
# ===================================================

def extract_embedding(img_path):

    emb = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False
    )

    vec = np.array(
        emb[0]["embedding"],
        dtype=np.float32
    )

    vec = vec / np.linalg.norm(vec)

    return vec


# ===================================================
# COSINE
# ===================================================

def cosine_similarity(a, b):

    return np.dot(a, b)


# ===================================================
# BUILD PAIRS
# ===================================================

same_scores = []
diff_scores = []

people = sorted(
    os.listdir(TEST_DIR)
)

# SAME PERSON
for person in people:

    person_dir = os.path.join(
        TEST_DIR,
        person
    )

    imgs = os.listdir(person_dir)

    if len(imgs) < 2:
        continue

    for i in range(len(imgs)-1):

        img1 = os.path.join(
            person_dir,
            imgs[i]
        )

        img2 = os.path.join(
            person_dir,
            imgs[i+1]
        )

        emb1 = extract_embedding(img1)
        emb2 = extract_embedding(img2)

        sim = cosine_similarity(
            emb1,
            emb2
        )

        same_scores.append(sim)


# DIFFERENT PERSON
for _ in range(len(same_scores)):

    p1, p2 = random.sample(
        people,
        2
    )

    img1 = random.choice(
        os.listdir(
            os.path.join(TEST_DIR, p1)
        )
    )

    img2 = random.choice(
        os.listdir(
            os.path.join(TEST_DIR, p2)
        )
    )

    path1 = os.path.join(
        TEST_DIR,
        p1,
        img1
    )

    path2 = os.path.join(
        TEST_DIR,
        p2,
        img2
    )

    emb1 = extract_embedding(path1)
    emb2 = extract_embedding(path2)

    sim = cosine_similarity(
        emb1,
        emb2
    )

    diff_scores.append(sim)


# ===================================================
# EVALUATE
# ===================================================

y_true = []
y_pred = []

for s in same_scores:

    y_true.append(1)

    y_pred.append(
        1 if s > THRESHOLD else 0
    )

for s in diff_scores:

    y_true.append(0)

    y_pred.append(
        1 if s > THRESHOLD else 0
    )

acc = accuracy_score(
    y_true,
    y_pred
)

print("\n======================")
print("ARCFACE AUTHENTICATION")
print("======================")
print(f"Accuracy: {acc:.4f}")
print("======================")