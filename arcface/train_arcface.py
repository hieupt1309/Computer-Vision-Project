import os
import pickle
import numpy as np

from tqdm import tqdm
from deepface import DeepFace

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# ===================================================
# PATH
# ===================================================

TRAIN_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_arcface\processed_dataset\train"

MODEL_SAVE = "arcface_classifier.pkl"

EMBEDDING_CACHE = "train_embedding_cache.pkl"


# ===================================================
# SETTINGS
# ===================================================

MODEL_NAME = "ArcFace"

DETECTOR = "retinaface"


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

    # IMPORTANT
    emb = emb / np.linalg.norm(emb)

    return emb


# ===================================================
# BUILD EMBEDDINGS
# ===================================================

def build_embeddings():

    X = []
    y = []

    people = sorted(
        os.listdir(TRAIN_DIR)
    )

    for person in tqdm(
        people,
        desc="Extracting embeddings"
    ):

        person_path = os.path.join(
            TRAIN_DIR,
            person
        )

        if not os.path.isdir(
            person_path
        ):
            continue

        for img_name in os.listdir(
            person_path
        ):

            img_path = os.path.join(
                person_path,
                img_name
            )

            try:

                emb = extract_embedding(
                    img_path
                )

                X.append(emb)
                y.append(person)

            except Exception as e:

                print(
                    f"Skip: {img_path}"
                )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(y)

    with open(
        EMBEDDING_CACHE,
        "wb"
    ) as f:

        pickle.dump(
            (X, y),
            f
        )

    return X, y


# ===================================================
# LOAD CACHE
# ===================================================

if os.path.exists(
    EMBEDDING_CACHE
):

    print(
        "Loading embedding cache..."
    )

    with open(
        EMBEDDING_CACHE,
        "rb"
    ) as f:

        X_train, y_train = pickle.load(f)

else:

    X_train, y_train = build_embeddings()


# ===================================================
# LABEL ENCODER
# ===================================================

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(
    y_train
)


# ===================================================
# TRAIN CLASSIFIER
# ===================================================

print("\nTraining classifier...")

clf = LogisticRegression(

    C=10,

    max_iter=5000,

    solver="lbfgs",

    n_jobs=-1,

    multi_class="multinomial"
)

clf.fit(
    X_train,
    y_train_encoded
)


# ===================================================
# SAVE
# ===================================================

with open(
    MODEL_SAVE,
    "wb"
) as f:

    pickle.dump(
        (clf, encoder),
        f
    )


print("\n======================")
print("TRAINING DONE")
print("======================")
print(
    f"Saved: {MODEL_SAVE}"
)