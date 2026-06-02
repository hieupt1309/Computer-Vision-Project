import os, sys, pickle, cv2
import numpy as np
from deepface import DeepFace

GALLERY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "gallery.pkl")
BACKEND = "opencv"
MODEL = "ArcFace"

os.makedirs(os.path.dirname(GALLERY), exist_ok=True)

def get_embedding(img_path):
    emb = DeepFace.represent(img_path, model_name=MODEL, detector_backend=BACKEND, enforce_detection=False)
    return np.array(emb[0]["embedding"])

def register(name, img_path):
    emb = get_embedding(img_path)
    gallery = {"embeddings": [], "names": []}
    if os.path.exists(GALLERY):
        with open(GALLERY, "rb") as f:
            gallery = pickle.load(f)
    gallery["embeddings"].append(emb)
    gallery["names"].append(name)
    with open(GALLERY, "wb") as f:
        pickle.dump(gallery, f)
    print(f"Registered {name}")

def register_from_dir(person_dir):
    name = os.path.basename(person_dir.rstrip("/\\"))
    gallery = {"embeddings": [], "names": []}
    if os.path.exists(GALLERY):
        with open(GALLERY, "rb") as f:
            gallery = pickle.load(f)
    count = 0
    for f in os.listdir(person_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            emb = get_embedding(os.path.join(person_dir, f))
            gallery["embeddings"].append(emb)
            gallery["names"].append(name)
            count += 1
    with open(GALLERY, "wb") as f:
        pickle.dump(gallery, f)
    print(f"Registered {name} with {count} images")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python register.py <name> <image_path>")
        print("       python register.py --dir <person_dir>")
        sys.exit(1)
    if sys.argv[1] == "--dir":
        register_from_dir(sys.argv[2])
    else:
        register(sys.argv[1], sys.argv[2])
