import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from model import FaceModel


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


def load_model(checkpoint_path):
    model = FaceModel(EMBEDDING_DIM, NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def get_embedding(model, face_img):
    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model.backbone(img)
    return F.normalize(emb).squeeze(0).numpy()


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return faces


class FaceAuth:
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "epoch_050.pth")
        self.model = load_model(checkpoint_path)
        self.database = {}

    def register(self, name, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read {img_path}")
            return False
        faces = detect_face(img)
        if len(faces) == 0:
            print("No face detected")
            return False
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        emb = get_embedding(self.model, face)
        self.database[name] = emb
        print(f"Registered {name}")
        return True

    def recognize(self, img_path, threshold=0.5):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read {img_path}")
            return None, 0
        faces = detect_face(img)
        if len(faces) == 0:
            print("No face detected")
            return None, 0
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        emb = get_embedding(self.model, face)

        best_name, best_sim = None, -1
        for name, db_emb in self.database.items():
            sim = float(np.dot(emb, db_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < threshold:
            return None, best_sim
        return best_name, best_sim


if __name__ == "__main__":
    auth = FaceAuth()
    while True:
        print("\n1. Register  2. Recognize  3. Exit")
        choice = input("Choose: ").strip()
        if choice == "1":
            name = input("Name: ").strip()
            path = input("Image path: ").strip()
            auth.register(name, path)
        elif choice == "2":
            path = input("Image path: ").strip()
            name, sim = auth.recognize(path)
            if name:
                print(f"Recognized: {name} (similarity: {sim:.4f})")
            else:
                print(f"No match (best sim: {sim:.4f})")
        elif choice == "3":
            break
