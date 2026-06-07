import os, sys, pickle, cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import load_checkpoint

CKPT = os.path.join(os.path.dirname(__file__), "best_model_new.pth")
GALLERY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "gallery_resnet50.pkl")

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_embedding(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        # take features before fc head (avgpool output)
        x = model.conv1(img); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x); x = torch.flatten(x, 1)
        emb = F.normalize(x)
    return emb.squeeze(0).numpy()

def register(model, name, img_path):
    os.makedirs(os.path.dirname(GALLERY), exist_ok=True)
    gallery = {"embeddings": [], "names": []}
    if os.path.exists(GALLERY):
        with open(GALLERY, "rb") as f:
            gallery = pickle.load(f)
    emb = get_embedding(model, img_path)
    gallery["embeddings"].append(emb)
    gallery["names"].append(name)
    with open(GALLERY, "wb") as f:
        pickle.dump(gallery, f)
    print(f"Registered {name}")

def register_from_dir(model, person_dir):
    name = os.path.basename(person_dir.rstrip("/\\"))
    gallery = {"embeddings": [], "names": []}
    if os.path.exists(GALLERY):
        with open(GALLERY, "rb") as f:
            gallery = pickle.load(f)
    count = 0
    for f in os.listdir(person_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            emb = get_embedding(model, os.path.join(person_dir, f))
            gallery["embeddings"].append(emb)
            gallery["names"].append(name)
            count += 1
    with open(GALLERY, "wb") as f:
        pickle.dump(gallery, f)
    print(f"Registered {name} with {count} images")

def recognize(model, img_path, threshold=0.35):
    if not os.path.exists(GALLERY):
        print("No gallery. Register first."); return None, 0
    with open(GALLERY, "rb") as f:
        gallery = pickle.load(f)
    if len(gallery["embeddings"]) == 0:
        print("Empty gallery."); return None, 0
    emb = get_embedding(model, img_path)
    best_name, best_sim = None, -1
    for e, n in zip(gallery["embeddings"], gallery["names"]):
        sim = float(np.dot(emb, e))
        if sim > best_sim:
            best_sim, best_name = sim, n
    if best_sim < threshold: return None, best_sim
    return best_name, best_sim

def live(model, threshold=0.35):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Press ESC to exit")
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite("_tmp.jpg", frame[y:y+h, x:x+w])
            name, sim = recognize(model, "_tmp.jpg", threshold)
            label = f"{name} ({sim:.2f})" if name else "Unknown"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("ResNet50 Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()
    if os.path.exists("_tmp.jpg"): os.remove("_tmp.jpg")

if __name__ == "__main__":
    model = load_checkpoint(CKPT, num_classes=512)
    model.eval()
    print("Model loaded")

    if len(sys.argv) < 2:
        print("Usage: register <name> <img> | register --dir <dir> | recognize <img> | --live")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "register":
        if sys.argv[2] == "--dir":
            register_from_dir(model, sys.argv[3])
        else:
            register(model, sys.argv[2], sys.argv[3])
    elif cmd == "recognize":
        name, sim = recognize(model, sys.argv[2])
        if name: print(f"Recognized: {name} ({sim:.4f})")
        else: print(f"No match ({sim:.4f})")
    elif cmd == "--live":
        live(model, float(sys.argv[2]) if len(sys.argv) > 2 else 0.35)
