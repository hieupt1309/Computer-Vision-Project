import os, sys, pickle, cv2
import numpy as np
from deepface import DeepFace

GALLERY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "gallery.pkl")
BACKEND = "opencv"
MODEL = "ArcFace"

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def recognize(img_path, threshold=0.35):
    if not os.path.exists(GALLERY):
        print("No gallery. Register first."); return None, 0
    with open(GALLERY, "rb") as f:
        gallery = pickle.load(f)
    if len(gallery["embeddings"]) == 0:
        print("Empty gallery."); return None, 0

    try:
        emb = DeepFace.represent(img_path, model_name=MODEL, detector_backend=BACKEND, enforce_detection=False)
    except:
        print("No face detected"); return None, 0
    emb = np.array(emb[0]["embedding"])

    best_name, best_sim = None, -1
    for e, n in zip(gallery["embeddings"], gallery["names"]):
        sim = cosine_similarity(emb, e)
        if sim > best_sim:
            best_sim, best_name = sim, n
    if best_sim < threshold:
        return None, best_sim
    return best_name, best_sim

def live(threshold=0.35):
    cap = cv2.VideoCapture(0)
    print("Press ESC to exit")
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")\
                .detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite("_tmp.jpg", frame[y:y+h, x:x+w])
            name, sim = recognize("_tmp.jpg", threshold)
            label = f"{name} ({sim:.2f})" if name else "Unknown"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()
    if os.path.exists("_tmp.jpg"): os.remove("_tmp.jpg")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recognize.py <image_path>")
        print("       python recognize.py --live")
        sys.exit(1)
    if sys.argv[1] == "--live":
        live(float(sys.argv[2]) if len(sys.argv) > 2 else 0.35)
    else:
        name, sim = recognize(sys.argv[1])
        if name: print(f"Recognized: {name} ({sim:.4f})")
        else: print(f"No match ({sim:.4f})")
