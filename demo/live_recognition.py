import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import argparse
import threading
import cv2
import numpy as np
from deepface import DeepFace

GALLERY_FILE = os.path.join(os.path.dirname(__file__), "gallery.pkl")
FACE_SIZE = 112


def load_gallery():
    if os.path.exists(GALLERY_FILE):
        with open(GALLERY_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_gallery(gallery):
    with open(GALLERY_FILE, "wb") as f:
        pickle.dump(gallery, f)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    results = []
    for (x, y, w, h) in faces:
        margin = int(min(w, h) * 0.15)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)
        results.append(((x, y, w, h), face))
    return results


def get_embedding(face_img):
    emb = DeepFace.represent(
        img_path=face_img,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False
    )
    vec = np.array(emb[0]["embedding"], dtype=np.float32)
    return vec / np.linalg.norm(vec)


def recognize(emb, gallery, threshold=0.30):
    best_name = "Unknown"
    best_sim = 0
    for name, ref_embs in gallery.items():
        for ref in ref_embs:
            sim = np.dot(emb, ref)
            if sim > best_sim:
                best_sim = sim
                best_name = name
    if best_sim < threshold:
        return "Unknown", best_sim
    return best_name, best_sim


def register(name, count=10, camera_id=0):
    gallery = load_gallery()
    embs = gallery.get(name, [])

    print("Warming up ArcFace model...")
    dummy = np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
    get_embedding(dummy)
    print("Model ready!\n")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    captured = 0
    print(f"Registering '{name}' — look at the camera")
    print("Press SPACE to capture | ESC to quit")

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        faces = detect_faces(frame)

        if faces:
            x, y, w, h = faces[0][0]
            cv2.rectangle(display, (display.shape[1] - x - w, y),
                          (display.shape[1] - x, y + h), (0, 255, 0), 2)

        cv2.putText(display, f"Capture {captured+1}/{count}  SPACE=capture  ESC=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Register", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32 and faces:
            print(f"  Capturing {captured+1}/{count}...")
            emb = get_embedding(faces[0][1])
            embs.append(emb)
            captured += 1

    cap.release()
    cv2.destroyAllWindows()

    if captured > 0:
        gallery[name] = embs
        save_gallery(gallery)
        print(f"Saved {captured} embeddings for '{name}'")
    else:
        print("No photos captured")


def recognize_webcam(camera_id=0):
    gallery = load_gallery()
    if not gallery:
        print("No registered faces. Run with --register first.")
        return

    names = list(gallery.keys())
    print(f"Registered: {', '.join(names)}")

    print("Warming up ArcFace model...")
    dummy = np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
    get_embedding(dummy)
    print("Model ready!\n")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    cv2.namedWindow("Recognition (ESC to exit)", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty("Recognition (ESC to exit)", cv2.WND_PROP_TOPMOST, 1)

    # shared state between threads
    latest_frame = [None]
    latest_faces = [[]]
    latest_result = ["..."]
    running = [True]
    lock = threading.Lock()

    def capture_loop():
        while running[0]:
            ret, frame = cap.read()
            if not ret:
                break
            with lock:
                latest_frame[0] = frame
                latest_faces[0] = detect_faces(frame)

    def recognition_loop():
        while running[0]:
            with lock:
                faces = latest_faces[0].copy() if latest_faces[0] else []
            if faces:
                face = faces[0][1]
                try:
                    emb = get_embedding(face)
                    name, conf = recognize(emb, gallery)
                    with lock:
                        latest_result[0] = f"{name} ({conf:.0%})"
                except Exception as e:
                    with lock:
                        latest_result[0] = "Error"
            threading.Event().wait(0.15)

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=recognition_loop, daemon=True).start()

    print("Webcam started — ESC to exit")

    while running[0]:
        with lock:
            frame = latest_frame[0]
            faces = latest_faces[0].copy() if latest_faces[0] else []
            label = latest_result[0]

        if frame is None:
            cv2.waitKey(10)
            continue

        display = cv2.flip(frame, 1)

        if faces:
            (x, y, w, h), _ = faces[0]
            mx = display.shape[1] - x - w
            cv2.rectangle(display, (mx, y), (mx + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, label, (mx, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Recognition (ESC to exit)", display)
        if cv2.waitKey(1) & 0xFF == 27:
            running[0] = False
            break

    cap.release()
    cv2.destroyAllWindows()


def list_registered():
    gallery = load_gallery()
    if gallery:
        print("Registered people:")
        for name, embs in gallery.items():
            print(f"  {name}: {len(embs)} embeddings")
    else:
        print("No one registered yet")


def main():
    parser = argparse.ArgumentParser(description="Live Face Recognition (no training needed)")
    parser.add_argument("--register", type=str, help="Register a new person (name)")
    parser.add_argument("--count", type=int, default=10, help="Photos to capture during registration")
    parser.add_argument("--webcam", action="store_true", help="Run live recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--list", action="store_true", help="List registered people")
    args = parser.parse_args()

    if args.list:
        list_registered()
    elif args.register:
        register(args.register, args.count, args.camera)
    elif args.webcam:
        recognize_webcam(args.camera)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
