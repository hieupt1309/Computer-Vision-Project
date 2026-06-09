import os
import pickle
import argparse
import cv2
import numpy as np
from deepface import DeepFace


MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"


def extract_embedding(img_path_or_array):
    embedding = DeepFace.represent(
        img_path=img_path_or_array,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=False
    )
    emb = np.array(embedding[0]["embedding"], dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb


def load_classifier(model_path):
    with open(model_path, "rb") as f:
        clf, encoder = pickle.load(f)
    return clf, encoder


def predict(emb, clf, encoder):
    pred = clf.predict([emb])[0]
    name = encoder.inverse_transform([pred])[0]
    probs = clf.predict_proba([emb])[0]
    conf = float(probs[pred])
    return name, conf


def demo_image(image_path, clf, encoder):
    print(f"Processing {image_path}...")
    emb = extract_embedding(image_path)
    name, conf = predict(emb, clf, encoder)
    print(f"Prediction: {name}  (confidence: {conf:.2%})")

    image = cv2.imread(image_path)
    if image is None:
        return

    cv2.putText(image, f"{name} ({conf:.2%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("ArcFace Demo", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_webcam(clf, encoder, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print("Webcam demo started. Press ESC to exit.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run DeepFace every 3rd frame for speed
        if frame_count % 3 == 0:
            try:
                emb = extract_embedding(frame)
                name, conf = predict(emb, clf, encoder)
                label = f"{name} ({conf:.2%})"
            except Exception:
                label = "Detection failed"
        else:
            if "label" not in locals():
                label = "Detecting..."

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ArcFace Demo (ESC to exit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ArcFace Face Recognition Demo")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--webcam", action="store_true", help="Run real-time webcam demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--model", type=str,
                        default=os.path.join("arcface", "arcface_classifier.pkl"),
                        help="Path to pickled classifier")
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.print_help()
        return

    clf, encoder = load_classifier(args.model)
    print(f"Loaded classifier with {len(encoder.classes_)} identities")

    if args.image:
        demo_image(args.image, clf, encoder)
    else:
        demo_webcam(clf, encoder, args.camera)


if __name__ == "__main__":
    main()
