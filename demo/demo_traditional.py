import os
import sys
import argparse
import cv2
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from traditional.src.utils import FacePreprocessor, FeatureExtractor
from traditional.src.model import TraditionalPipeline


def load_pipeline(model_path):
    pipeline = TraditionalPipeline.load(model_path)
    label_names = getattr(pipeline, "label_names", None)
    return pipeline, label_names


def demo_image(image_path, pipeline, label_names, preprocessor, extractor):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    res = preprocessor.preprocess_image(img)
    if res is None:
        print("Face detection failed")
        return

    gray_face, hsv_face, bbox = res
    feat = extractor.extract_and_concatenate(gray_face, hsv_face)

    pred_label = pipeline.predict([feat])[0]
    pred_name = label_names[pred_label] if label_names else str(pred_label)

    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, pred_name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Prediction: {pred_name}")

    cv2.imshow("Traditional Demo", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_webcam(pipeline, label_names, preprocessor, extractor, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print("Webcam demo started. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = preprocessor.preprocess_image(frame)

        if res is not None:
            gray_face, hsv_face, bbox = res
            feat = extractor.extract_and_concatenate(gray_face, hsv_face)
            pred_label = pipeline.predict([feat])[0]
            pred_name = label_names[pred_label] if label_names else str(pred_label)

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = pred_name
        else:
            label = "No face"

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Traditional Demo (ESC to exit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Traditional CV Face Recognition Demo")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--webcam", action="store_true", help="Run real-time webcam demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--model", type=str,
                        default=os.path.join("traditional", "models", "traditional_pipeline.pkl"),
                        help="Path to pickled pipeline")
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.print_help()
        return

    pipeline, label_names = load_pipeline(args.model)
    print(f"Loaded pipeline. Identities: {len(label_names) if label_names else 'N/A'}")

    preprocessor = FacePreprocessor()
    extractor = FeatureExtractor()

    if args.image:
        demo_image(args.image, pipeline, label_names, preprocessor, extractor)
    else:
        demo_webcam(pipeline, label_names, preprocessor, extractor, args.camera)


if __name__ == "__main__":
    main()
