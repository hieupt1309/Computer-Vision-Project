import os
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 112


def get_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(2048, num_classes)
    )
    return model


def align_and_crop_face(image, landmarks, output_size=IMAGE_SIZE):
    h, w = image.shape[:2]

    left_eye = landmarks[33]
    right_eye = landmarks[263]

    lx = left_eye.x * w
    ly = left_eye.y * h
    rx = right_eye.x * w
    ry = right_eye.y * h

    dx = rx - lx
    dy = ry - ly
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = (int((lx + rx) / 2), int((ly + ry) / 2))
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    rotated_points = []
    for lm in landmarks:
        px = lm.x * w
        py = lm.y * h
        nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        rotated_points.append((nx, ny))
    rotated_points = np.array(rotated_points)

    xs = rotated_points[:, 0]
    ys = rotated_points[:, 1]
    x_min, y_min = int(xs.min()), int(ys.min())
    x_max, y_max = int(xs.max()), int(ys.max())

    face_w = x_max - x_min
    face_h = y_max - y_min

    pad_x = int(face_w * 0.30)
    pad_y = int(face_h * 0.35)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    face = rotated[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.resize(face, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return face


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_idx_to_class(mapping_path):
    with open(mapping_path, "r") as f:
        class_to_idx = json.load(f)
    return {str(v): k for k, v in class_to_idx.items()}


def load_idx_to_class_from_dir(data_dir):
    classes = sorted(os.listdir(data_dir))
    return {str(i): cls for i, cls in enumerate(classes)}


def predict(model, face_tensor, idx_to_class):
    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        outputs = model(face_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        name = idx_to_class.get(str(pred.item()), f"unknown_{pred.item()}")
        return name, conf.item()


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)


def demo_image(image_path, model, idx_to_class, face_mesh):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected")
        cv2.imshow("No face detected", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    landmarks = results.multi_face_landmarks[0].landmark
    face = align_and_crop_face(image, landmarks)
    if face is None:
        print("Face alignment failed")
        return

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_tensor = transform(face_rgb).unsqueeze(0)
    name, conf = predict(model, face_tensor, idx_to_class)

    print(f"Prediction: {name}  (confidence: {conf:.2%})")

    draw_landmarks(image, landmarks)
    cv2.putText(image, f"{name} ({conf:.2%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_webcam(model, idx_to_class, face_mesh, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print("Webcam demo started. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            face = align_and_crop_face(frame, landmarks)
            if face is not None:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_tensor = transform(face_rgb).unsqueeze(0)
                name, conf = predict(model, face_tensor, idx_to_class)
                label = f"{name} ({conf:.2%})"
            else:
                label = "Alignment failed"
        else:
            label = "No face"

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if results.multi_face_landmarks:
            draw_landmarks(frame, results.multi_face_landmarks[0].landmark)

        cv2.imshow("ResNet50 Demo (ESC to exit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ResNet50 Face Recognition Demo")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--webcam", action="store_true", help="Run real-time webcam demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--model", type=str, default=os.path.join("Resnet50", "models", "best_model.pth"),
                        help="Path to model weights")
    parser.add_argument("--mapping", type=str, default=os.path.join("Resnet50", "models", "class_to_idx.json"),
                        help="Path to class mapping JSON")
    parser.add_argument("--data-dir", type=str,
                        help="Dataset directory to infer class names (fallback if no mapping)")
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.print_help()
        return

    if os.path.exists(args.mapping):
        idx_to_class = load_idx_to_class(args.mapping)
    elif args.data_dir:
        idx_to_class = load_idx_to_class_from_dir(args.data_dir)
    else:
        print("Error: Need either --mapping or --data-dir to resolve class names")
        return

    num_classes = len(idx_to_class)
    print(f"Device: {device}")
    print(f"Loaded {num_classes} classes")

    model = load_model(args.model, num_classes)
    print("Model loaded successfully")

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=args.image is not None,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        if args.image:
            demo_image(args.image, model, idx_to_class, face_mesh)
        else:
            demo_webcam(model, idx_to_class, face_mesh, args.camera)


if __name__ == "__main__":
    main()
