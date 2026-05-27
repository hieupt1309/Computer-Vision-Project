import os
import argparse
import cv2
import numpy as np
import mediapipe as mp


def align_and_crop_face(image, landmarks, output_size=112):
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


def main():
    parser = argparse.ArgumentParser(description="Capture face images for training")
    parser.add_argument("--name", type=str, default="Me", help="Person identity name")
    parser.add_argument("--count", type=int, default=30, help="Number of photos to capture")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--out", type=str, default="captured_data", help="Output directory")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of photos to use as test set")
    args = parser.parse_args()

    train_dir = os.path.join(args.out, "train", args.name)
    test_dir = os.path.join(args.out, "test", args.name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return

    print(f"Capturing {args.count} photos of '{args.name}'")
    print("Press SPACE to capture | ESC to quit")
    print(f"Saving to: {out_dir}")
    print()

    captured = 0
    test_count = int(args.count * args.test_split)

    while captured < args.count:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_detected = False
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            for lm in landmarks:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(display, (cx, cy), 1, (0, 255, 0), -1)

        status = f"[{captured}/{args.count}] Face: {'YES' if face_detected else 'NO'}  | SPACE=capture  ESC=quit"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Capture Faces", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Capture cancelled")
            break
        elif key == 32 and face_detected:
            face = align_and_crop_face(frame, landmarks)
            if face is not None:
                is_test = captured < test_count
                save_dir = test_dir if is_test else train_dir
                path = os.path.join(save_dir, f"{captured:03d}.jpg")
                cv2.imwrite(path, face)
                captured += 1
                print(f"  Captured {captured}/{args.count} -> {'test/' if is_test else 'train/'}{args.name}/{captured:03d}.jpg")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    print(f"\nDone! Captured {captured} photos of '{args.name}'")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}")


if __name__ == "__main__":
    main()
