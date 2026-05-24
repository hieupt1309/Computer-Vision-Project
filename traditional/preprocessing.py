import os
import cv2
import numpy as np
import mediapipe as mp


# ==================================================
# PATH
# ==================================================

INPUT_DIR = (
    r"C:\Users\Admin\OneDrive\Desktop"
    r"\Computer Vision Project"
    r"\venv_traditional\raw_dataset"
)

OUTPUT_DIR = (
    r"C:\Users\Admin\OneDrive\Desktop"
    r"\Computer Vision Project"
    r"\venv_traditional\processed_dataset"
)


# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = 160
PADDING_RATIO = 0.20


# ==================================================
# MEDIAPIPE FACEMESH
# ==================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# ==================================================
# CLAHE
# ==================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# ==================================================
# ALIGN + CROP
# ==================================================

def align_and_crop_face(
    image,
    landmarks,
    output_size=160
):

    h, w = image.shape[:2]

    # ==========================================
    # EYE LANDMARKS
    # ==========================================

    left_eye = landmarks[33]
    right_eye = landmarks[263]

    lx = left_eye.x * w
    ly = left_eye.y * h

    rx = right_eye.x * w
    ry = right_eye.y * h

    # ==========================================
    # COMPUTE ROTATION ANGLE
    # ==========================================

    dx = rx - lx
    dy = ry - ly

    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    eye_center = (
        int((lx + rx) / 2),
        int((ly + ry) / 2)
    )

    # ==========================================
    # ROTATE IMAGE
    # ==========================================

    M = cv2.getRotationMatrix2D(
        eye_center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    # ==========================================
    # ROTATE LANDMARKS
    # ==========================================

    xs = []
    ys = []

    for lm in landmarks:

        px = lm.x * w
        py = lm.y * h

        nx = (
            M[0, 0] * px
            + M[0, 1] * py
            + M[0, 2]
        )

        ny = (
            M[1, 0] * px
            + M[1, 1] * py
            + M[1, 2]
        )

        xs.append(nx)
        ys.append(ny)

    # ==========================================
    # FACE BOX
    # ==========================================

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    face_w = x_max - x_min
    face_h = y_max - y_min

    pad_x = int(face_w * PADDING_RATIO)
    pad_y = int(face_h * PADDING_RATIO)

    x1 = max(
        0,
        int(x_min - pad_x)
    )

    y1 = max(
        0,
        int(y_min - pad_y)
    )

    x2 = min(
        w,
        int(x_max + pad_x)
    )

    y2 = min(
        h,
        int(y_max + pad_y)
    )

    face = rotated[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    # ==========================================
    # RESIZE
    # ==========================================

    face = cv2.resize(
        face,
        (output_size, output_size),
        interpolation=cv2.INTER_CUBIC
    )

    return face


# ==================================================
# PROCESS DATASET
# ==================================================

saved = 0
skipped = 0


# ==================================================
# PROCESS TRAIN + TEST
# ==================================================

for split in ["train", "test"]:

    input_split_dir = os.path.join(
        INPUT_DIR,
        split
    )

    output_split_dir = os.path.join(
        OUTPUT_DIR,
        split
    )

    os.makedirs(
        output_split_dir,
        exist_ok=True
    )

    print(f"\nPROCESSING {split.upper()} SET")

    # ==========================================
    # LOOP PERSONS
    # ==========================================

    for person_name in os.listdir(
        input_split_dir
    ):

        input_person_dir = os.path.join(
            input_split_dir,
            person_name
        )

        if not os.path.isdir(
            input_person_dir
        ):
            continue

        output_person_dir = os.path.join(
            output_split_dir,
            person_name
        )

        os.makedirs(
            output_person_dir,
            exist_ok=True
        )

        count = 0

        # ======================================
        # LOOP IMAGES
        # ======================================

        for img_name in os.listdir(
            input_person_dir
        ):

            img_path = os.path.join(
                input_person_dir,
                img_name
            )

            image = cv2.imread(
                img_path
            )

            if image is None:

                skipped += 1
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(
                rgb
            )

            # ==================================
            # NO FACE
            # ==================================

            if not results.multi_face_landmarks:

                skipped += 1
                continue

            landmarks = (
                results
                .multi_face_landmarks[0]
                .landmark
            )

            # ==================================
            # ALIGN + CROP
            # ==================================

            face = align_and_crop_face(
                image,
                landmarks,
                IMAGE_SIZE
            )

            if face is None:

                skipped += 1
                continue

            # ==================================
            # GRAYSCALE
            # ==================================

            gray = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

            # ==================================
            # LIGHT DENOISE
            # ==================================

            gray = cv2.GaussianBlur(
                gray,
                (3, 3),
                0
            )

            # ==================================
            # CLAHE
            # ==================================

            gray = clahe.apply(
                gray
            )

            # ==================================
            # SAVE
            # ==================================

            save_path = os.path.join(
                output_person_dir,
                f"{count}.jpg"
            )

            cv2.imwrite(
                save_path,
                gray
            )

            count += 1
            saved += 1

        print(
            f"{split}/{person_name}: "
            f"{count} processed"
        )


# ==================================================
# FINAL RESULT
# ==================================================

print("\n===================")
print("PREPROCESS DONE")
print("===================")

print("Saved:", saved)
print("Skipped:", skipped)