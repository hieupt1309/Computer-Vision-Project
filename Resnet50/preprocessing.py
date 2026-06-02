import os
import cv2
import numpy as np
import mediapipe as mp
from dataloader import DataLoader 

# ==================================================
# PATH
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
dataloader = DataLoader(base_dir=BASE_DIR)
dataloader.setup_dataset()

INPUT_DIR = os.path.join(os.path.join(BASE_DIR, "data"), "raw_dataset")
OUTPUT_DIR = os.path.join(os.path.join(BASE_DIR, "data"), "processed_dataset")

# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = 112


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
# FACE ALIGNMENT
# ==================================================

def align_and_crop_face(
    image,
    landmarks,
    output_size=112
):

    h, w = image.shape[:2]

    # --------------------------------
    # EYE LANDMARKS
    # --------------------------------

    left_eye = landmarks[33]
    right_eye = landmarks[263]

    lx = left_eye.x * w
    ly = left_eye.y * h

    rx = right_eye.x * w
    ry = right_eye.y * h

    # --------------------------------
    # COMPUTE ANGLE
    # --------------------------------

    dx = rx - lx
    dy = ry - ly

    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    eye_center = (
        int((lx + rx) / 2),
        int((ly + ry) / 2)
    )

    # --------------------------------
    # ROTATE IMAGE
    # --------------------------------

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

    # --------------------------------
    # ROTATE LANDMARKS
    # --------------------------------

    rotated_points = []

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

        rotated_points.append(
            (nx, ny)
        )

    rotated_points = np.array(
        rotated_points
    )

    xs = rotated_points[:, 0]
    ys = rotated_points[:, 1]

    # --------------------------------
    # FACE BOX
    # --------------------------------

    x_min = int(xs.min())
    y_min = int(ys.min())

    x_max = int(xs.max())
    y_max = int(ys.max())

    face_w = x_max - x_min
    face_h = y_max - y_min

    # padding
    pad_x = int(face_w * 0.30)
    pad_y = int(face_h * 0.35)

    x1 = max(
        0,
        x_min - pad_x
    )

    y1 = max(
        0,
        y_min - pad_y
    )

    x2 = min(
        w,
        x_max + pad_x
    )

    y2 = min(
        h,
        y_max + pad_y
    )

    face = rotated[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    # --------------------------------
    # RESIZE
    # --------------------------------

    face = cv2.resize(
        face,
        (
            output_size,
            output_size
        ),
        interpolation=cv2.INTER_CUBIC
    )

    return face


# ==================================================
# PROCESS DATASET
# ==================================================

saved_count = 0
skipped_count = 0


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

    print(
        f"\nPROCESSING "
        f"{split.upper()} SET"
    )

    # =============================================
    # LOOP PERSONS
    # =============================================

    for person_name in sorted(
        os.listdir(input_split_dir)
    ):

        person_path = os.path.join(
            input_split_dir,
            person_name
        )

        if not os.path.isdir(
            person_path
        ):
            continue

        save_person_path = os.path.join(
            output_split_dir,
            person_name
        )

        os.makedirs(
            save_person_path,
            exist_ok=True
        )

        count = 0

        print(
            f"Processing: "
            f"{split}/{person_name}"
        )

        # =========================================
        # LOOP IMAGES
        # =========================================

        for img_name in os.listdir(
            person_path
        ):

            img_path = os.path.join(
                person_path,
                img_name
            )

            image = cv2.imread(
                img_path
            )

            if image is None:

                skipped_count += 1
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            results = (
                face_mesh.process(
                    rgb
                )
            )

            # =====================================
            # NO FACE
            # =====================================

            if not (
                results
                .multi_face_landmarks
            ):

                skipped_count += 1
                continue

            landmarks = (
                results
                .multi_face_landmarks[0]
                .landmark
            )

            # =====================================
            # ALIGN + CROP
            # =====================================

            face = (
                align_and_crop_face(
                    image,
                    landmarks,
                    IMAGE_SIZE
                )
            )

            if face is None:

                skipped_count += 1
                continue

            # =====================================
            # SAVE
            # =====================================

            save_img_path = os.path.join(
                save_person_path,
                f"{count}.jpg"
            )

            cv2.imwrite(
                save_img_path,
                face
            )

            count += 1
            saved_count += 1

        print(
            f"Saved: {count}"
        )


# ==================================================
# FINAL RESULT
# ==================================================

print("\n======================")
print("PREPROCESS DONE")
print("======================")

print(
    f"Saved: "
    f"{saved_count}"
)

print(
    f"Skipped: "
    f"{skipped_count}"
)