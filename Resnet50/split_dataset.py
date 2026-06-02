import os
import random
import shutil


# ==================================================
# PATH
# ==================================================

INPUT_TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed_dataset", "train")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset_split")


# ==================================================
# SETTINGS
# ==================================================

TRAIN_RATIO = 0.85
VAL_RATIO = 0.15

SEED = 42

random.seed(SEED)


# ==================================================
# CREATE DIRS
# ==================================================

train_dir = os.path.join(
    OUTPUT_DIR,
    "train"
)

val_dir = os.path.join(
    OUTPUT_DIR,
    "val"
)

os.makedirs(
    train_dir,
    exist_ok=True
)

os.makedirs(
    val_dir,
    exist_ok=True
)


# ==================================================
# SPLIT TRAIN -> TRAIN + VAL
# ==================================================

for person_name in sorted(
    os.listdir(INPUT_TRAIN_DIR)
):

    person_path = os.path.join(
        INPUT_TRAIN_DIR,
        person_name
    )

    if not os.path.isdir(
        person_path
    ):
        continue

    images = os.listdir(
        person_path
    )

    random.shuffle(
        images
    )

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    train_imgs = images[
        :train_end
    ]

    val_imgs = images[
        train_end:
    ]

    # ==========================================
    # CREATE PERSON DIR
    # ==========================================

    train_person_dir = os.path.join(
        train_dir,
        person_name
    )

    val_person_dir = os.path.join(
        val_dir,
        person_name
    )

    os.makedirs(
        train_person_dir,
        exist_ok=True
    )

    os.makedirs(
        val_person_dir,
        exist_ok=True
    )

    # ==========================================
    # COPY TRAIN
    # ==========================================

    for img_name in train_imgs:

        src = os.path.join(
            person_path,
            img_name
        )

        dst = os.path.join(
            train_person_dir,
            img_name
        )

        shutil.copy2(
            src,
            dst
        )

    # ==========================================
    # COPY VAL
    # ==========================================

    for img_name in val_imgs:

        src = os.path.join(
            person_path,
            img_name
        )

        dst = os.path.join(
            val_person_dir,
            img_name
        )

        shutil.copy2(
            src,
            dst
        )

    print(
        f"{person_name}: "
        f"{len(train_imgs)} train | "
        f"{len(val_imgs)} val"
    )


print("\n====================")
print("SPLIT DONE")
print("====================")