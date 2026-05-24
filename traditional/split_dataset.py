import os
import random
import shutil


# =====================================
# PATH
# =====================================

SOURCE_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_traditional\dataset"

OUTPUT_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_traditional\dataset_split"


# =====================================
# SETTINGS
# =====================================

TRAIN_RATIO = 0.7
SEED = 42

random.seed(SEED)


# =====================================
# CREATE DIR
# =====================================

train_dir = os.path.join(
    OUTPUT_DIR,
    "train"
)

test_dir = os.path.join(
    OUTPUT_DIR,
    "test"
)

os.makedirs(
    train_dir,
    exist_ok=True
)

os.makedirs(
    test_dir,
    exist_ok=True
)


# =====================================
# SPLIT
# =====================================

for person_name in os.listdir(
    SOURCE_DIR
):

    person_path = os.path.join(
        SOURCE_DIR,
        person_name
    )

    if not os.path.isdir(
        person_path
    ):
        continue

    images = os.listdir(
        person_path
    )

    if len(images) < 2:
        continue

    random.shuffle(
        images
    )

    n_train = int(
        len(images)
        * TRAIN_RATIO
    )

    train_imgs = images[
        :n_train
    ]

    test_imgs = images[
        n_train:
    ]

    train_person_dir = os.path.join(
        train_dir,
        person_name
    )

    test_person_dir = os.path.join(
        test_dir,
        person_name
    )

    os.makedirs(
        train_person_dir,
        exist_ok=True
    )

    os.makedirs(
        test_person_dir,
        exist_ok=True
    )

    # train
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

    # test
    for img_name in test_imgs:

        src = os.path.join(
            person_path,
            img_name
        )

        dst = os.path.join(
            test_person_dir,
            img_name
        )

        shutil.copy2(
            src,
            dst
        )

    print(
        f"{person_name}: "
        f"train={len(train_imgs)} "
        f"test={len(test_imgs)}"
    )


print("\nDONE SPLIT!")