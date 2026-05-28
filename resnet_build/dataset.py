import os
import cv2

from torch.utils.data import Dataset


class FaceDataset(Dataset):

    def __init__(
        self,
        root,
        transform=None
    ):

        self.root = root
        self.transform = transform

        self.images = []
        self.labels = []

        self.class_to_idx = {}

        # ==================================
        # SUPPORTED IMAGE FORMAT
        # ==================================

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        )

        # ==================================
        # GET ALL CLASSES
        # ==================================

        classes = sorted([

            cls for cls in os.listdir(root)

            if os.path.isdir(
                os.path.join(root, cls)
            )

        ])

        # ==================================
        # BUILD DATASET
        # ==================================

        for idx, cls in enumerate(
            classes
        ):

            self.class_to_idx[
                cls
            ] = idx

            cls_path = os.path.join(
                root,
                cls
            )

            for img_name in os.listdir(
                cls_path
            ):

                # skip non-image files
                if not img_name.lower().endswith(
                    valid_extensions
                ):
                    continue

                img_path = os.path.join(
                    cls_path,
                    img_name
                )

                self.images.append(
                    img_path
                )

                self.labels.append(
                    idx
                )

        print(
            f"\nLoaded "
            f"{len(self.images)} images "
            f"from {len(classes)} identities"
        )

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        idx
    ):

        img_path = self.images[
            idx
        ]

        label = self.labels[
            idx
        ]

        # ==================================
        # READ IMAGE
        # ==================================

        img = cv2.imread(
            img_path
        )

        # handle broken image
        if img is None:

            raise ValueError(
                f"Cannot read image:\n"
                f"{img_path}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # ==================================
        # TRANSFORM
        # ==================================

        if self.transform:

            img = self.transform(
                img
            )

        return img, label