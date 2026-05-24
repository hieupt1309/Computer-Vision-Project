import os
import cv2

from torch.utils.data import Dataset


class FaceDataset(
    Dataset
):

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

        classes = sorted(
            os.listdir(root)
        )

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

            if not os.path.isdir(
                cls_path
            ):
                continue

            for img_name in os.listdir(
                cls_path
            ):

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

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        idx
    ):

        img = cv2.imread(
            self.images[idx]
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        label = self.labels[
            idx
        ]

        if self.transform:

            img = self.transform(
                img
            )

        return img, label