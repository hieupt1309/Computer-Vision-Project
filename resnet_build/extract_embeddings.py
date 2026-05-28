import os
import pickle

import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import FaceDataset
from model import get_model


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ==========================================
# PATH
# ==========================================

TRAIN_DIR = (

     r"C:\Users\Admin"
    r"\OneDrive\Desktop"
    r"\Computer Vision Project"
    r"\venv_deeplearning"
    r"\dataset_split\train"

)

MODEL_PATH = (
    "models/best_model_new.pth"
)


# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485, 0.456, 0.406],

        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# DATASET
# ==========================================

dataset = FaceDataset(
    TRAIN_DIR,
    transform
)

loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=False,

    num_workers=0
)


# ==========================================
# MODEL
# ==========================================

model = get_model()

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(
    device
)

model.eval()


# ==========================================
# LABEL MAPPING
# ==========================================

idx_to_class = {

    v: k

    for k, v in
    dataset.class_to_idx.items()

}


# ==========================================
# STORE EMBEDDINGS
# ==========================================

embeddings = {}


print(
    "\nExtracting embeddings..."
)

with torch.no_grad():

    for imgs, labels in loader:

        imgs = imgs.to(
            device
        )

        outputs = model(
            imgs
        )

        outputs = (
            outputs
            .cpu()
            .numpy()
        )

        labels = (
            labels
            .cpu()
            .numpy()
        )

        for emb, label in zip(
            outputs,
            labels
        ):

            class_name = (
                idx_to_class[
                    label
                ]
            )

            if (
                class_name
                not in embeddings
            ):

                embeddings[
                    class_name
                ] = []

            embeddings[
                class_name
            ].append(
                emb
            )


# ==========================================
# AVERAGE EMBEDDING
# ==========================================

gallery = {}

for identity in embeddings:

    gallery[
        identity
    ] = np.mean(

        embeddings[
            identity
        ],

        axis=0
    )


# ==========================================
# SAVE
# ==========================================

os.makedirs(
    "embeddings",
    exist_ok=True
)

save_path = (
    "embeddings/gallery.pkl"
)

with open(
    save_path,
    "wb"
) as f:

    pickle.dump(
        gallery,
        f
    )


print(
    "\n======================"
)

print(
    "EMBEDDING EXTRACTED"
)

print(
    "======================"
)

print(
    f"Identities: "
    f"{len(gallery)}"
)

print(
    f"Saved to:"
)

print(
    save_path
)