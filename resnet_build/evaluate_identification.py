import pickle

import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import (
    cosine_similarity
)

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

TEST_DIR = (
r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_deeplearning\processed_dataset\test"

)

MODEL_PATH = (
    "models/best_model_new.pth"
)

GALLERY_PATH = (
    "embeddings/gallery.pkl"
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
# LOAD TEST DATASET
# ==========================================

test_set = FaceDataset(
    TEST_DIR,
    transform
)

test_loader = DataLoader(

    test_set,

    batch_size=1,

    shuffle=False,

    num_workers=0
)


# ==========================================
# LABEL MAPPING
# ==========================================

idx_to_class = {

    v: k

    for k, v in
    test_set.class_to_idx.items()

}


# ==========================================
# LOAD MODEL
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
# LOAD GALLERY
# ==========================================

with open(
    GALLERY_PATH,
    "rb"
) as f:

    gallery = pickle.load(f)

print(
    f"\nGallery size:"
    f" {len(gallery)}"
)


# ==========================================
# EVALUATION
# ==========================================

correct_top1 = 0
correct_top5 = 0

total = 0


print(
    "\nEvaluating..."
)

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(
            device
        )

        # ===============================
        # QUERY EMBEDDING
        # ===============================

        query_embedding = model(
            imgs
        )

        query_embedding = (

            query_embedding
            .cpu()
            .numpy()

        )

        # ===============================
        # COSINE SIMILARITY
        # ===============================

        similarity_scores = {}

        for identity, gallery_emb in gallery.items():

            similarity = (
                cosine_similarity(

                    query_embedding,

                    gallery_emb.reshape(
                        1,
                        -1
                    )

                )[0][0]
            )

            similarity_scores[
                identity
            ] = similarity

        # ===============================
        # SORT
        # ===============================

        ranked_results = sorted(

            similarity_scores.items(),

            key=lambda x: x[1],

            reverse=True
        )

        # ===============================
        # TOP-1 / TOP-5
        # ===============================

        top1_prediction = (
            ranked_results[0][0]
        )

        top5_prediction = [

            item[0]

            for item in
            ranked_results[:5]

        ]

        true_identity = (
            idx_to_class[
                labels.item()
            ]
        )

        if (
            top1_prediction
            ==
            true_identity
        ):

            correct_top1 += 1

        if (
            true_identity
            in
            top5_prediction
        ):

            correct_top5 += 1

        total += 1


# ==========================================
# RESULT
# ==========================================

top1_acc = (
    correct_top1
    / total
)

top5_acc = (
    correct_top5
    / total
)

print(
    "\n========================"
)

print(
    "IDENTIFICATION RESULT"
)

print(
    "========================"
)

print(
    f"Top-1 Accuracy:"
    f" {top1_acc:.4f}"
)

print(
    f"Top-5 Accuracy:"
    f" {top5_acc:.4f}"
)

print(
    f"Correct Top-1:"
    f" {correct_top1}"
)

print(
    f"Correct Top-5:"
    f" {correct_top5}"
)

print(
    f"Total:"
    f" {total}"
)

print(
    "========================"
)