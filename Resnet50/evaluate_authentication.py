import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import FaceDataset
from model import get_model


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==================================================
# PATH
# ==================================================

MODEL_PATH = "models/best_model.pth"

TEST_DIR = r"C:\Users\Admin\OneDrive\Desktop\Computer Vision Project\venv_deeplearning\processed_dataset\test"


# ==================================================
# TRANSFORM
# ==================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==================================================
# DATASET
# ==================================================

test_set = FaceDataset(
    TEST_DIR,
    transform
)

test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


num_classes = len(test_set.class_to_idx)


# ==================================================
# MODEL (REMOVE CLASSIFIER HEAD FOR EMBEDDING)
# ==================================================

model = get_model(num_classes)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()


# ==================================================
# EXTRACT FEATURE (BEFORE FC LAYER)
# ==================================================

def extract_feature(x):

    # forward until penultimate layer
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    return x


# ==================================================
# COSINE SIMILARITY
# ==================================================

def cosine_sim(a, b):

    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)

    return np.dot(a, b)


# ==================================================
# BUILD GALLERY (REFERENCE FEATURES)
# ==================================================

gallery = {}

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(device)

        feats = extract_feature(imgs)

        feats = feats.cpu().numpy()

        for i in range(len(labels)):

            label = labels[i].item()

            if label not in gallery:
                gallery[label] = []

            gallery[label].append(feats[i])


# ==================================================
# AUTHENTICATION EVALUATION
# ==================================================

TP = 0
TN = 0
FP = 0
FN = 0

thresholds = np.linspace(0.3, 0.9, 50)

best_threshold = 0
best_acc = 0


# ==================================================
# SEARCH BEST THRESHOLD
# ==================================================

for THRESHOLD in thresholds:

    TP = TN = FP = FN = 0

    with torch.no_grad():

        for imgs, labels in test_loader:

            imgs = imgs.to(device)

            feats = extract_feature(imgs)
            feats = feats.cpu().numpy()

            for i in range(len(labels)):

                true_label = labels[i].item()
                query = feats[i]

                # pick positive sample
                pos_feats = gallery[true_label]
                pos_score = max([
                    cosine_sim(query, g)
                    for g in pos_feats
                ])

                if pos_score >= THRESHOLD:
                    TP += 1
                else:
                    FN += 1

                # pick negative random identity
                neg_label = np.random.choice(list(gallery.keys()))
                while neg_label == true_label:
                    neg_label = np.random.choice(list(gallery.keys()))

                neg_feats = gallery[neg_label]
                neg_score = max([
                    cosine_sim(query, g)
                    for g in neg_feats
                ])

                if neg_score >= THRESHOLD:
                    FP += 1
                else:
                    TN += 1

    acc = (TP + TN) / (TP + TN + FP + FN)

    if acc > best_acc:
        best_acc = acc
        best_threshold = THRESHOLD


# ==================================================
# FINAL EVALUATION
# ==================================================

TP = TN = FP = FN = 0

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(device)

        feats = extract_feature(imgs)
        feats = feats.cpu().numpy()

        for i in range(len(labels)):

            true_label = labels[i].item()
            query = feats[i]

            pos_feats = gallery[true_label]
            pos_score = max([
                cosine_sim(query, g)
                for g in pos_feats
            ])

            is_genuine = pos_score >= best_threshold

            if is_genuine:
                TP += 1
            else:
                FN += 1

            neg_label = np.random.choice(list(gallery.keys()))
            while neg_label == true_label:
                neg_label = np.random.choice(list(gallery.keys()))

            neg_feats = gallery[neg_label]
            neg_score = max([
                cosine_sim(query, g)
                for g in neg_feats
            ])

            is_impostor = neg_score >= best_threshold

            if is_impostor:
                FP += 1
            else:
                TN += 1


# ==================================================
# METRICS
# ==================================================

FAR = FP / (FP + TN + 1e-8)
FRR = FN / (TP + FN + 1e-8)
ACC = (TP + TN) / (TP + TN + FP + FN)


# ==================================================
# RESULT
# ==================================================

print("\n========================")
print("AUTHENTICATION RESULT")
print("========================")
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Accuracy: {ACC:.4f}")
print(f"FAR: {FAR:.4f}")
print(f"FRR: {FRR:.4f}")
print("========================")