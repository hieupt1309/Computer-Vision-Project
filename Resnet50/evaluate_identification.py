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

MODEL_PATH = "best_model_new.pth"

TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset_split", "val")


# ==================================================
# TRANSFORM (MUST SAME AS TRAIN/VAL)
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
# MODEL
# ==================================================

model = get_model(num_classes)

state = torch.load(
    MODEL_PATH,
    map_location=device
)
new_state = {}
for k, v in state.items():
    key = k.replace("embedding", "fc") if k.startswith("embedding") else k
    new_state[key] = v
model.load_state_dict(new_state)

model = model.to(device)
model.eval()


# ==================================================
# EVALUATION
# ==================================================

correct = 0
total = 0

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)


# ==================================================
# RESULT
# ==================================================

acc = correct / total

print("\n========================")
print("IDENTIFICATION RESULT")
print("========================")
print(f"Accuracy: {acc:.4f}")
print(f"Correct: {correct}")
print(f"Total: {total}")
print("========================")