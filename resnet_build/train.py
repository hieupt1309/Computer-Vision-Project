import os
import torch
import torch.nn as nn

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
    "\nDevice:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
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

VAL_DIR = (
    r"C:\Users\Admin"
    r"\OneDrive\Desktop"
    r"\Computer Vision Project"
    r"\venv_deeplearning"
    r"\dataset_split\val"
)


# ==========================================
# SETTINGS
# ==========================================

BATCH_SIZE = 32
EPOCHS = 100

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 10


# ==========================================
# TRANSFORM
# ==========================================

train_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (224, 224)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485, 0.456, 0.406],

        [0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

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

train_set = FaceDataset(
    TRAIN_DIR,
    train_transform
)

val_set = FaceDataset(
    VAL_DIR,
    val_transform
)


train_loader = DataLoader(

    train_set,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=True
)


val_loader = DataLoader(

    val_set,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)


# ==========================================
# MODEL
# ==========================================

model = get_model().to(
    device
)

num_classes = len(
    train_set.class_to_idx
)

print(
    "\nNumber of classes:",
    num_classes
)

# train-time classifier
classifier = nn.Linear(
    512,
    num_classes
).to(device)


# ==========================================
# LOSS
# ==========================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(

    list(model.parameters())
    +
    list(classifier.parameters()),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ==========================================
# LR SCHEDULER
# ==========================================

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS
    )
)


# ==========================================
# MIXED PRECISION
# ==========================================

scaler = torch.amp.GradScaler(
    "cuda"
)


# ==========================================
# TRAIN LOOP
# ==========================================

best_acc = 0
early_stop_counter = 0


for epoch in range(
    EPOCHS
):

    print(
        f"\nEpoch "
        f"[{epoch+1}/{EPOCHS}]"
    )

    # ======================================
    # TRAIN
    # ======================================

    model.train()
    classifier.train()

    total_loss = 0

    correct = 0
    total = 0

    for imgs, labels in train_loader:

        imgs = imgs.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        with torch.amp.autocast(
            "cuda"
        ):

            embeddings = model(
                imgs
            )

            logits = classifier(
                embeddings
            )

            loss = criterion(
                logits,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += (
            loss.item()
        )

        _, preds = torch.max(
            logits,
            1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(
            0
        )

    train_acc = (
        correct / total
    )

    # ======================================
    # VALIDATION
    # ======================================

    model.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for imgs, labels in val_loader:

            imgs = imgs.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            embeddings = model(
                imgs
            )

            logits = classifier(
                embeddings
            )

            _, preds = torch.max(
                logits,
                1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(
                0
            )

    val_acc = (
        correct / total
    )

    scheduler.step()

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    print(
        f"Loss: "
        f"{total_loss:.4f}"
    )

    print(
        f"Train Acc: "
        f"{train_acc:.4f}"
    )

    print(
        f"Val Acc: "
        f"{val_acc:.4f}"
    )

    print(
        f"LR: "
        f"{current_lr:.7f}"
    )

    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_acc > best_acc:

        best_acc = val_acc

        early_stop_counter = 0

        os.makedirs(
            "models",
            exist_ok=True
        )

        torch.save(

            model.state_dict(),

            "models/best_model_new.pth"
        )

        print(
            f"BEST MODEL SAVED "
            f"({best_acc:.4f})"
        )

    else:

        early_stop_counter += 1

    # ======================================
    # EARLY STOPPING
    # ======================================

    if (
        early_stop_counter
        >= PATIENCE
    ):

        print(
            "\nEARLY STOPPING"
        )

        break


print("\n======================")
print("TRAINING DONE")
print("======================")

print(
    f"Best Val Acc: "
    f"{best_acc:.4f}"
)