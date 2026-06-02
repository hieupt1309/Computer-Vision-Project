import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "data", "dataset_split", "train") 

VAL_DIR = os.path.join(BASE_DIR, "data", "dataset_split", "val")

TEST_DIR = os.path.join(BASE_DIR, "data", "processed_dataset", "test")

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_efficientnet_b1.pth")

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
IMAGE_SIZE = 240
EARLY_STOPPING_PATIENCE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((260, 260)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.12)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)

NUM_CLASSES = len(train_dataset.classes)
print(f"Number of classes: {NUM_CLASSES}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# MODEL
# ============================================================

weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1

model = models.efficientnet_b1(
    weights=weights
)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

scaler = GradScaler('cuda')

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch(loader):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training")

    for images, labels in loop:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

        acc = 100 * correct / total

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{acc:.2f}%"
        )

    epoch_loss = total_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================
# EVALUATE FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    loss = total_loss / len(loader)
    acc = correct / total

    return loss, acc


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_acc = 0
best_model_weights = None
patience_counter = 0

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"EPOCH [{epoch+1}/{NUM_EPOCHS}]")
    print("=" * 60)

    # Unfreeze backbone after 5 epochs
    if epoch == 5:

        print("\nUnfreezing backbone...\n")

        for param in model.features.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

    train_loss, train_acc = train_one_epoch(
        train_loader
    )

    val_loss, val_acc = evaluate(
        val_loader
    )

    scheduler.step(val_acc)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_model_weights,
            SAVE_PATH
        )

        print("Best model saved!")

        patience_counter = 0

    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:

        print("\nEarly stopping triggered!")
        break

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        SAVE_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_acc = evaluate(
    test_loader
)

print("\n" + "=" * 60)
print("FINAL TEST RESULT")
print("=" * 60)
print(f"Top-1 Accuracy: {test_acc:.4f}")
print("=" * 60)