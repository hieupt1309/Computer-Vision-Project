import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from model import FaceModel


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device(DEVICE)
    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_set = FaceDataset(TRAIN_DIR, train_transform)
    val_set = FaceDataset(VAL_DIR, val_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(train_set.class_to_idx)
    model = FaceModel(EMBEDDING_DIM, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_acc = 0
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(imgs, labels)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs, labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs, labels)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Loss: {total_loss:.4f}  Val Acc: {val_acc:.4f}  LR: {current_lr:.7f}")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pth"))

        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
            print(f"BEST MODEL SAVED ({best_acc:.4f})")
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print("EARLY STOPPING")
            break

    print(f"\nBest Val Acc: {best_acc:.4f}")
    with open(os.path.join(MODELS_DIR, "class_to_idx.json"), "w") as f:
        json.dump(train_set.class_to_idx, f, indent=2)
    print("Class mapping saved.")


if __name__ == "__main__":
    main()
