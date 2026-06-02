import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from model import FaceModel
from train import FaceDataset


def evaluate(checkpoint_path):
    device = torch.device(DEVICE)
    model = FaceModel(EMBEDDING_DIM, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_dataset = FaceDataset(VAL_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_embeds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            embeds = model.backbone(imgs)
            all_embeds.append(embeds.cpu())
            all_labels.append(labels)

    all_embeds = torch.cat(all_embeds)
    all_labels = torch.cat(all_labels)

    all_embeds = F.normalize(all_embeds, dim=1)

    gallery = {}
    for i in range(len(test_dataset)):
        lbl = test_dataset.labels[i]
        if lbl not in gallery:
            gallery[lbl] = all_embeds[i]

    gallery_ids = sorted(gallery.keys())
    gallery_embs = torch.stack([gallery[lb] for lb in gallery_ids])

    sim = torch.matmul(all_embeds, gallery_embs.t())
    preds = gallery_ids[sim.argmax(dim=1)]

    acc = accuracy_score(all_labels.numpy(), preds)
    print(f"Test Accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CHECKPOINT_DIR, "epoch_050.pth")
    evaluate(ckpt)
