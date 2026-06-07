import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=512):

    model = models.resnet50(
        weights="IMAGENET1K_V1"
    )

    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes)
    )

    return model


def load_checkpoint(path, num_classes=512, map_location="cpu"):
    model = get_model(num_classes)
    state = torch.load(path, map_location=map_location)
    new_state = {}
    for k, v in state.items():
        key = k.replace("embedding", "fc") if k.startswith("embedding") else k
        new_state[key] = v
    model.load_state_dict(new_state)
    return model