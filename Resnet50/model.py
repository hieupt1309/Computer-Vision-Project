import torch.nn as nn
from torchvision import models


def get_model(num_classes):

    model = models.resnet50(
        weights="IMAGENET1K_V1"
    )

    model.fc = nn.Sequential(

        nn.Dropout(0.3),

        nn.Linear(
            2048,
            num_classes
        )
    )

    return model