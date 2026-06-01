import os
import numpy as np
from .dataloader import DataLoader

# Base directory for the traditional pipeline (traditional/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataloader = DataLoader(base_dir=BASE_DIR)
dataloader.setup_dataset()

# Paths (Conforms to Kaggle dataset structure: train and test)
DATA_DIR = os.path.join(BASE_DIR, "data")
GALLERY_DIR = os.path.join(DATA_DIR, "train")
PROBE_DIR = os.path.join(DATA_DIR, "test")
RESULT_DIR = os.path.join(BASE_DIR, "result")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "traditional_pipeline.pkl")

# Dataset limits (set to None to run on the entire dataset)
MAX_IDENTITIES = 20
MAX_IMAGES_PER_IDENTITY = 30

# Preprocessing Settings
IMAGE_SIZE = 128

# Feature Extraction Settings
HOG_PARAMS = {
    "orientations": 8,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (1, 1),
    "visualize": True,
}

LBP_PARAMS = {
    "P": 8,
    "R": 1,
    "method": "uniform",
}

COLOR_PARAMS = {
    "h_bins": 16,
    "s_bins": 16,
}

GABOR_PARAMS = {
    "orientations": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    "lambdas": [3.0, 5.0, 7.0, 9.0],
    "sigma": 2.0,
    "gamma": 0.5,
}

GLCM_PARAMS = {
    "distances": [1, 2],
    "angles": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    "properties": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
}

# Model Settings
PCA_COMPONENTS = 0.95  # Retain 95% variance
KNN_NEIGHBORS = 5
