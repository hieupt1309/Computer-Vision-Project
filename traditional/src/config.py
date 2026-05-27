import os
import numpy as np

# Base directory for the traditional pipeline (traditional/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def check_data_dir(d):
    return os.path.exists(d) and len([sub for sub in os.listdir(d) if os.path.isdir(os.path.join(d, sub))]) > 0

# If local directories are not found or empty, download and copy from kagglehub
if not (check_data_dir(GALLERY_DIR) and check_data_dir(PROBE_DIR)):
    try:
        import kagglehub
        import shutil
        print(f"Dataset not found locally. Downloading vggface2 dataset directly to {DATA_DIR}...")
        
        # Download (non-blocking, anonymous, and cached)
        kaggle_path = kagglehub.dataset_download("trunghieupham130906/vggface2")
        
        # Copy raw_dataset/train and raw_dataset/test directly into data/train and data/test
        src_train = os.path.join(kaggle_path, "raw_dataset", "train")
        src_test = os.path.join(kaggle_path, "raw_dataset", "test")
        
        print(f"Copying dataset files to local project data folder: {DATA_DIR}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        
        if os.path.exists(src_train):
            shutil.copytree(src_train, GALLERY_DIR, dirs_exist_ok=True)
        if os.path.exists(src_test):
            shutil.copytree(src_test, PROBE_DIR, dirs_exist_ok=True)
            
        print("Dataset setup completed successfully!")
    except Exception as e:
        print(f"Warning: Failed to set up dataset ({e}). Falling back to default paths.")

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
