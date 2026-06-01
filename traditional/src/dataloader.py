import os
import numpy as np

# Base directory
class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.gallery_dir = os.path.join(self.data_dir, "train")
        self.probe_dir = os.path.join(self.data_dir, "test")
        self.result_dir = os.path.join(base_dir, "result")
        self.models_dir = os.path.join(base_dir, "models")
        self.model_path = os.path.join(self.models_dir, "traditional_pipeline.pkl")
    
    def check_data_dir(self, d):
        return os.path.exists(d) and len([sub for sub in os.listdir(d) if os.path.isdir(os.path.join(d, sub))]) > 0

    def setup_dataset(self):
        if not (self.check_data_dir(self.gallery_dir) and self.check_data_dir(self.probe_dir)):
            try:
                import kagglehub
                import shutil
                print(f"Dataset not found locally. Downloading vggface2 dataset directly to {self.data_dir}...")
                
                # Download (non-blocking, anonymous, and cached)
                kaggle_path = kagglehub.dataset_download("trunghieupham130906/vggface2")
                
                # Copy raw_dataset/train and raw_dataset/test directly into data/train and data/test
                src_train = os.path.join(kaggle_path, "raw_dataset", "train")
                src_test = os.path.join(kaggle_path, "raw_dataset", "test")
                
                print(f"Copying dataset files to local project data folder: {self.data_dir}...")
                os.makedirs(self.data_dir, exist_ok=True)
                
                if os.path.exists(src_train):
                    shutil.copytree(src_train, self.gallery_dir, dirs_exist_ok=True)
                if os.path.exists(src_test):
                    shutil.copytree(src_test, self.probe_dir, dirs_exist_ok=True)
                    
                print("Dataset setup completed successfully!")
            except Exception as e:
                print(f"Warning: Failed to set up dataset ({e}). Falling back to default paths.")
