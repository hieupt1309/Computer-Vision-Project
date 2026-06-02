import os


class DataLoader:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.raw_dir = os.path.join(self.data_dir, "raw_dataset")
        self.processed_dir = os.path.join(self.data_dir, "processed_dataset")
        self.split_dir = os.path.join(self.data_dir, "dataset_split")
        self.train_dir = os.path.join(self.split_dir, "train")
        self.val_dir = os.path.join(self.split_dir, "val")
        self.result_dir = os.path.join(base_dir, "result")
        self.models_dir = os.path.join(base_dir, "models")

    def check_data_dir(self, d):
        return os.path.exists(d) and any(os.path.isdir(os.path.join(d, sub)) for sub in os.listdir(d))

    def setup_dataset(self):
        if self.check_data_dir(os.path.join(self.raw_dir, "train")) and self.check_data_dir(os.path.join(self.raw_dir, "test")):
            print("Dataset already exists.")
            return

        try:
            import kagglehub
            import shutil
        except ImportError:
            print("Install kagglehub: pip install kagglehub")
            return

        print("Downloading VGGFace2 dataset...")
        path = kagglehub.dataset_download("trunghieupham130906/vggface2")

        os.makedirs(self.raw_dir, exist_ok=True)
        for split in ["train", "test"]:
            src = os.path.join(path, "raw_dataset", split)
            if os.path.exists(src):
                shutil.copytree(src, os.path.join(self.raw_dir, split), dirs_exist_ok=True)

        print(f"Dataset ready at {self.raw_dir}")
