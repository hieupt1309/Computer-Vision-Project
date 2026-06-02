import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(PROJECT_DIR, "result")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "dataset_split", "train")
VAL_DIR = os.path.join(DATA_DIR, "dataset_split", "val")

IMG_SIZE = 112
EMBEDDING_DIM = 512
NUM_CLASSES = 480

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
