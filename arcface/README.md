# ArcFace Face Recognition

Plain PyTorch implementation of [ArcFace](https://arxiv.org/abs/1801.07698) (Additive Angular Margin Loss) for 1-to-N face identification. Minimal — no Lightning, no subpackages.

---

## Model Architecture

```
Input (3×112×112)
  │
  ▼
┌─────────────────────────────┐
│  ResNet50 Backbone          │
│                             │
│  conv1 (7×7, 64, /2)       │
│  BN + ReLU + MaxPool (/2)   │
│  Layer1 (Bottleneck×3, 256) │
│  Layer2 (Bottleneck×4, 512) │
│  Layer3 (Bottleneck×6,1024) │
│  Layer4 (Bottleneck×3,2048) │
│  AdaptiveAvgPool → FC(512)  │
└──────────┬──────────────────┘
           │ L2-normalized embedding
           ▼
┌─────────────────────────────┐
│  ArcFace Head               │
│  Cos(θ) + margin → s·Cos(θ+m)│
│  CrossEntropyLoss            │
└─────────────────────────────┘
```

**Total params**: 24.8M
- Backbone: standard ResNet50 with 4-stage Bottleneck blocks
- Embedding: 512-d L2-normalized feature vector
- ArcFace margin `m=0.5`, scale `s=64`
- Optimizer: AdamW (lr=1e-4, wd=1e-4) + ReduceLROnPlateau
- Mixed precision (AMP) when GPU available
- Label smoothing (0.05) + early stopping (patience=5)

---

## Data Layout

```
data/
├── raw_dataset/              # Original VGGFace2 images
│   ├── train/                #   person_X/*.jpg
│   └── test/
├── processed_dataset/        # Haar-cascade cropped → 112×112
│   ├── train/
│   └── test/
└── dataset_split/            # 85/15 per-identity split from train
    ├── train/                #   85% of each identity
    └── val/                  #   15% of each identity
```

---

## Usage

### 1. Setup

```bash
cd arcface
pip install -r requirements.txt
```

### 2. Full pipeline (download → preprocess → split)

```bash
pip install kagglehub
python datasetup.py
```

Or run each step individually:

```bash
python -c "from dataloader import DataLoader; DataLoader().setup_dataset()"
python preprocessing.py
python split_dataset.py
```

### 3. Train

```bash
python train.py
```

Checkpoints saved to `checkpoints/`, best model to `models/best_model.pth` + `class_to_idx.json`.

### 4. Evaluate

```bash
python evaluate.py
```

### 5. Face Auth CLI

```bash
python face_auth.py
```

---

## Files

| File | Purpose |
|------|---------|
| `dataloader.py` | Path definitions + auto-download VGGFace2 |
| `preprocessing.py` | Haar cascade face detection + crop → 112×112 |
| `split_dataset.py` | 85/15 per-identity train/val split |
| `datasetup.py` | Run all three steps sequentially |
| `model.py` | ResNet50 backbone + ArcFace head |
| `train.py` | Training with early stopping + best model |
| `evaluate.py` | Nearest-neighbor accuracy on val set |
| `face_auth.py` | Register / recognize CLI |
| `config.py` | Shared paths & hyperparameters |
