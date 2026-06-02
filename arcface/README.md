# ArcFace — Face Recognition

Pretrained ArcFace via DeepFace (`register.py` / `recognize.py`) + PyTorch architecture reference (`arcface.py`).

## Files

| File | What it does |
|------|-------------|
| `arcface.py` | PyTorch implementation of ArcFace: ResNet50 backbone (Bottleneck blocks) + ArcFace additive angular margin head. Optionally loads ImageNet pretrained weights for the backbone. |
| `register.py` | Extract ArcFace embeddings via DeepFace and save to `demo/gallery.pkl`. |
| `recognize.py` | Match a face against the gallery via cosine similarity. Includes `--live` webcam mode. |

## Quick start

```bash
# Register a person from an image
python register.py "Name" path/to/photo.jpg

# Register all images in a folder (uses folder name as person name)
python register.py --dir path/to/name_folder/

# Recognize from an image
python recognize.py path/to/photo.jpg

# Live webcam recognition
python recognize.py --live
```

## Architecture (`arcface.py`)

```
Input (3×112×112)
  │
conv1 (7×7, 64, /2) → BN → ReLU → MaxPool (/2)
  │
layer1 ─ Bottleneck × 3  (64 → 256)
layer2 ─ Bottleneck × 4  (128 → 512)  stride 2
layer3 ─ Bottleneck × 6  (256 → 1024) stride 2
layer4 ─ Bottleneck × 3  (512 → 2048) stride 2
  │
AdaptiveAvgPool → FC (2048 → 512)
  │  L2-normalized embedding
  │
ArcFaceHead: cos(θ+m) · s  →  CrossEntropyLoss
```

- **Backbone**: 24.1M params, standard ResNet50 with 4-stage Bottleneck blocks
- **Embedding**: 512-d L2-normalized feature vector
- **ArcFace margin**: `m=0.5`, scale `s=64`
- `ArcFace(pretrained=True)` loads ImageNet weights into the backbone (ignoring mismatched fc layer)
