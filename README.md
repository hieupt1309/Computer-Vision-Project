# 1-to-N Face Identification

Face identification pipeline — register people from webcam or photos, then identify faces in images or live video.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-FF6F00?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv)

---

## Project structure

```
arcface/                        DeepFace ArcFace tools + architecture
  arcface.py                    PyTorch ArcFace model (ResNet50 + ArcFaceHead)
  register.py                   Register faces from image/webcam → gallery
  recognize.py                  Recognize faces from image/webcam
  README.md                     ArcFace module docs
  requirements.txt

resnet_build/                   Custom ResNet50 from scratch (metric learning)
  model.py                      ResNet50 with Bottleneck blocks
  dataset.py                    FaceDataset loader
  train.py                      Train embedding model + classifier
  extract_embeddings.py         Build gallery.pkl from trained model
  evaluate_identification.py    Top-1 / Top-5 via cosine similarity
  register.py                   Register/recognize/live with pretrained checkpoint

Resnet50/                       Fine-tuned ResNet50 (torchvision pretrained)
  model.py                      ResNet50 + FC head
  dataset.py                    FaceDataset loader
  train.py                      Training script
  preprocessing.py              MediaPipe face alignment + crop
  split_dataset.py              Train/val split
  evaluate_identification.py    Test set evaluation
  requirements.txt

efficientnet/                   EfficientNet-B1 transfer learning
  train_efficientnet.py         Training script

traditional/                    Hand-crafted features (no deep learning)
  train.py                      Feature extraction → PCA → KNN
  infer.py                      Inference
  requirements.txt

data/                           VGGFace2 dataset (auto-downloaded)
  raw/                          Original images
  processed/                    Haar-cropped faces
  dataset_split/                Train/val (85/15 per identity)

README.md
```

---

## Usage guides

### ArcFace (DeepFace)

```bash
# Register from webcam
python arcface/register.py register-live "You" 10

# Register from image
python arcface/register.py register "You" photo.jpg

# Recognize
python arcface/recognize.py photo.jpg
python arcface/recognize.py --live
```

### ResNet-Build (custom ResNet50 + pretrained checkpoint)

```bash
# Register from webcam
python resnet_build/register.py register-live "You" 10

# Register from image/folder
python resnet_build/register.py register "You" photo.jpg
python resnet_build/register.py register --dir path/to/photos/

# Recognize
python resnet_build/register.py recognize photo.jpg

# Live recognition
python resnet_build/register.py --live [threshold]
```

### ResNet50 fine-tuned (torchvision)

### Traditional CV

```bash
python traditional/train.py
python traditional/infer.py --image photo.jpg
```

---

## Pipeline (resnet_build/register.py)

```
Webcam frame
  │
  ├─ flip horizontally (mirror display)
  │
  └─ every 3rd frame ──→ Haar Cascade face detect
                            │
                            ▼
                         Crop face → 224×224 → normalize (ImageNet stats)
                            │
                            ▼
                         Custom ResNet50 backbone
                         (Bottleneck ×3,4,6,3 → 2048-d)
                            │
                            ▼
                         Embedding head
                         FC(2048→1024) → BN → ReLU → Dropout → FC(1024→512)
                            │
                         L2-normalize → 512-d embedding
                            │
                            ▼
                         Cosine similarity vs gallery embeddings
                            │
                         threshold 0.55 → "Name" or "Unknown"
                            │
                         Draw box + label on display
```

---

## Training from scratch

```bash
# 1. Download VGGFace2 + preprocess
pip install kagglehub
python arcface/setup_data.py

# 2. Train resnet_build
python resnet_build/train.py

# 3. Extract gallery embeddings
python resnet_build/extract_embeddings.py

# 4. Evaluate
python resnet_build/evaluate_identification.py
```

---

## Comparison

| Approach | Face Detection | Classifier | Training | Best for |
|----------|---------------|------------|:--------:|----------|
| ArcFace (arcface/) | Haar Cascade | DeepFace ArcFace + cosine sim | No | Quick demo, few photos |
| ResNet-Build (resnet_build/) | Haar Cascade | Custom embedding + cosine sim | Yes | Metric learning research |
| ResNet50 (Resnet50/) | MediaPipe | Fine-tuned torchvision ResNet50 | Yes | Large datasets, GPU |
| EfficientNet-B1 | — | Fine-tuned EfficientNet-B1 | Yes | Higher accuracy, GPU |
| Traditional CV (traditional/) | Haar Cascade | HOG/LBP/... + PCA + KNN | Yes | No GPU, interpretability |

---

## Dependencies

| Package | Used by |
|---------|---------|
| `deepface` | arcface/register, arcface/recognize, demo/live_recognition, demo/demo_arcface |
| `opencv-python` | All modules — face detection, webcam, image I/O |
| `numpy` | All modules |
| `torch` + `torchvision` | resnet_build, Resnet50, efficientnet, arcface/arcface.py |
| `scikit-learn` | arcface (LR), traditional (PCA/KNN) |
| `scikit-image` | traditional (HOG, LBP, GLCM) |
| `mediapipe` | Resnet50 preprocessing + demo |
| `Pillow` | Image loading |
| `tqdm` | Training progress bars |
| `kagglehub` | Optional: VGGFace2 download |
