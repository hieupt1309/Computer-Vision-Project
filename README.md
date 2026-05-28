# 🧠 1-to-N Face Identification

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![DeepFace](https://img.shields.io/badge/DeepFace-4285F4)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-FF6F00?logo=pytorch)

**Zero-shot 1-to-N face identification** — register multiple people from webcam, then identify who appears in front of the camera. No training needed.

Also includes trained alternatives: ArcFace + LogisticRegression, fine-tuned ResNet50, custom ResNet50 from scratch (metric learning), EfficientNet-B1, and traditional hand-crafted features.

---

## 🚀 Quick Start

```bash
# 1. Install
pip install deepface opencv-python numpy

# 2. Register yourself (press SPACE to capture ~10 photos)
python demo/live_recognition.py --register "You"

# 3. Add more people
python demo/live_recognition.py --register "Friend"

# 4. Run live recognition (mirrored view, always-on-top window)
python demo/live_recognition.py --webcam
```

### Commands

| Command | What it does |
|---------|-------------|
| `--register "Name"` | Capture face photos for a new person |
| `--register "Name" --count 5` | Capture only 5 photos |
| `--webcam` | Start live recognition |
| `--image photo.jpg` | Recognize faces in a single image |
| `--list` | Show all registered people |
| `--camera 1` | Use a different camera |

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture a photo (during registration) |
| `ESC` | Exit |

---

## 📁 Project Structure

```
├── demo/                           Demo scripts (ready to run)
│   ├── live_recognition.py    ⭐  Zero-shot: register + recognize, no training
│   ├── demo_arcface.py            ArcFace + LogisticRegression
│   ├── demo_resnet50.py           MediaPipe + fine-tuned ResNet50
│   ├── demo_traditional.py        Haar cascade + hand-crafted features
│   ├── check_camera.py            Debug available cameras
│   ├── gallery.pkl                Saved face embeddings (your registered data)
│   └── photo.jpg                  Sample image for testing
│
├── arcface/                        ArcFace embeddings + LogisticRegression
│   ├── train_arcface.py            Extract embeddings → train classifier
│   ├── evaluate_arcface.py         Test set evaluation
│   ├── evaluate_verification.py    1-to-1 verification evaluation
│   └── requirements.txt
│
├── Resnet50/                       Fine-tuned ResNet50 (PyTorch)
│   ├── train.py                    Training script
│   ├── model.py                    ResNet50 + custom FC head
│   ├── dataset.py                  FaceDataset loader
│   ├── preprocessing.py            MediaPipe face alignment + cropping
│   ├── split_dataset.py            Train/val split
│   ├── evaluate_identification.py  Test set evaluation
│   └── demo.py                     Standalone demo
│
├── resnet_build/                   ResNet50 from scratch (metric learning)
│   ├── model.py                    Custom ResNet with Bottleneck blocks
│   ├── dataset.py                  FaceDataset loader
│   ├── train.py                    Train embedding model + classifier
│   ├── extract_embeddings.py       Extract gallery embeddings after training
│   └── evaluate_identification.py  Top-1 / Top-5 via cosine similarity
│
├── efficientnet/                   EfficientNet-B1 (PyTorch)
│   └── train_efficientnet.py       Training script with mixed precision
│
├── traditional/                    Hand-crafted features (no deep learning)
│   ├── train.py                    Feature extraction → PCA → KNN
│   ├── infer.py                    Inference script
│   └── src/
│       ├── config.py               Paths and hyperparameters
│       ├── model.py                StandardScaler → PCA → KNN pipeline
│       └── utils.py                FacePreprocessor, FeatureExtractor, plotting
│
├── main.tex                        Beamer presentation (LaTeX)
└── README.md
```

---

## How it works

1. **Register** — captures webcam photos → extracts ArcFace embeddings → saves to `gallery.pkl`
2. **Recognize** — detects faces with Haar Cascade → extracts embeddings → matches by cosine similarity

Runs on separate threads so the camera stays smooth regardless of recognition speed.

---

## User guides

### 🎯 ArcFace + LogisticRegression

Uses DeepFace ArcFace embeddings (pre-trained on millions of faces) with a LogisticRegression classifier. More accurate than zero-shot but requires a training step.

**Step 1 — Prepare dataset**

Face images organized by person:

```
demo/arcface_data/train/
├── PersonA/  *.jpg
├── PersonB/  *.jpg
└── ...
```

**Step 2 — Train**

```bash
python arcface/train_arcface.py
```

**Step 3 — Run demo**

```bash
python demo/demo_arcface.py --image photo.jpg
python demo/demo_arcface.py --webcam
```

---

### 🧬 ResNet50 (fine-tuned)

Fine-tunes a torchvision ResNet50 on your face dataset. Uses MediaPipe for face alignment.

**Step 1 — Preprocess dataset**

```bash
# Organize raw images in raw_dataset/train/<person>/*.jpg
python Resnet50/preprocessing.py      # MediaPipe align + crop → 112x112
python Resnet50/split_dataset.py      # 85/15 train/val split
```

**Step 2 — Update paths & train**

Edit `TRAIN_DIR` / `VAL_DIR` in `Resnet50/train.py`, then:

```bash
python Resnet50/train.py
```

Saves `Resnet50/models/best_model.pth` + `class_to_idx.json`.

**Step 3 — Run demo**

```bash
python demo/demo_resnet50.py --image photo.jpg
python demo/demo_resnet50.py --webcam
```

---

### 🧬 ResNet50 from scratch (metric learning)

Custom ResNet50 implementation with Bottleneck blocks. Trains L2-normalized embeddings via a linear classifier, then evaluates by cosine similarity (Top-1 / Top-5).

```bash
python resnet_build/train.py                 # trains embedding model
python resnet_build/extract_embeddings.py    # builds gallery.pkl
python resnet_build/evaluate_identification.py
```

---

### ⚡ EfficientNet-B1

EfficientNet-B1 with transfer learning. Uses progressive unfreezing (backbone frozen for first 5 epochs). Mixed precision training with label smoothing.

```bash
python efficientnet/train_efficientnet.py
```

---

### 🏗️ Traditional CV

Hand-crafted features (HOG, LBP, color histograms, Gabor filters, GLCM) → PCA → KNN. No GPU needed.

**Step 1 — Prepare dataset**

```
traditional/data/train/<person>/*.jpg
traditional/data/test/<person>/*.jpg
```

Or auto-download VGGFace2 via kagglehub:
```bash
pip install kagglehub
python traditional/train.py   # downloads if data dir is empty
```

**Step 2 — Train & demo**

```bash
python traditional/train.py
python demo/demo_traditional.py --webcam
```

---

## Comparison

| Approach | Face Detection | Classifier | Training | Best for |
|----------|---------------|------------|:--------:|----------|
| ⭐ Live Recognition | Haar Cascade | ArcFace + cosine similarity | ❌ | Quick demo, few photos |
| 🎯 ArcFace + LR | RetinaFace | ArcFace + LogisticRegression | ✅ | Accuracy, small datasets |
| 🧬 ResNet50 (fine-tuned) | MediaPipe | Fine-tuned ResNet50 | ✅ | Large datasets, GPU |
| 🧬 ResNet50 (scratch) | — | Custom embedding + cosine sim | ✅ | Metric learning research |
| ⚡ EfficientNet-B1 | — | Fine-tuned EfficientNet-B1 | ✅ | Higher accuracy, GPU |
| 🏗️ Traditional CV | Haar Cascade | HOG/LBP/... + PCA + KNN | ✅ | No GPU, interpretability |

---

## 📦 Dependencies

| Package | Used by |
|---------|---------|
| `deepface` | Live recognition, ArcFace pipeline |
| `opencv-python` | All demos, face detection, webcam |
| `numpy` | All pipelines |
| `torch` + `torchvision` | ResNet50, EfficientNet, resnet_build |
| `mediapipe` | ResNet50 preprocessing, ResNet50 demo |
| `scikit-learn` | ArcFace (LR), Traditional (PCA/KNN) |
| `scikit-image` | Traditional (HOG, LBP, GLCM) |
| `matplotlib` | Traditional (visualization) |
| `tqdm` | Training progress bars |
| `kagglehub` | Optional: Traditional auto-download |
