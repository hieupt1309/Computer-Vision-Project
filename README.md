# 🧠 1-to-N Face Identification

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![DeepFace](https://img.shields.io/badge/DeepFace-4285F4)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-FF6F00?logo=pytorch)

**Zero-shot 1-to-N face identification** — register multiple people from webcam, then identify who appears in front of the camera. No training needed.

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
| `--list` | Show all registered people |
| `--camera 1` | Use a different camera |

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture a photo (during registration) |
| `ESC` | Exit |

---

## 📁 Project

```
├── demo/
│   ├── live_recognition.py      ⭐ Register + recognize (no training)
│   ├── demo_arcface.py           ArcFace + trained classifier
│   ├── demo_resnet50.py          MediaPipe + ResNet50
│   ├── demo_traditional.py       Haar cascade + hand-crafted features
│   ├── capture_faces.py          Dataset capture for training
│   └── check_camera.py           Debug which cameras are available
│
├── Resnet50/                     PyTorch ResNet50 pipeline
├── arcface/                      DeepFace ArcFace pipeline
└── traditional/                  Traditional CV pipeline
```

---

## How it works

1. **Register** — captures webcam photos → extracts ArcFace embeddings → saves to `gallery.pkl`
2. **Recognize** — detects faces with Haar Cascade → extracts embeddings → matches by cosine similarity

Runs on separate threads so the camera stays smooth regardless of recognition speed.

---

## User guides

### 🎯 ArcFace + LogisticRegression

Uses DeepFace ArcFace embeddings (pre-trained on millions of faces) with a LogisticRegression classifier trained on your dataset. More accurate than the zero-shot approach but requires a training step.

**Step 1 — Prepare dataset**

Organize face images into folders by person:

```
demo/arcface_data/train/
├── PersonA/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── PersonB/
│   ├── 001.jpg
│   └── ...
└── ...
```

You can use the capture tool:
```bash
python demo/capture_faces.py --name "PersonA"
python demo/capture_faces.py --name "PersonB"
```

**Step 2 — Train**

Extracts ArcFace embeddings from all images, then trains a LogisticRegression classifier:

```bash
python arcface/train_arcface.py
```

**Step 3 — Run demo**

```bash
# Single image
python demo/demo_arcface.py --image path/to/photo.jpg

# Webcam (uses RetinaFace detection, slower but more accurate)
python demo/demo_arcface.py --webcam
```

---

### 🧬 ResNet50 (PyTorch)

Fine-tunes a ResNet50 on your face dataset. Requires more data per person but gives a dedicated neural network classifier.

**Step 1 — Prepare dataset**

Similar structure as ArcFace. Use MediaPipe to preprocess (align + crop faces):

```bash
# First, organize raw images:
#   raw_dataset/train/PersonA/*.jpg
#   raw_dataset/test/PersonA/*.jpg

# Then preprocess: detects faces, aligns by eyes, crops to 112x112
python Resnet50/preprocessing.py

# Split training set into train/val (85/15)
python Resnet50/split_dataset.py
```

**Step 2 — Update paths**

Edit `Resnet50/train.py` to point `TRAIN_DIR` and `VAL_DIR` to your split dataset.

**Step 3 — Train**

```bash
python Resnet50/train.py
```

Saves best model to `Resnet50/models/best_model.pth` + class mapping to `Resnet50/models/class_to_idx.json`.

**Step 4 — Run demo**

```bash
# Single image
python demo/demo_resnet50.py --image path/to/photo.jpg

# Webcam (uses MediaPipe, smooth)
python demo/demo_resnet50.py --webcam
```

---

### 🏗️ Traditional CV

Hand-crafted features (HOG, LBP, color histograms, Gabor filters, GLCM) → PCA → KNN. No deep learning needed, fully interpretable. Can optionally auto-download the VGGFace2 dataset via kagglehub.

**Step 1 — Prepare dataset**

Place images in `traditional/data/train/<person>/` and `traditional/data/test/<person>/`:

```
traditional/data/train/
├── PersonA/
│   ├── img1.jpg
│   └── ...
├── PersonB/
│   └── ...
└── ...
```

Or let it auto-download VGGFace2 (requires kagglehub):
```bash
pip install kagglehub
# Then run train.py and it downloads on first run if data dir is empty
```

**Step 2 — Train**

Extracts HOG + LBP + color histogram + Gabor + GLCM features → PCA → KNN:

```bash
python traditional/train.py
```

Saves pipeline to `traditional/models/traditional_pipeline.pkl`.

**Step 3 — Run demo**

```bash
# Single image
python demo/demo_traditional.py --image path/to/photo.jpg

# Webcam
python demo/demo_traditional.py --webcam
```

---

### Comparison

| Approach | Face Detection | Classifier | Training | Best for |
|----------|---------------|------------|:--------:|----------|
| ⭐ Live Recognition | Haar Cascade | ArcFace + cosine similarity | ❌ | Quick demo, few photos |
| 🎯 ArcFace + LR | RetinaFace | ArcFace + LogisticRegression | ✅ | Accuracy, small datasets |
| 🧬 ResNet50 | MediaPipe | Fine-tuned ResNet50 | ✅ | Large datasets, GPU |
| 🏗️ Traditional CV | Haar Cascade | HOG/LBP/... + PCA + KNN | ✅ | No GPU, interpretability |
