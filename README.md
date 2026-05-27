# Computer Vision Project — Face Recognition

Three face recognition approaches: ResNet50 (PyTorch), ArcFace (DeepFace), and Traditional CV (hand-crafted features).

## Project Structure

```
├── Resnet50/              # Deep learning — PyTorch ResNet50
│   ├── train.py           # Train classifier
│   ├── model.py           # ResNet50 model definition
│   ├── dataset.py         # FaceDataset loader
│   ├── preprocessing.py   # MediaPipe face alignment
│   ├── split_dataset.py   # Train/val split
│   └── evaluate_identification.py
├── arcface/               # DeepFace ArcFace embeddings
│   ├── train_arcface.py   # Extract embeddings → train LogisticRegression
│   ├── evaluate_arcface.py
│   └── evaluate_verification.py
├── traditional/           # Hand-crafted features + PCA + KNN
│   ├── train.py
│   ├── infer.py
│   └── src/
│       ├── config.py
│       ├── model.py       # StandardScaler → PCA → KNN pipeline
│       └── utils.py       # FacePreprocessor, FeatureExtractor, plotting
└── demo/                  # Reusable demo scripts (all three models)
    ├── demo_resnet50.py   # MediaPipe face detection + ResNet50
    ├── demo_arcface.py    # DeepFace ArcFace + LogisticRegression
    ├── demo_traditional.py# Haar cascade + HOG/LBP/color/Gabor/GLCM
    └── capture_faces.py   # Webcam face capture for training data
```

## Quick Start (ArcFace — recommended)

ArcFace uses a pre-trained face recognition model so it works well with minimal data.

```bash
# 1. Install dependencies
pip install deepface opencv-python numpy scikit-learn tqdm

# 2. Capture face photos (press SPACE to capture, ESC to quit)
python demo/capture_faces.py --name "Me"
python demo/capture_faces.py --name "Friend"   # add more people for 1-to-N

# 3. Train classifier (embeddings + LogisticRegression)
python arcface/train_arcface.py

# 4. Run live webcam demo
python demo/demo_arcface.py --webcam
```

## Demos

All three demos support the same interface:

```
python demo/demo_resnet50.py --image path/to/photo.jpg   # single image
python demo/demo_resnet50.py --webcam                     # live webcam
```

| Model | Face Detection | Classifier |
|-------|---------------|------------|
| ResNet50 | MediaPipe FaceMesh | Fine-tuned ResNet50 |
| ArcFace | RetinaFace (DeepFace) | ArcFace embeddings + LogisticRegression |
| Traditional | Haar Cascade | HOG+LBP+Color+Gabor+GLCM → PCA → KNN |

## Training

Each model needs to be trained on a dataset organized as `train/<person>/<images>.jpg`:

```bash
python Resnet50/train.py          # produces models/best_model.pth
python arcface/train_arcface.py   # produces arcface_classifier.pkl
python traditional/train.py       # produces models/traditional_pipeline.pkl
```
