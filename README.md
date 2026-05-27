# 🧠 Face Recognition — Live Demo

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![DeepFace](https://img.shields.io/badge/DeepFace-4285F4)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-FF6F00?logo=pytorch)

**Zero-shot face recognition** — register your face from webcam, recognize instantly. No training needed.

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

## Other approaches

| Script | Face Detection | Recognition | Needs training |
|--------|---------------|-------------|:---:|
| `demo/live_recognition.py` ⭐ | Haar Cascade | ArcFace + cosine similarity | ❌ |
| `demo/demo_arcface.py` | RetinaFace | ArcFace + LogisticRegression | ✅ |
| `demo/demo_resnet50.py` | MediaPipe | Fine-tuned ResNet50 | ✅ |
| `demo/demo_traditional.py` | Haar Cascade | HOG/LBP/... + PCA + KNN | ✅ |

```bash
python demo/demo_resnet50.py --image photo.jpg
python demo/demo_arcface.py --webcam
```
