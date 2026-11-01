# Deepfake Detector

An end-to-end computer vision system for detecting AI-generated and manipulated facial media using **EfficientNet-B0**, trained with **PyTorch Lightning**, and served through an interactive **Flask** inference web application.

---

## Overview

The proliferation of hyper-realistic generative models has made distinguishing authentic media from synthetic manipulations increasingly critical. **Deepfake Detector** provides a complete machine learning workflow—from multi-source dataset ingestion and configurable transfer learning to real-time inference via a modern web interface.

### Key Capabilities

- **Binary Classification:** Classifies input media as **Real** or **Deepfake** with calibrated softmax confidence scores.
- **Image & Video Ingestion:** Accepts standard image formats (JPEG, PNG) and video files (MP4, MOV).
- **EfficientNet-B0 Backbone:** Balances lightweight parameter size with high feature sensitivity across facial manipulation artifacts.
- **Config-Driven Training:** Centralized hyperparameter and dataset configuration via `config.yaml`.
- **PyTorch Lightning Pipeline:** Automated training harness with early stopping callbacks, learning rate scheduling on plateau, and TensorBoard logging.
- **Browser-Based Inference:** Single-service Flask application with responsive drag-and-drop file upload, live progress indicators, and visual preview.

---

## System Architecture

```
User Upload (Image / Video)
           │
           ▼
Flask Web App (app.py)
           │
  ┌────────┴──────────────────────────┐
  │                                   │
[Image Path]                     [Video Path]
RGB PIL Image                    OpenCV Frame Extraction (1st frame)
  │                                   │
  └────────┬──────────────────────────┘
           ▼
Torchvision Preprocessing Pipeline
(Resize 224x224, ToTensor, ImageNet Normalization)
           │
           ▼
EfficientNet-B0 Classifier Head (Linear: 1280 -> 2)
           │
           ▼
Softmax Probability Distribution
           │
  ┌────────┴──────────────────────────┐
  │                                   │
Prediction (Real / Deepfake)     Confidence Score (%)
```

---

## Repository Structure

```
DeepFake_Detector/
├── app.py               # Flask inference server with embedded responsive UI
├── detector.py          # PyTorch Lightning module (architecture, loss, optimizer, metrics)
├── hybrid_loader.py     # Custom PyTorch Dataset supporting multi-directory sources & label overrides
├── main_trainer.py      # Training entry point with PyTorch Lightning Trainer
├── config.yaml          # Declarative training hyperparameters & dataset path configuration
├── best_model-v3.pt     # Pretrained model weights (EfficientNet-B0 state dictionary)
├── requirements.txt     # Python runtime dependencies
└── README.md            # Project documentation
```

---

## Model Architecture

| Parameter | Specification |
|---|---|
| **Base Backbone** | EfficientNet-B0 (ImageNet pretrained weights) |
| **Classifier Head** | `Linear(in_features=1280, out_features=2)` |
| **Input Resolution** | 224 × 224 × 3 |
| **Loss Function** | Cross-Entropy Loss |
| **Optimizer** | Adam (`lr=0.0001` configurable via `config.yaml`) |
| **Target Classes** | `0`: Real, `1`: Deepfake |
| **Training Framework** | PyTorch Lightning |

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/moheezkiani01/DeepFake_Detector.git
cd DeepFake_Detector
```

### 2. Environment Setup

It is recommended to use a Python virtual environment (Python 3.9 – 3.11):

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

> **GPU Acceleration:** If using an NVIDIA GPU, install the CUDA-enabled PyTorch build matching your driver from [pytorch.org](https://pytorch.org/).

### 3. Prepare Model Weights for Inference

The Flask inference server expects model weights inside a `models/` directory:

```bash
# Windows PowerShell:
New-Item -ItemType Directory -Path "models" -Force
Copy-Item "best_model-v3.pt" -Destination "models/best_model-v3.pt"

# Linux / macOS:
mkdir -p models
cp best_model-v3.pt models/best_model-v3.pt
```

---

## Running the Web Application

Start the local Flask inference server:

```bash
python app.py
```

Access the interface in your browser:
```
http://localhost:5000
```

1. Drag and drop or browse for an image (`.jpg`, `.png`) or video (`.mp4`, `.mov`).
2. Click **Analyze File**.
3. View the prediction verdict (**Real** vs. **Deepfake**), model confidence percentage, and extracted frame preview.

---

## Training Pipeline

### 1. Dataset Organization

Organize your training and validation data into class subdirectories:

```
Dataset/
├── train/
│   ├── real/
│   └── fake/
└── validation/
    ├── real/
    └── fake/
```

### 2. Configuration (`config.yaml`)

Specify your dataset paths and hyperparameters:

```yaml
lr: 0.0001
batch_size: 4
num_epochs: 10

train_paths:
  - C:/Datasets/custom_deepfakes/train

val_paths:
  - C:/Datasets/custom_deepfakes/validation

early_stopping_patience: 3
scheduler_factor: 0.5
scheduler_patience: 2
monitor_metric: val_loss
log_every_n_steps: 1
save_top_k: 1
```

### 3. Execute Training

```bash
python main_trainer.py
```

Monitor training loss and metrics via TensorBoard:
```bash
tensorboard --logdir lightning_logs/
```

---

## Status & Technical Limitations

- **Video Processing:** Current video inference inspects the initial keyframe. Full-temporal video analysis across multiple sequence frames is planned for future iterations.
- **Model Scope:** Trained as a binary classification research prototype; real-world detection in adversarial scenarios benefits from ensembling with frequency-domain / facial-landmark analysis.
- **Data Dependency:** Generalization depends heavily on source dataset diversity across lighting, compression artifacts, and generative model architectures (e.g., GANs vs. Diffusion models).
