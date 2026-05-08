# 🕵️ Deepfake Detector

A deep learning system for detecting AI-generated / manipulated media using **EfficientNet-B0**, trained with **PyTorch Lightning** and served through a **Flask web app** with a modern drag-and-drop UI.

---

## ✨ Features

- **Binary classification** — Real vs. Deepfake with confidence score
- **Image & video support** — JPG, PNG, MP4, MOV
- **EfficientNet-B0 backbone** — lightweight yet accurate
- **Custom dataset loader** — supports mixed data sources with optional label overrides
- **Config-driven training** — all hyperparameters in `config.yaml`
- **Early stopping & LR scheduling** — smart training callbacks
- **Flask web interface** — sleek dark UI with live confidence bar and media preview

---

## 📁 Project Structure

```
deepfake-detector/
│
├── datasets/
│   └── hybrid_loader.py       # Custom PyTorch Dataset with multi-source support
│
├── lightning_modules/
│   └── detector.py            # PyTorch Lightning module (train/val/optimizer)
│
├── models/
│   └── best_model-v3.pt       # Trained model weights (not included in repo)
│
├── web-app1.py                # Flask inference server + embedded UI
├── main_trainer.py            # Training entry point
├── config.yaml                # Hyperparameters and dataset paths
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/moheezkiani01/deepfake-detector.git
cd deepfake-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **CUDA users:** Make sure your `torch` install matches your CUDA version.  
> Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the right install command.

### 3. Prepare your dataset

Your dataset directories should follow this structure:

```
C:/Datasets/your_dataset/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   └── ...
│   └── fake/
│       ├── image1.jpg
│       └── ...
└── validation/
    ├── real/
    └── fake/
```

Then update `config.yaml` with your actual paths:

```yaml
train_paths:
  - C:/Datasets/your_dataset/train

val_paths:
  - C:/Datasets/your_dataset/validation
```

---

## 🚀 Training

```bash
python main_trainer.py
```

The best model checkpoint will be saved to `models/best_model.ckpt`.

### Training configuration (`config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `lr` | `0.0001` | Learning rate |
| `batch_size` | `4` | Batch size |
| `num_epochs` | `1` | Max training epochs |
| `early_stopping_patience` | `3` | Epochs to wait before stopping |
| `scheduler_factor` | `0.5` | LR reduction factor on plateau |
| `scheduler_patience` | `2` | Epochs to wait before reducing LR |
| `monitor_metric` | `val_loss` | Metric tracked by callbacks |
| `save_top_k` | `1` | Number of best checkpoints to keep |

---

## 🌐 Running the Web App

> Make sure you have a trained model saved at `models/best_model-v3.pt`.

```bash
python web-app1.py
```

Then open your browser at **[http://localhost:5000](http://localhost:5000)**.

### How it works

1. Drag & drop or browse for an image (JPG/PNG) or video (MP4/MOV)
2. Click **Analyze File**
3. The app returns a **Real / Deepfake** verdict with a confidence percentage and a preview of the analyzed frame

> For video files, inference is performed on the **first frame**.

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Classifier head | `Dropout(0.4)` → `Linear(1280, 2)` |
| Loss | Cross-Entropy |
| Optimizer | Adam |
| Input size | 224 × 224 |
| Output classes | 2 (Real = 0, Fake = 1) |

---

## 📦 Dependencies

```
torch
torchvision
pytorch-lightning
flask
opencv-python
PyYAML
Pillow
numpy
scikit-learn
tensorboard
```

---

## 📝 Notes

- The `HybridDeepfakeDataset` supports loading from **multiple directories** at once and allows per-source label overrides, making it easy to combine datasets (e.g., FaceForensics++, Celeb-DF, custom data).
- Training automatically uses **GPU if available**, falling back to CPU.
- Logs are available via **TensorBoard**: `tensorboard --logdir lightning_logs/`

---

