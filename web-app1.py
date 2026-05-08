import os
import io
import base64
import mimetypes
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision import transforms
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# === Load Model ===
def load_model():
    model = efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("models/best_model-v3.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === Preprocessing ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Inference Logic ===
def predict_file(file_bytes, mime_type):
    preview_b64 = None

    if mime_type and mime_type.startswith("image"):
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        label = "Real" if pred.item() == 0 else "Deepfake"

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        preview_b64 = base64.b64encode(buf.getvalue()).decode()

        return label, round(conf.item() * 100, 2), preview_b64

    elif mime_type and mime_type.startswith("video"):
        tmp_path = "/tmp/upload_video"
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        os.remove(tmp_path)

        if not ret:
            return "Error reading video", 0, None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        label = "Real" if pred.item() == 0 else "Deepfake"

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        preview_b64 = base64.b64encode(buf.getvalue()).decode()

        return f"{label} (1st frame)", round(conf.item() * 100, 2), preview_b64

    return "Unsupported file type", 0, None


# === HTML Template ===
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepfake Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #111118;
    --border: #1e1e2e;
    --accent: #00e5ff;
    --accent2: #ff3c6e;
    --text: #e8e8f0;
    --muted: #5a5a7a;
    --real: #00ff88;
    --fake: #ff3c6e;
    --font-display: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-display);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 24px;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    top: -40%;
    left: -20%;
    width: 80%;
    height: 80%;
    background: radial-gradient(ellipse, rgba(0,229,255,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
  }
  body::after {
    content: '';
    position: fixed;
    bottom: -30%;
    right: -10%;
    width: 60%;
    height: 60%;
    background: radial-gradient(ellipse, rgba(255,60,110,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
  }

  .container {
    width: 100%;
    max-width: 760px;
    position: relative;
    z-index: 1;
  }

  header {
    text-align: center;
    margin-bottom: 52px;
  }

  .eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
    opacity: 0.8;
  }

  h1 {
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--text) 30%, var(--muted) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .subtitle {
    margin-top: 14px;
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--muted);
    font-weight: 300;
  }

  /* Drop Zone */
  .drop-zone {
    border: 1.5px dashed var(--border);
    border-radius: 16px;
    background: var(--surface);
    padding: 56px 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    overflow: hidden;
  }

  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(0,229,255,0.03);
  }

  .drop-zone input[type="file"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    width: 100%;
    height: 100%;
  }

  .drop-icon {
    width: 52px;
    height: 52px;
    margin: 0 auto 20px;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.02);
  }

  .drop-icon svg { width: 24px; height: 24px; stroke: var(--accent); fill: none; stroke-width: 1.5; }

  .drop-label {
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 8px;
  }

  .drop-hint {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
  }

  .filename-badge {
    margin-top: 16px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,229,255,0.07);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 999px;
    padding: 6px 14px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
    display: none;
  }

  /* Analyze Button */
  .btn-analyze {
    width: 100%;
    margin-top: 16px;
    padding: 16px;
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 12px;
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
  }

  .btn-analyze:hover { opacity: 0.88; transform: translateY(-1px); }
  .btn-analyze:active { transform: translateY(0); }
  .btn-analyze:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }

  /* Loading */
  .loader {
    display: none;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-top: 28px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
  }

  .loader.visible { display: flex; }

  .spinner {
    width: 18px;
    height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* Results */
  .results {
    margin-top: 32px;
    display: none;
    gap: 16px;
    flex-direction: column;
    animation: fadeUp 0.4s ease;
  }

  .results.visible { display: flex; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    display: flex;
    align-items: center;
    gap: 24px;
  }

  .verdict-indicator {
    width: 56px;
    height: 56px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    flex-shrink: 0;
  }

  .verdict-indicator.real { background: rgba(0,255,136,0.1); }
  .verdict-indicator.fake { background: rgba(255,60,110,0.1); }

  .verdict-text {
    flex: 1;
  }

  .verdict-label {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }

  .verdict-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
  }

  .verdict-value.real { color: var(--real); }
  .verdict-value.fake { color: var(--fake); }

  /* Confidence bar */
  .conf-section { }

  .conf-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
  }

  .conf-title {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .conf-pct {
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 500;
    color: var(--text);
  }

  .conf-bar-bg {
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
  }

  .conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
    width: 0%;
  }

  .conf-bar-fill.real { background: var(--real); }
  .conf-bar-fill.fake { background: var(--fake); }

  /* Preview */
  .preview-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
  }

  .preview-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .preview-card img {
    width: 100%;
    display: block;
    max-height: 360px;
    object-fit: contain;
    background: #07070e;
  }

  /* Error */
  .error-msg {
    display: none;
    margin-top: 16px;
    background: rgba(255,60,110,0.07);
    border: 1px solid rgba(255,60,110,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--fake);
  }

  .error-msg.visible { display: block; }

  footer {
    margin-top: 64px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    text-align: center;
    opacity: 0.5;
  }
</style>
</head>
<body>

<div class="container">
  <header>
    <div class="eyebrow">Neural Analysis System</div>
    <h1>Deepfake<br>Detector</h1>
    <p class="subtitle">EfficientNet-B0 · Binary Classification · Real-time</p>
  </header>

  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.mp4,.mov">
    <div class="drop-icon">
      <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
    </div>
    <div class="drop-label">Drop a file or click to browse</div>
    <div class="drop-hint">Supports JPG, PNG, MP4, MOV</div>
    <div class="filename-badge" id="filenameBadge">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      <span id="filenameText"></span>
    </div>
  </div>

  <button class="btn-analyze" id="analyzeBtn" disabled>Analyze File</button>

  <div class="loader" id="loader">
    <div class="spinner"></div>
    <span>Running inference…</span>
  </div>

  <div class="error-msg" id="errorMsg"></div>

  <div class="results" id="results">
    <div class="result-card">
      <div class="verdict-indicator" id="verdictIcon"></div>
      <div class="verdict-text">
        <div class="verdict-label">Prediction</div>
        <div class="verdict-value" id="verdictValue"></div>
      </div>
    </div>

    <div class="result-card conf-section">
      <div style="width:100%">
        <div class="conf-header">
          <span class="conf-title">Confidence</span>
          <span class="conf-pct" id="confPct"></span>
        </div>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" id="confBar"></div>
        </div>
      </div>
    </div>

    <div class="preview-card" id="previewCard" style="display:none">
      <div class="preview-header">Preview</div>
      <img id="previewImg" src="" alt="preview">
    </div>
  </div>

  <footer>Deepfake Detector · EfficientNet-B0 · MIT License</footer>
</div>

<script>
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const loader = document.getElementById('loader');
  const results = document.getElementById('results');
  const errorMsg = document.getElementById('errorMsg');
  const filenameBadge = document.getElementById('filenameBadge');
  const filenameText = document.getElementById('filenameText');

  let selectedFile = null;

  // Drag & Drop
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  function setFile(file) {
    selectedFile = file;
    filenameText.textContent = file.name;
    filenameBadge.style.display = 'inline-flex';
    analyzeBtn.disabled = false;
    results.classList.remove('visible');
    errorMsg.classList.remove('visible');
  }

  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loader.classList.add('visible');
    results.classList.remove('visible');
    errorMsg.classList.remove('visible');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();

      if (data.error) {
        errorMsg.textContent = data.error;
        errorMsg.classList.add('visible');
      } else {
        renderResult(data);
      }
    } catch (err) {
      errorMsg.textContent = 'Server error. Is the Flask app running?';
      errorMsg.classList.add('visible');
    } finally {
      loader.classList.remove('visible');
      analyzeBtn.disabled = false;
    }
  });

  function renderResult(data) {
    const isReal = data.label.toLowerCase().includes('real');
    const cls = isReal ? 'real' : 'fake';

    document.getElementById('verdictIcon').className = `verdict-indicator ${cls}`;
    document.getElementById('verdictIcon').textContent = isReal ? '✓' : '✕';
    document.getElementById('verdictValue').textContent = data.label;
    document.getElementById('verdictValue').className = `verdict-value ${cls}`;
    document.getElementById('confPct').textContent = data.confidence + '%';

    const bar = document.getElementById('confBar');
    bar.className = `conf-bar-fill ${cls}`;
    setTimeout(() => { bar.style.width = data.confidence + '%'; }, 50);

    if (data.preview) {
      document.getElementById('previewImg').src = 'data:image/jpeg;base64,' + data.preview;
      document.getElementById('previewCard').style.display = 'block';
    } else {
      document.getElementById('previewCard').style.display = 'none';
    }

    results.classList.add('visible');
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    mime_type, _ = mimetypes.guess_type(file.filename)
    file_bytes = file.read()

    label, confidence, preview_b64 = predict_file(file_bytes, mime_type)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "preview": preview_b64
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)