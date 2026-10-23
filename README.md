---
title: Pose-Based Violence Detection
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Pose-Based Spatio-Temporal Violence Detection

This repository contains a train-to-deploy video classification pipeline. YOLOv8-Pose extracts 17 body keypoints per frame; a single-layer, 64-unit PyTorch LSTM classifies 30-frame sequences of normalized x/y coordinates (34 features per frame). The model is exported to ONNX and the backend emits a positive violence decision when the output probability is greater than `0.85`.

The project accepts webcam, RTSP/RTSPS, and MP4 sources. It uses a one-frame producer-consumer queue, paces local video at 30 FPS by default, and logs incident timestamps, probability, severity, FPS, and inference latency to SQLite and JSONL.

## Data and model workflow

### 1. Install dependencies

From the repository root, create a Python virtual environment and install the requirements:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Configure Kaggle access and download the dataset

Authenticate using `kaggle auth login`, set `KAGGLE_API_TOKEN`, or configure `KAGGLE_USERNAME` and `KAGGLE_KEY` / the standard `~/.kaggle/kaggle.json` file. Keep all credential files out of version control.

```bash
python kaggle_pipeline.py
```

The downloader extracts `mohamedmustafa/real-life-violence-situations-dataset` under `data/raw/`. Dataset access is subject to Kaggle's access and use terms.

### 3. Extract pose sequences and train

The dataset preparation script expects class directories named `Violence` and `NonViolence` (or equivalent recognized class-folder names). It samples 30 frames per video, runs YOLOv8-Pose concurrently, and writes float32 features and binary labels:

```bash
python dataset_prep.py --dataset-dir data/raw --output-dir data/processed --workers 2
python train_model.py --features data/processed/X_train.npy --labels data/processed/y_train.npy --onnx-out backend/violence_lstm.onnx
```

Training uses a stratified 80/20 split and prints held-out precision, recall, and F1 at the same strict `> 0.85` decision threshold used by the online backend, plus reference metrics at `0.50`. Do not treat benchmark metrics as meaningful until the dataset split, class distribution, and model have been reviewed. The trained ONNX file is required to start the inference backend and build the Docker image.

## Run locally

Start the backend from the `backend` directory after exporting the ONNX artifact:

```bash
cd backend
python main.py
```

The API listens on port 7860 by default. In a separate terminal, run the Next.js UI:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. Use the Webcam control for camera index zero, upload an MP4 with the Upload Video control, or provide an `rtsp://` / `rtsps://` URL and select Use RTSP. MP4 playback runs once at the configured ingestion rate. Uploads are limited to 500 MiB by default.

## WebSocket and REST API

`/ws/stream` broadcasts JSON frame events:

```json
{
  "type": "frame",
  "data": "<base64 JPEG>",
  "is_violence": false,
  "probability": 0.12,
  "severity": "NONE",
  "fps": 29.4,
  "inference_ms": 42.1,
  "timestamp": 1792685082.84
}
```

Available endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend and ingestion status |
| `GET` | `/api/incidents` | Recent SQLite incidents |
| `GET` | `/api/stats` | Severity counts |
| `POST` | `/source/webcam` | Switch to local webcam index zero |
| `POST` | `/source/rtsp` | Switch source; JSON body `{"source":"rtsp://..."}` |
| `POST` | `/upload` | Upload an MP4 using multipart field `file` |
| `WS` | `/ws/stream` | Annotated frames, probability, and violence flag |

## Configuration

| Variable | Default | Description |
|---|---:|---|
| `STREAM_SOURCE` | `0` | Initial webcam index, RTSP URL, or local file |
| `FRAME_WIDTH` | `640` | Inference frame width |
| `FRAME_HEIGHT` | `480` | Inference frame height |
| `FRAME_FPS` | `30` | Ingestion and local-video pacing rate |
| `POSE_MODEL_PATH` | `yolov8n-pose.pt` | Ultralytics pose model |
| `ONNX_MODEL_PATH` | `backend/violence_lstm.onnx` | Trained LSTM ONNX model |
| `ONNX_INTRA_OP_THREADS` | `1` | ONNX Runtime CPU threads |
| `MAX_UPLOAD_BYTES` | `524288000` | Maximum uploaded video size |
| `JPEG_QUALITY` | `75` | WebSocket JPEG quality |
| `PORT` | `7860` | FastAPI listen port |

## Evidence and limitations

The trained ONNX artifact was produced from 2,000 downloaded videos (1,000 per class), with a stratified 400-sequence test split. At the backend's `> 0.85` threshold, held-out precision/recall/F1 were 1.000/0.330/0.496; at `0.50`, they were 0.847/0.775/0.809. A single offline 66-frame clip ran at 15.89 FPS on an Intel Core i5-1135G7; this is not a live-site or broad performance benchmark. The checked-in `backend/data/benchmark_summary.json` and 22 older `incidents.jsonl` rows belong to the earlier velocity-heuristic prototype and are not LSTM results. See [paper.md](paper.md) for scope and limitations.

The implementation selects the highest-confidence person independently in each frame and represents missing poses with zero coordinates. Multiple-person identity switches, camera motion, occlusion, non-violent high-motion activity, and threshold calibration require testing on representative labeled footage before operational use.

## Hugging Face Spaces deployment

The root `Dockerfile` builds the static frontend and backend image for port 7860. Train and place `backend/violence_lstm.onnx` in the repository first. Then configure `HF_TOKEN` in your shell and deploy:

```bash
HF_TOKEN=hf_... ./deploy.sh username/space-name
```

On PowerShell, set `$env:HF_TOKEN` before invoking the script from Git Bash or WSL. The deployment helper creates or updates a Docker Space and excludes Kaggle credentials, downloaded datasets, NumPy training arrays, and local pose checkpoints.

## Research report

See [paper.md](paper.md) for the IEEE-style engineering report, including the distinction between historical prototype logs and unmeasured LSTM performance.
