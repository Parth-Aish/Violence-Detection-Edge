# Pose-Based Spatio-Temporal Violence Detection

This repository contains a train-to-deploy video classification pipeline. An exported YOLOv8-Pose ONNX model extracts 17 body keypoints per frame; a single-layer, 64-unit LSTM ONNX model classifies 30-frame sequences of normalized x/y coordinates (34 features per frame). The production backend uses ONNX Runtime and OpenCV only for inference; it does not install PyTorch or Ultralytics. The backend uses a default probability threshold of `0.75` and emits a positive violence decision only after three consecutive above-threshold inference results.

The project accepts browser webcam, RTSP/RTSPS, and MP4 sources. The browser captures webcam frames through `getUserMedia` and sends them over `/ws/stream`; cloud servers do not attempt to open a local camera device. Local ingestion defaults to 15 FPS with an asyncio queue bounded to 30 frames. The classifier uses a default probability threshold of `0.75` and requires three consecutive above-threshold inference results before raising a violence alert. Pose inference runs every second frame by default, with intermediate keypoints extrapolated from observed motion. Render pins numerical-library and ONNX Runtime threads to one, uses a 15 FPS ingestion cap, `POSE_STRIDE=2`, and `VIOLENCE_THRESHOLD=0.85`. WebSocket previews are resized to at most 480 pixels wide and encoded as JPEG at quality 60. Incident timestamps, probability, severity, FPS, and inference latency are logged to SQLite and JSONL.

## Data and model workflow

### 1. Install runtime dependencies

For API serving and ONNX inference, create a Python virtual environment and install the minimal runtime requirements:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Install training-only dependencies

Only needed when downloading data, extracting keypoints, or retraining the models. These do not belong in the Render runtime image:

```bash
python -m pip install -r training-requirements.txt
```

### 3. Configure Kaggle access and download the dataset

Authenticate using `kaggle auth login`, set `KAGGLE_API_TOKEN`, or configure `KAGGLE_USERNAME` and `KAGGLE_KEY` / the standard `~/.kaggle/kaggle.json` file. Keep all credential files out of version control.

```bash
python kaggle_pipeline.py
```

The downloader extracts `mohamedmustafa/real-life-violence-situations-dataset` under `data/raw/`. Dataset access is subject to Kaggle's access and use terms.

### 4. Extract pose sequences and train

The dataset preparation script expects class directories named `Violence` and `NonViolence` (or equivalent recognized class-folder names). Download the YOLOv8-Pose checkpoint as `yolov8n-pose.pt` in the project root; `.pt` files and all of `data/` are excluded from Git and Docker. The script samples 30 frames per video, runs YOLOv8-Pose concurrently, and writes float32 features and binary labels:

```bash
python dataset_prep.py --dataset-dir data/raw --output-dir data/processed --workers 2 --model yolov8n-pose.pt
python train_model.py --features data/processed/X_train.npy --labels data/processed/y_train.npy --onnx-out backend/violence_lstm.onnx
```

Export the pose checkpoint for deployment after preparing the sequences. The export lands in the project root; move it into the production backend directory:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='onnx', simplify=True, imgsz=320)"
python -c "from pathlib import Path; Path('yolov8n-pose.onnx').replace('backend/yolov8n-pose.onnx')"
```

This creates `backend/yolov8n-pose.onnx`. Training uses a stratified 80/20 split and prints held-out precision, recall, and F1 at the `> 0.75` probability threshold, plus reference metrics at `0.50`. The training report evaluates individual sequences and does not model the online three-consecutive-result debounce. Both ONNX files are required to start the inference backend and build the Docker image.

## Run locally

Start the backend from the `backend` directory after exporting the ONNX artifact:

```bash
cd backend
python main.py
```

The API listens on port 7860 by default for local development and starts idle until a source is selected. Render supplies its own port (configured as 10000). In a separate terminal, run the Next.js UI:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. Use the Webcam control for camera index zero, upload an MP4 with the Upload Video control, or provide an `rtsp://` / `rtsps://` URL and select Use RTSP. MP4 playback runs once at the configured ingestion rate. Uploads are limited to 100 MiB by default. Local ingestion defaults to and is capped at 15 FPS.

## WebSocket and REST API

`/ws/stream` accepts browser camera frames and returns annotated JSON frame events:

```json
{"type":"client_frame","data":"<base64 JPEG>"}
```

```json
{
  "type": "frame",
  "data": "<base64 JPEG>",
  "is_violence": false,
  "probability": 0.12,
  "severity": "NONE",
  "fps": 9.8,
  "inference_ms": 42.1,
  "latency_ms": 42.1,
  "timestamp": 1792685082.84
}
```

The browser waits for each annotated response before sending its next camera frame, preventing a backlog on slower cloud instances. If frames are queued, the server retains only the newest waiting frame.

Available endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend, ingestion, violence threshold, debounce, pose stride, and ONNX thread configuration |
| `GET` | `/api/incidents` | Recent SQLite incidents |
| `GET` | `/api/stats` | Severity counts |
| `POST` | `/api/source/webcam` (`/source/webcam` alias) | Stop server-side ingestion and prepare for browser webcam frames |
| `POST` | `/source/rtsp` | Switch source; JSON body `{"source":"rtsp://..."}` |
| `POST` | `/upload` | Upload an MP4 using multipart field `file` |
| `WS` | `/ws/stream` | Accept `client_frame` / `stop_server_stream` messages and return annotated predictions |

## Configuration

| Variable | Default | Description |
|---|---:|---|
| `STREAM_SOURCE` | `idle` | Initial webcam index, RTSP URL, local file, or `idle` |
| `FRAME_WIDTH` | `640` | Inference frame width |
| `FRAME_HEIGHT` | `480` | Inference frame height |
| `FRAME_FPS` | `15` | Ingestion and local-video pacing rate (maximum 15 FPS) |
| `POSE_STRIDE` | `2` | Run pose ONNX every N frames; intermediate poses use recent keypoint velocity |
| `POSE_MODEL_PATH` | `backend/yolov8n-pose.onnx` | Exported pose model |
| `ONNX_MODEL_PATH` | `backend/violence_lstm.onnx` | Trained LSTM ONNX model |
| `ONNX_INTRA_OP_THREADS` | `1` | ONNX Runtime CPU threads |
| `MAX_UPLOAD_BYTES` | `104857600` | Maximum uploaded video size |
| `JPEG_QUALITY` | `60` | WebSocket JPEG quality |
| `PORT` | `7860` | FastAPI listen port locally; Render supplies its port |
| `VIOLENCE_THRESHOLD` | `0.75` | Strict classifier decision threshold (`probability > threshold`); requires three consecutive triggers; Render Blueprint overrides to `0.85` |
| `ORT_THREADS` | `1` | ONNX Runtime intra-op threads per session |

## Evidence and limitations

The trained ONNX artifact was produced from 2,000 downloaded videos (1,000 per class), with a stratified 400-sequence test split. Raw, per-sequence measurements at thresholds `> 0.85`, `> 0.65`, and `0.50` yielded held-out precision/recall/F1 of 1.000/0.330/0.496, 0.944/0.670/0.784, and 0.847/0.775/0.809, respectively. These historical threshold-only results do not evaluate the current `> 0.75` threshold combined with the three-consecutive-result debounce. A stride-two simulation on that same split changed precision/recall/F1 at `> 0.75` from 0.952/0.500/0.656 to 0.971/0.495/0.656; at the Render threshold `> 0.85`, accuracy changed from 0.665 to 0.6475 and recall from 0.330 to 0.295, with precision remaining 1.000. This is a keypoint-array simulation, not end-to-end validation, and indicates a small measured degradation at the existing Render threshold. On one local 66-frame clip, measured detector processing averaged 31.98 ms/frame at stride one and 16.75 ms/frame at stride two. Neither result predicts Render performance: the single-clip test ran on an Intel Core i5-1135G7, while the reported 15.89 FPS baseline preceded stride optimization. Historical logs and synthetic benchmark measurements belong to the earlier velocity-heuristic prototype and are not LSTM results. See [paper/paper.md](paper/paper.md) for scope and limitations.

The implementation selects the highest-confidence person independently in each frame and represents missing poses with zero coordinates. Multiple-person identity switches, camera motion, occlusion, non-violent high-motion activity, and threshold calibration require testing on representative labeled footage before operational use.

## Render deployment

The root `render.yaml` defines a Docker web service using the included `Dockerfile`. It pins ONNX/BLAS/OpenMP thread counts to one, caps ingestion at 15 FPS, and defaults to pose stride two. Both model files must be present:

```text
backend/violence_lstm.onnx
backend/yolov8n-pose.onnx
```

Push the repository to GitHub, then in Render choose **New + → Blueprint** and connect the repository. Render reads `render.yaml`. The service binds to Render's `$PORT`, starts idle without a camera, and accepts uploaded MP4 or RTSP sources. The free instance has limited CPU and ephemeral storage, may sleep after inactivity, and loses local SQLite logs and uploaded files after restart or redeploy.

To test locally:

```bash
docker build -t violence-detection-edge .
docker run --rm -p 10000:10000 violence-detection-edge
```

## Research report

See [paper/paper.md](paper/paper.md) for the IEEE-style engineering report, including the distinction between historical prototype logs and unmeasured LSTM performance.
