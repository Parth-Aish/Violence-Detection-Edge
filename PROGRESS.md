# Violence Detection System — Build Progress

> **Project**: Edge-Optimized Spatio-Temporal Violence Detection for Legacy Surveillance Infrastructure  
> **Author**: L6 SWE / AI Researcher  
> **Last Updated**: 2026-10-23T11:33:53+05:30

## Requested Pose-LSTM Workflow

The items below track the requested Pose-LSTM implementation. Existing project components are preserved until the corresponding replacement or integration has been validated.

### Phase 1 — ML Training Pipeline (Kaggle + PyTorch)

- [x] `kaggle_pipeline.py` — Download `mohamedmustafa/real-life-violence-situations-dataset` through the Kaggle CLI/API with credential checks and safe overwrite handling
- [x] `dataset_prep.py` — Completed extraction of 2,000 videos into sorted 30-frame, normalized 34-feature sequences; writes `X_train.npy` and `y_train.npy`
- [x] `train_model.py` — Trained the input-34, hidden-64, one-layer LSTM; reported metrics at 0.85 and 0.50 thresholds and exported `violence_lstm.onnx`

### Phase 2 — Production Backend (FastAPI + Multiprocessing)

- [x] `backend/inference.py` — ONNX Runtime pose inference with rolling 30×34 coordinates and `> 0.85` fighting threshold; startup fails clearly if model artifact is absent
- [x] `backend/ingest.py` — Multiprocessing ingestion for webcam, RTSP, and local files; local playback uses dynamic sleep at the 30 FPS default
- [x] `backend/logger.py` — Persist prediction timestamps, severity, confidence, FPS, and inference latency to SQLite and JSONL
- [x] `backend/main.py` — FastAPI/WebSocket frame payload with `is_violence`, plus hot-swappable webcam/RTSP/MP4 sources
- [x] `backend/requirements.txt` — Delegates to the root training/runtime dependency manifest to avoid divergent installs

### Phase 3 — Frontend (Next.js + Tailwind)

- [x] `frontend/src/app/page.tsx` — Minimal dark/glass UI with Webcam, Upload Video, RTSP controls, and dynamic score/status badge

### Phase 4 — Research Paper

- [x] `paper.md` / `paper/paper.md` — Synced IEEE-style report documents actual held-out metrics and a clearly scoped offline runtime measurement, separate from legacy logs

### Phase 5 — Automated Deployment

- [x] `Dockerfile` — Multi-stage static frontend + ONNX/pose backend container on port 7860; requires trained ONNX artifact at build time
- [x] `requirements.txt` — FastAPI, OpenCV/Ultralytics, PyTorch, ONNX, Kaggle, scikit-learn, and deployment dependencies
- [x] `deploy.sh` / `deploy.py` — Hugging Face Docker Space creation and filtered repository upload; requires HF authentication and a trained ONNX artifact
- [x] `.gitignore` — Excludes Kaggle credentials, datasets, and arrays while allowing the trained backend ONNX model artifact
- [x] `README.md` — Documents Kaggle preparation, LSTM training, backend/frontend startup, endpoints, and deployment prerequisites

---

---

## Backend Core (Python / FastAPI)

- [x] `backend/ingest.py` — Producer-Consumer RTSP ingestion via FFmpeg + multiprocessing.Queue
- [x] `backend/inference.py` — YOLOv8-Pose extraction + temporal velocity heuristic classifier
- [x] `backend/logger.py` — JSON + SQLite incident logger (timestamps, bboxes, severity)
- [x] `backend/main.py` — FastAPI server, process orchestration, WebSocket `/ws/stream`
- [x] `backend/requirements.txt` — Python dependency manifest

## Real-Time Logging & Data Pipeline

- [x] `backend/data/incidents.db` — SQLite schema auto-created at runtime
- [x] `backend/data/incidents.jsonl` — Append-only JSON Lines log (created at runtime)

## Frontend UI (Next.js + Tailwind CSS)

- [x] `frontend/` — Next.js project scaffold
- [x] `frontend/src/app/page.tsx` — Main UI: dark mode, glassmorphism, live feed, incident log
- [x] `frontend/src/app/layout.tsx` — Root layout with metadata and fonts
- [x] `frontend/src/app/globals.css` — Tailwind + custom glassmorphism utilities
- [x] `frontend/tailwind.config.ts` — Using Tailwind v4 (config-free, CSS-first)

## Research Paper

- [x] `paper/paper.md` — IEEE-formatted research paper (full sections)
- [x] `paper/figures/` — Architecture diagrams (described in markdown/mermaid)

## Configuration

- [x] `.gitignore` — Project-level gitignore
- [x] `README.md` — Setup, usage, and deployment instructions
- [x] `Dockerfile` — Hugging Face Spaces Docker image
- [x] `requirements.txt` — Runtime dependencies, including multipart uploads
- [x] `deploy.sh` / `deploy.py` — Hugging Face Spaces deployment helper; avoids auto-committing the worktree and excludes common secret files

## Hybrid Input & Status Tagging

- [x] `backend/ingest.py` — RTSP and MP4 via FFmpeg, webcam via `cv2.VideoCapture(0)`, and real-time pacing for local files
- [x] `backend/main.py` — MP4 upload with size limit and temporary storage, hot-swappable ingestion, webcam source switch, and boolean `is_violence` WebSocket tag
- [x] `frontend/src/app/page.tsx` — Webcam control, MP4 upload/drop, and dynamic SAFE/VIOLENCE feed badge
- [x] `README.md` — Document input controls, upload limit, source endpoints, and WebSocket payload

## Final Research & Deployment Phases

- [x] `paper.md` and `paper/paper.md` — Updated from `benchmark_summary.json` and `incidents.jsonl`; distinguish controlled synthetic motion results from real-world accuracy claims
- [x] Hugging Face packaging — Existing Dockerfile, requirements, and deployment helper confirmed present; frontend static export builds successfully
- [x] Validation — Python syntax, backend route registration, Next.js production build, and `git diff --check`
- [x] Synthetic smoke validation — Concurrent extraction produced `(4, 30, 34)` sequences; a temporary ONNX graph exercised rolling inference and SQLite/JSONL incident logging. These fixtures do not represent trained-model accuracy.
- [x] Logger resource handling — SQLite connections are explicitly closed after each operation to avoid lingering database handles.
- [x] Real dataset preparation — 2,000 unique videos processed; balanced `(2000, 30, 34)` features, zero skipped videos
- [x] Real training and ONNX export — 30 epochs, seed 42, 1,600/400 stratified split; artifact successfully loaded by ONNX Runtime and processed a real video clip

---

## Status Legend

**Execution status:** Dataset preparation, LSTM training, ONNX export, actual-clip inference, and a full FastAPI/WebSocket smoke test are complete. Held-out scores at the backend's `> 0.85` threshold are precision 1.000, recall 0.330, and F1 0.496; reference scores at 0.50 are 0.847, 0.775, and 0.809. A single offline 66-frame clip measured 15.89 FPS on this CPU, below the 30 FPS target. Docker is not installed and Hugging Face returned HTTP 402 for a free Docker Space. GitHub commit `8670204` is created locally, but pushing to the configured repository is pending GitHub CLI/Git Credential Manager authentication.

**Runtime note:** The updated FastAPI backend is running at `http://localhost:7860` on a looping local non-violent demo clip, and the Next.js UI is running at `http://localhost:3000`. `/api/health` reports the trained model loaded, and `/ws/stream` delivered JPEG frames with a boolean `is_violence` field.

**Dataset update (2026-10-23):** The downloaded `archive.zip` contains two byte-identical copies of the 2,000 labeled videos. One copy is extracted under `data/raw` (1,000 Violence, 1,000 NonViolence). CPU-only batched pose extraction completed at 11:27 local time. All 2,000 videos were processed with no failures; `X_train.npy` is `(2000, 30, 34)` and `y_train.npy` contains 1,000 examples of each class. Training used 30 epochs, batch size 64, and seed 42; the resulting `backend/violence_lstm.onnx` accepts `[1, 30, 34]` and returns one probability.

| Symbol | Meaning        |
|--------|----------------|
| `[ ]`  | Not started    |
| `[~]`  | In progress    |
| `[x]`  | Complete       |
| `[!]`  | Blocked        |
