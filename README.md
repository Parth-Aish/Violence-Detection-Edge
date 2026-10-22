---
title: Violence Detection System
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Edge-Optimized Violence Detection System

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/YOLOv8-Pose-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/Next.js-15-black?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-v4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License" />
</p>

> **Spatio-Temporal Violence Detection for Legacy Surveillance Infrastructure**  
> Eliminates heavy 3D-CNNs and video transformers by pairing single-shot 2D pose estimation (YOLOv8n-pose) with zero-parameter temporal velocity kinematics. Runs real-time (15 FPS) on commodity CPU edge hardware with zero frame-buffer memory leaks.

---

## Overview

Modern action-recognition models (SlowFast, I3D, TimeSformer) demand high-end discrete GPUs, making them financially and thermally impractical for retrofit installations across convenience stores, transit stations, and schools. 

This repository provides an end-to-end, edge-deployable violence detection system that replaces 200+ GFLOP 3D convolutions with:
1. **Perception**: Single-pass 17-keypoint 2D human pose estimation via YOLOv8n-pose (~3.2 GFLOPs).
2. **Temporal Reasoning**: A 15-frame rolling kinematic velocity heuristic with exponential moving average (EMA) smoothing ($\alpha = 0.4$) and proximity-aware severity escalation.
3. **Robust Ingestion**: An isolated FFmpeg producer process for RTSP and MP4, with native OpenCV webcam capture and a single-slot `multiprocessing.Queue(maxsize=1)` drop-old policy that prevents an unbounded frame backlog.
4. **Dual-Sink Persistence**: Concurrent incident logging to SQLite in Write-Ahead Logging (WAL) mode and append-only JSON Lines (`incidents.jsonl`).
5. **Real-Time Glassmorphism UI**: Next.js 16 dashboard streaming annotated video and incident alerts over WebSockets.

---

## System Architecture

The following Mermaid.js diagram illustrates the Producer-Consumer multiprocessing pipeline and bidirectional WebSocket streaming architecture:

```mermaid
flowchart TD
    subgraph Capture["1. Ingestion Layer (Producer Process)"]
        A[Legacy RTSP IP Camera / Video / Webcam] -->|TCP H.264 / H.265| B[FFmpeg Subprocess]
        B -->|Raw BGR24 Pipe stdout| C[Frame Worker Loop]
        C -->|Drop Old Frames If Full| D[("multiprocessing.Queue(maxsize=1)")]
    end

    subgraph Core["2. Core Engine (Asyncio Event Loop)"]
        D -->|run_in_executor get| E[Frame Consumer]
        E --> F["YOLOv8n-Pose (PyTorch CPU / NPU)"]
        F -->|17 Keypoints per Subject| G[Temporal Velocity Classifier]
        
        subgraph Kinematics["Kinematic Heuristic Logic"]
            G --> H1["Euclidean Joint Delta: Wrists, Elbows, Ankles"]
            H1 --> H2["EMA Smoothing (alpha = 0.4)"]
            H2 --> H3["Consecutive Frame Counter (min_trigger >= 3)"]
            H3 --> H4{"Proximity Check (< 150px)?"}
            H4 -- Yes --> H5["Escalate Severity (+1 Tier)"]
            H4 -- No --> H6["Classify: LOW / MED / HIGH / CRITICAL"]
        end
    end

    subgraph Logging["3. Dual-Sink Persistence"]
        H5 & H6 -->|Deduplication Cooldown 2.0s| I[IncidentLogger]
        I -->|Thread-Safe Commit| J[("SQLite DB (WAL Mode)")]
        I -->|Append Line| K["Streaming incidents.jsonl"]
    end

    subgraph Interface["4. Presentation Layer"]
        E --> L[Frame Annotator OpenCV]
        L -->|JPEG Base64 Encode| M[FastAPI WebSocket Server /ws/stream]
        H5 & H6 -->|JSON Event Payload| M
        M -->|WebSocket Broadcast| N["Next.js Glassmorphism Dashboard"]
        J -->|REST /api/incidents| N
    end
```

---

## Edge Hardware Deployment Requirements

The system is engineered specifically for low-power edge nodes without requiring cloud compute:

| Specification | Minimum Tier (e.g. Raspberry Pi 5) | Recommended Tier (e.g. Jetson Orin Nano / Intel NUC) |
| :--- | :--- | :--- |
| **Processor** | Quad-core ARM Cortex-A76 @ 2.4GHz | 6-core ARM Cortex-A78AE or Intel Core i3 / i5 |
| **Hardware Acceleration** | None (Pure CPU inference) | CUDA / TensorRT / OpenVINO |
| **System Memory (RAM)** | 4 GB LPDDR4X | 8 GB LPDDR5 / DDR4 |
| **Memory Footprint (RSS)** | ~490 MB Resident | ~550 MB Resident |
| **Operating System** | Ubuntu 22.04 LTS / Debian 12 / Windows 11 | Ubuntu 22.04 LTS (JetPack 6.x) |
| **Power Consumption** | ~8W – 12W | ~15W – 25W |
| **Network Interface** | 100Mbps Ethernet (RJ45) | Gigabit Ethernet (1000BASE-T) |
| **Camera Interface** | RTSP (H.264/H.265), USB V4L2 webcam, or file | Multi-channel RTSP streams |

---

## Real-World Impact & Benchmarks

All metrics reported below were recorded directly on an unaccelerated local CPU environment across 130 sequential test frames, verified in `backend/data/incidents.db`:

- **YOLOv8n-Pose Inference Latency:** Mean 65.43 ms (Median: 63.95 ms, p95: 90.67 ms)
- **Kinematic Velocity Classifier Latency:** 0.095 ms (95 microseconds per frame)
- **End-to-End Pipeline Latency:** Mean 66.95 ms (p95: 74.31 ms)
- **Sustained System Throughput:** 14.90 FPS
- **Recorded Incidents in Test Run:** 11 events (6 CRITICAL, 4 HIGH, 1 MEDIUM)
- **Dual-Sink Storage Overhead:** 16.0 KB (SQLite WAL) + 3.6 KB (JSONL)

---

## Local Setup & Installation

### Prerequisites

Ensure you have the following installed on your machine:
- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **Node.js 18+** & **npm**
- **FFmpeg** installed and accessible from your system `PATH` (verify with `ffmpeg -version`)

---

### Step 1: Backend Setup (FastAPI & Inference Engine)

1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```

2. (Optional but recommended) Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the benchmark validation suite (verifies pose extraction, runs the kinematic heuristics, and generates `backend/data/` logs):
   ```bash
   python benchmark_run.py
   ```

5. Launch the backend server:
   ```bash
   # Option A: Run using default webcam (index 0)
   python main.py

   # Option B: Run with an RTSP camera stream
   STREAM_SOURCE="rtsp://admin:password@192.168.1.100:554/live" python main.py

   # Option C: Run with a local MP4 test video
   STREAM_SOURCE="test_fight.mp4" python main.py
   ```

The FastAPI application will start on `http://localhost:7860`.

---

### Step 2: Frontend Setup (Next.js 16 Dashboard)

1. Open a new terminal tab and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install JavaScript dependencies:
   ```bash
   npm install
   ```

3. Start the Next.js development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

The dashboard will automatically connect to `ws://localhost:7860/ws/stream`, displaying the annotated camera feed, frame rate, and incident log.

Use **Use Webcam** to switch to local camera index zero. Choose or drop an `.mp4` file with **Upload Video File** to replace the active source; uploaded clips play once at the configured frame rate. Uploads are limited to 500 MiB by default.

---

## Configuration & Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `STREAM_SOURCE` | `"0"` | RTSP stream URI, video file path, or webcam index (`"0"`, `"1"`) |
| `FRAME_WIDTH` | `640` | Video frame downscale width in pixels |
| `FRAME_HEIGHT` | `480` | Video frame downscale height in pixels |
| `FRAME_FPS` | `15` | Target ingestion and inference frame rate |
| `MODEL_PATH` | `"yolov8n-pose.pt"` | Ultralytics YOLOv8 pose model checkpoint |
| `MAX_UPLOAD_BYTES` | `524288000` | Maximum uploaded MP4 size in bytes |
| `JPEG_QUALITY` | `70` | Compression quality for WebSocket video frames (1–100) |
| `LOG_LEVEL` | `"INFO"` | Standard Python logger level (`DEBUG`, `INFO`, `WARNING`) |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:7860/ws/stream` | WebSocket streaming endpoint for Next.js |
| `NEXT_PUBLIC_API_URL`| `http://localhost:7860` | REST API base URL for incident queries |

---

## API Endpoints

- `GET /`: Health check and system operational status.
- `GET /api/incidents`: Query recent incident history from SQLite (`limit`, `min_severity`, `since_timestamp`).
- `GET /api/stats`: Retrieve 24-hour incident volume and severity counts.
- `POST /source/webcam`: Switch ingestion to local webcam index zero.
- `POST /upload`: Upload and play an MP4 file (multipart field: `file`).
- `WS /ws/stream`: Broadcast base64 JPEG frames with `is_violence`, detection count, timestamp, and incident events.

---

## Repository Structure

```
Violence/
├── backend/
│   ├── benchmark_run.py     # End-to-end benchmark suite and test runner
│   ├── data/                # Runtime SQLite WAL database & JSONL incident logs
│   │   ├── benchmark_summary.json
│   │   ├── incidents.db
│   │   └── incidents.jsonl
│   ├── inference.py         # YOLOv8n-Pose & temporal velocity classifier
│   ├── ingest.py            # FFmpeg RTSP ingestion (Producer-Consumer Queue)
│   ├── logger.py            # Dual-sink incident logger (SQLite WAL + JSONL)
│   ├── main.py              # FastAPI server, lifespan orchestration, WebSocket
│   └── requirements.txt     # Python dependency specifications
├── frontend/
│   ├── src/app/
│   │   ├── globals.css      # Custom glassmorphism & dark-mode styling
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Next.js interactive surveillance dashboard
│   ├── package.json         # Node.js project manifest
│   └── tsconfig.json        # TypeScript configuration
├── paper/
│   └── paper.md             # Data-driven research paper
├── paper.md                 # Root research paper copy
├── PROGRESS.md              # Project status tracking checklist
├── README.md                # Project documentation & setup guide
└── .gitignore               # Ignored build artifacts, models, and virtualenvs
```

---

## Research Paper

The complete academic paper is available at [`paper/paper.md`](paper/paper.md):
> **"Edge-Optimized Spatio-Temporal Violence Detection for Legacy Surveillance Infrastructure"**  
> *Applied Systems & Vision Group, October 2026*

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
