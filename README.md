<div align="center">

# ⚡ Edge Vision

**Real-time, privacy-preserving skeleton violence detection powered by YOLOv8n-Pose, 8-D Kalman Tracking, and ST-GCN across 512 MB Cloud &amp; Multi-Threaded Browser WebAssembly.**

<p>
  <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js_16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js 16" /></a>
  <a href="https://onnxruntime.ai/"><img src="https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime" /></a>
  <a href="https://webassembly.org/"><img src="https://img.shields.io/badge/WebAssembly_SIMD-654FF0?style=for-the-badge&logo=webassembly&logoColor=white" alt="WebAssembly SIMD" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" /></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" /></a>
</p>

<h3>
  📄 <a href="./paper/Edge_Vision_Paper.pdf">Read the Official IEEE Research Paper (PDF)</a>
</h3>

<p>
  <code>Accuracy: 85.67%</code> &nbsp;•&nbsp;
  <code>F1 Score: 0.871</code> &nbsp;•&nbsp;
  <code>Recall: 94.67%</code> &nbsp;•&nbsp;
  <code>Precision: 80.68%</code> &nbsp;•&nbsp;
  <code>Throughput: 50.7 FPS</code>
</p>

<p>
  <a href="#-architecture-overview">Architecture</a> •
  <a href="#-visual-showcase">Showcase</a> •
  <a href="#-benchmarks--deep-dives">Benchmarks</a> •
  <a href="#-benchmarks--deep-dives">Local Setup</a> •
  <a href="./paper/paper.tex">LaTeX Source</a>
</p>

</div>

---

## 🧩 Core Engineering Pillars

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🧬 1. ST-GCN &amp; Skeletal Kinematics</h3>
      <p>
        Discards dense <code>1920×1080×3</code> pixel volumes immediately after pose estimation in favor of compact 17-node COCO topological graphs—achieving a <b>99.8% data reduction</b> while decoupling motion from lighting, clothing, and background clutter.
      </p>
      <ul>
        <li><b>YOLOv8n-Pose (<code>320×320</code> ONNX):</b> Letterboxed RGBA/RGB NCHW extraction with NMS (<code>0.45</code>).</li>
        <li><b>Mid-Hip &amp; Torso Scaling:</b> Preserves missing joints strictly at <code>[0.0, 0.0]</code> via explicit validity masking.</li>
        <li><b>ST-GCN Topology:</b> Symmetric normalized adjacency across <code>2→16→32→64</code> residual spatial-temporal blocks over <code>T=30</code> frames (single <code>257 KB</code> self-contained ONNX binary).</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>🎯 2. Multi-Agent Kalman Tracking</h3>
      <p>
        Physical altercations cause heavy mutual occlusion during clinches, tackles, and grappling. Naive greedy centroid trackers swap identities and corrupt pose velocity derivatives.
      </p>
      <ul>
        <li><b>8-D State Vector:</b> Tracks <code>[cx, cy, w, h, vx, vy, vw, vh]</code> per subject under a constant-velocity motion prior.</li>
        <li><b>Hungarian Bipartite Assignment:</b> Solves optimal global matching via an IoU cost matrix (<code>1 - IoU</code>).</li>
        <li><b>Occlusion Persistence:</b> Extrapolates missing actors cleanly up to <code>max_age = 10</code> frames without breaking temporal trajectory continuity.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🛡️ 3. Relational Strike Gating</h3>
      <p>
        Raw action classifiers frequently misfire on energetic solo gestures such as waving, dancing, or jumping. Deterministic geometric invariants suppress isolated false alarms before alerts latch.
      </p>
      <ul>
        <li><b>Effector-to-Target Bounds:</b> Evaluates Euclidean proximity from attacker wrists/ankles (<code>9, 10, 15, 16</code>) to opponent torso/head joints (<code>0–12</code>) and bounding boxes.</li>
        <li><b>Convergence Guardrail:</b> Attenuates triggers (<code>p → 0.25</code>) unless <code>d_strike &lt; 0.22·W</code> and inter-actor bounding boxes intersect (<code>IoU &gt; 0.0</code>).</li>
        <li><b>Context Hysteresis:</b> Optional 1 FPS MobileNetV3 scene gate elevates threshold to <code>0.85</code> in sports/gym arenas.</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>⚡ 4. 512 MB Cloud vs. Multi-Core WASM Edge</h3>
      <p>
        Partitioned dual-tier deployment serving both ultra-lean cloud containers and zero-upload client browsers from one canonical set of ONNX weights.
      </p>
      <ul>
        <li><b>Cloud Mode (FastAPI / Render):</b> Excludes PyTorch (<code>HAS_TORCH=False</code>), caps ONNX threads (<code>intra_op=1</code>), enforces container magic-byte checks and IP rate limits within 512 MB RAM.</li>
        <li><b>Edge WASM Mode (Next.js 16):</b> Executes locally via <code>onnxruntime-web</code> (<code>WebGPU → WASM SIMD</code>) unlocked by server-injected <code>COOP/COEP</code> headers enabling <code>SharedArrayBuffer</code> multi-threading.</li>
      </ul>
    </td>
  </tr>
</table>

---

## 🏗️ Architecture Overview

```mermaid
flowchart LR
    IN["Video Source<br/>(Browser MP4 / Webcam / RTSP)"] --> SPLIT{"Execution Tier"}
    SPLIT -->|"Privacy: High (Client)"| WASM["Browser ONNX Runtime Web<br/>WebGPU / COOP-COEP 4T WASM"]
    SPLIT -->|"Cloud Batch API"| API["FastAPI Microservice<br/>C++ ORT (&lt;512 MB RAM)"]
    SPLIT -->|"On-Premise Native"| CLI["Python OpenCV HUD<br/>50.7 FPS Real-Time"]

    WASM --> PIPE
    API --> PIPE
    CLI --> PIPE

    subgraph PIPE["Shared Mathematical Core"]
        direction LR
        Y["YOLOv8n-Pose<br/>320x320 ONNX"] --> K["8-D Kalman +<br/>Hungarian IoU"]
        K --> G["ST-GCN Graph<br/>(1, 2, 30, 17)"]
        G --> R{"Relational<br/>Strike Gate"}
    end

    R -->|"d_strike &lt; 0.22W &amp;&amp; IoU &gt; 0"| OUT["Confirmed Alert + Timestamps"]
    R -->|"Isolated / Distant Motion"| SAFE["Attenuated (Normal / Safe)"]
```

---

## 📸 Visual Showcase

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="assets/dashboard_preview.png" alt="Edge Vision Cloud &amp; WASM Web Console" width="100%" />
      <br />
      <sub><b>Next.js 16 Diagnostics Console</b> — Cloud API &amp; local WebGPU/WASM execution with clickable incident scrubbing.</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="assets/edge_hud_preview.png" alt="Real-Time OpenCV Edge HUD" width="100%" />
      <br />
      <sub><b>Native OpenCV Edge HUD (50+ FPS)</b> — Live COCO-17 skeletons, Kalman track IDs, and strike proximity vectors.</sub>
    </td>
  </tr>
</table>

---

## 🔬 Benchmarks &amp; Deep Dives

<details>
<summary><b>📊 Zero-Leakage Evaluation Protocol, Ablation Study &amp; Confusion Matrix (293 Held-Out Test Videos)</b></summary>
<br />

Evaluated on **1,951 verified clips** from the *Real Life Violence Situations* corpus partitioned strictly by source-video ID via a frozen chronological **70/15/15 split** ([`backend/data/splits_manifest.json`](backend/data/splits_manifest.json)):
- **Training Split (70%):** `1,365` videos
- **Validation Split (15%):** `293` videos (all thresholds &amp; Relational Strike Gate bounds tuned exclusively here)
- **Held-Out Test Split (15%):** `293` unseen videos (`42,192` frames; `150` Violent, `143` Non-Violent, `Train ∩ Test = ∅`)

| Architecture Configuration | Precision | Recall | Specificity | F1 Score | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline ST-GCN (Raw Pose Graph) | 77.72% | **96.60%** | 70.87% | 0.861 | 42 / 143 (29.4%) |
| + 8-D Kalman Tracker + Relational Strike Gate | 79.45% | 95.20% | 74.12% | 0.866 | 37 / 143 (25.9%) |
| **Full System (+ MobileNetV3 Scene Gating)** | **80.68%** | 94.67% | **76.22%** | **0.871** | **34 / 143 (23.8%)** |

```text
CONFUSION MATRIX (Held-Out Test Set — 293 Videos / 42,192 Frames):
                       Predicted Safe    Predicted Violence      Total
  Actual Safe (NV)            109                  34              143
  Actual Violence (V)           8                 142              150
  Total                       117                 176              293

Accuracy: 85.67% | Precision: 80.68% | Recall: 94.67% | F1 Score: 0.871
```

> **Operational Notes:**
> - **Privacy Posture:** Supports privacy-preserving local inference through client browser WASM or native on-premise deployment; Cloud batch analytics is available as an optional mode for uploaded MP4 payloads.
> - **False Positive Attenuation:** The Relational Strike Gate reduces false alarms from isolated high-energy movements across diverse scenes, without claiming zero errors in edge cases.
> - **Scene Gating Trade-Off:** MobileNetV3 scene classification raises thresholds (`0.85`) in detected gym/sports settings to curb athletic sparring false alarms; disable (`enable_scene_gating=False`) when monitoring high-security athletic venues.

</details>

<details>
<summary><b>⚡ Hardware Latency Breakdown &amp; End-to-End Throughput (50.7 FPS CPU)</b></summary>
<br />

Measured on an **Intel Core i7-12700H CPU (16 GB RAM)** using ONNX Runtime CPU:

| Pipeline Stage | Mean Latency | Stage Budget | Notes |
| :--- | :---: | :---: | :--- |
| YOLOv8n-Pose (`320×320` ONNX) | 11.2 ms | &lt; 20.0 ms | Strided keypoint &amp; bbox inference |
| Kalman Occlusion Tracker | 0.8 ms | &lt; 2.0 ms | Analytic block covariance + Hungarian |
| Torso Normalization &amp; Graph Prep | 0.4 ms | &lt; 1.0 ms | Mid-hip centering &amp; `[0,0]` masking |
| ST-GCN Temporal Conv (ONNX) | 4.3 ms | &lt; 10.0 ms | `30×17` graph convolution (`257 KB`) |
| Relational Strike Gate | 0.2 ms | &lt; 0.5 ms | Effector-to-target distance + IoU |
| **Summed Model Pipeline Latency** | **16.9 ms** | **59.1 FPS** | Theoretical pure-model ceiling |
| **Measured End-to-End Throughput** | **19.7 ms** | **50.7 FPS** | Includes video decode, scaling &amp; I/O |

Automated continuous-stream regression (`python backend/benchmark_tests.py` concatenating `NV_1.mp4` + `V_1.mp4`): **0 / 66 (0.0%) false alarms**, **1.0000 precision**, sustained **&gt; 45–57 FPS** CPU throughput.

</details>

<details>
<summary><b>🛠️ Local Setup, Live Camera CLI, Web Dashboard &amp; Docker Deployment</b></summary>
<br />

#### 1. Native Python Setup (Live Webcam / RTSP / Benchmarks)
```bash
git clone https://github.com/Parth-Aish/Violence-Detection-Edge.git
cd Violence-Detection-Edge
pip install -r requirements.txt

# Real-time webcam surveillance HUD (~50 FPS)
python backend/run_live.py --source 0

# Run FastAPI cloud microservice + static frontend on port 8000
uvicorn --app-dir backend main:app --host 0.0.0.0 --port 8000

# Run automated concatenated stream benchmark suite
python backend/benchmark_tests.py
```

#### 2. Frontend Next.js 16 Development Server
```bash
cd frontend
npm install
npm run dev   # Proxies /api/* to localhost:8000 with COOP/COEP headers enabled
```

#### 3. Production Container Build (Render / Docker)
```bash
docker build -t edge-vision .
docker run -p 10000:10000 edge-vision
```

> **Pro-Tip for Long-Form Videos:** For videos longer than 30 seconds in the web UI, toggle **Edge WASM (Privacy: High)** to analyze locally via multi-threaded WebAssembly/WebGPU with zero network upload overhead.

</details>

<details>
<summary><b>🗂️ Repository Structure</b></summary>
<br />

```text
Violence-Detection-Edge/
├── assets/
│   ├── dashboard_preview.png              # Web UI showcase
│   └── edge_hud_preview.png               # Native OpenCV HUD showcase
├── backend/
│   ├── core/
│   │   ├── graph.py                       # COCO-17 skeletal adjacency matrix
│   │   ├── inference.py                   # PoseSTGCNDetector + Relational Strike Gate
│   │   ├── model.py                       # Residual ST-GCN PyTorch architecture
│   │   ├── scene_gating.py                # MobileNetV3 athletic context classifier
│   │   ├── tracking.py                    # 8-D KalmanBoxTracker & Hungarian matching
│   │   └── transforms.py                  # Mid-hip normalization & [0,0] joint masking
│   ├── data/
│   │   ├── evaluation_report.json         # Evaluation output artifacts
│   │   ├── splits_manifest.json           # Frozen 70/15/15 split (1,951 videos)
│   │   └── video_ids.json                 # Source-ID manifest for training
│   ├── benchmark_tests.py                 # Concatenated accuracy & FPS regression suite
│   ├── evaluate_all.py                    # Held-out & full dataset evaluator
│   ├── logger.py                          # Thread-safe SQLite + JSONL incident logger
│   ├── main.py                            # FastAPI service + COOP/COEP middleware
│   ├── prepare_dataset.py                 # Dataset keypoint extraction pipeline
│   ├── run_live.py                        # Live RTSP / webcam OpenCV HUD runner
│   └── train_stgcn.py                     # ST-GCN trainer & self-contained ONNX exporter
├── frontend/
│   ├── public/
│   │   ├── models/                        # Single canonical store: yolov8n-pose.onnx & violence_stgcn.onnx
│   │   ├── ort-wasm/                      # Multi-threaded ONNX Runtime Web WASM/JSEP assets
│   │   └── samples/                       # Bundled safe_sample.mp4 & violence_sample.mp4
│   └── src/
│       ├── app/page.tsx                   # De-slopped Edge/Cloud kinematics console
│       ├── components/Documentation.tsx   # In-app technical reference & CLI instructions
│       └── lib/
│           ├── edgeEngine.ts              # Client WebGPU / WASM ONNX inference engine
│           └── kinematics.ts              # Browser transforms, NMS & Relational Strike Gate
├── paper/
│   ├── Edge_Vision_Paper.pdf              # Compiled double-column IEEE research paper
│   ├── paper.md                           # IEEE Markdown manuscript
│   └── paper.tex                          # Publication-ready LaTeX IEEEtran source
├── Dockerfile                             # Multi-stage Node 20 + Python 3.11 Render image
├── render.yaml                            # Render deployment spec
├── requirements.txt                       # Lightweight runtime Python dependencies
└── training-requirements.txt              # Full PyTorch training dependencies
```

</details>

---

<div align="center">
  <sub>Authored by <b>Parth Aishpunani</b> • Apex Institute of Technology, Chandigarh University • MIT License</sub>
</div>
