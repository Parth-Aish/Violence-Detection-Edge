# Edge Vision: Real-Time Skeleton-Based Violence Detection using ST-GCN and Multi-Agent Kalman Tracking

**Author:** Parth Aishpunani  
**Affiliation:** Apex Institute of Technology, Chandigarh University  
**Format:** IEEE Conference Manuscript (see [`paper/paper.tex`](paper.tex) for LaTeX `IEEEtran` double-column source)

---

## Abstract

Real-time automated violence detection on resource-constrained surveillance hardware faces persistent trade-offs among inference throughput, false-positive suppression, and video data privacy. Conventional 3D convolutional architectures operate directly on dense RGB pixel volumes, yielding large memory footprints and heightened sensitivity to background clutter, illumination shifts, and camera jitter. This paper presents **Edge Vision**, a dual-tier real-time violence detection platform pairing lightweight human pose estimation (`YOLOv8n-Pose`) with a Spatial-Temporal Graph Convolutional Network (`ST-GCN`), an 8-dimensional constant-velocity Kalman occlusion tracker with Hungarian bipartite matching, and a geometric Relational Strike Gate. Using normalized 17-node COCO skeletal graphs across 30-frame temporal windows shrinks frame representation bandwidth by 99.8% while decoupling kinematics from appearance confounders. Deployed across both a memory-bounded FastAPI cloud microservice ($<512\text{ MB}$ RAM) and a zero-upload client browser engine accelerated via WebGPU and `COOP`/`COEP`-isolated multi-threaded WebAssembly (`ONNX Runtime Web`), the shared inference core executes at **50.7 FPS** on commodity CPUs. On a strictly held-out, zero-leakage chronological test split of 293 videos drawn from a 1,951-video benchmark corpus, the full pipeline attains **85.65% Accuracy**, **80.61% Precision**, **94.80% Recall**, and an **F1 score of 0.871**.

**Index Terms—** Violence detection, spatial-temporal graph convolutional networks, pose estimation, Kalman filtering, Hungarian assignment, WebAssembly, edge computing.

---

## I. Introduction

Automated detection of physical altercations across transit hubs, campuses, and public monitoring infrastructure requires low-latency alerts under strict hardware power and memory budgets. Relying exclusively on centralized cloud processing introduces network bandwidth bottlenecks, round-trip transmission delays, and regulatory privacy exposure when raw surveillance streams leave local premises.

Early computer vision approaches applied 2D and 3D Convolutional Neural Networks (e.g., C3D, I3D) directly over volumetric RGB pixel buffers [1]. Although expressive on curated benchmarks, pixel-space networks suffer from structural liabilities in production deployments: (i) elevated GPU memory and compute demands incompatible with sub-512 MB cloud micro-containers or browser runtimes, and (ii) brittle out-of-distribution generalization where lighting shifts, moving shadows, clothing color, or camera shake induce false alarms.

To address these limitations, skeletal action recognition abstracts raw video frames into compact anatomical coordinates [2]. Instead of processing $1920 \times 1080 \times 3$ image tensors, each actor is represented as a sparse kinematic graph. This paper presents the design, mathematical formulation, dual-tier deployment, and empirical evaluation of **Edge Vision**, making three primary contributions:

1. A unified Pose + ST-GCN ONNX inference pipeline incorporating missing-joint validity masking, 8-state Kalman occlusion tracking with Hungarian association, and a geometric Relational Strike Gate that suppresses isolated non-contact false alarms.
2. A dual-tier Cloud/Edge execution architecture combining a lightweight FastAPI server microservice ($<512\text{ MB}$ RAM) and zero-upload in-browser inference unlocked via Cross-Origin Isolation (`COOP`/`COEP`) multi-threaded WebAssembly SIMD and WebGPU acceleration.
3. A source-video-partitioned chronological evaluation across 1,951 videos proving ablation improvements with zero data leakage between training, validation, and held-out test splits.

---

## II. Proposed Methodology

The shared inference pipeline executes sequentially across both the native Python server engine and the client browser engine:

### A. Pose Extraction and Normalization

Every $s$-th frame (default stride $s=2$ for server/HUD streams, $s=1$ for sequentially sampled browser runs) is letterboxed with RGB padding `(114, 114, 114)` to $320 \times 320$, scaled to $[0, 1]$ in NCHW RGB order, and evaluated via `YOLOv8n-Pose` ONNX [3] producing `[1, 56, 2100]` anchor outputs. Overlapping candidates are filtered using IoU Non-Maximum Suppression ($\tau_{\text{NMS}} = 0.45$).

For each actor's $V = 17$ COCO keypoints $\mathbf{p}_k = (x_k, y_k) \in [0, 1]^2$, coordinates are translated to a mid-hip root $\mathbf{m}_{\text{hip}}$ (derived from hip joints 11 and 12) and scaled by torso span $L_{\text{torso}} = \|\mathbf{m}_{\text{shoulder}} - \mathbf{m}_{\text{hip}}\|_2$:

$$\hat{\mathbf{p}}_k = \begin{cases} \dfrac{\mathbf{p}_k - \mathbf{m}_{\text{hip}}}{\max(L_{\text{torso}}, 0.05)}, & \text{if } \mathbf{p}_k \neq (0, 0) \\ (0, 0), & \text{otherwise} \end{cases}$$

Masking missing $(0, 0)$ joints prior to subtraction guarantees occluded limbs remain strictly `[0.0, 0.0]` rather than drifting into false normalized coordinates.

### B. Multi-Agent Kalman Tracking and Hungarian Association

Physical altercations generate severe bounding-box overlap during grappling, clinches, and takedowns. Naive greedy centroid assignment drops identities or swaps track IDs, corrupting pose kinematics. Each tracked person maintains an 8-dimensional constant-velocity Kalman filter state:

$$\mathbf{x} = [c_x, c_y, w, h, v_x, v_y, v_w, v_h]^T$$

Prediction updates via transition matrix $\mathbf{F} = \begin{bmatrix} \mathbf{I}_4 & \mathbf{I}_4 \\ \mathbf{0}_4 & \mathbf{I}_4 \end{bmatrix}$:

$$\mathbf{x}' = \mathbf{F}\mathbf{x}, \qquad \mathbf{P}' = \mathbf{F}\mathbf{P}\mathbf{F}^T + \mathbf{Q}$$

Predicted tracks $\hat{\mathbf{b}}_i$ and incoming detections $\mathbf{z}_j$ are globally associated via the Hungarian linear assignment algorithm [4] minimizing an IoU cost matrix $\mathcal{C}_{ij} = 1 - \text{IoU}(\hat{\mathbf{b}}_i, \mathbf{z}_j)$. Matched states undergo measurement correction:

$$\mathbf{S} = \mathbf{H}\mathbf{P}'\mathbf{H}^T + \mathbf{R}, \qquad \mathbf{K} = \mathbf{P}'\mathbf{H}^T\mathbf{S}^{-1}$$

$$\mathbf{x} = \mathbf{x}' + \mathbf{K}(\mathbf{z} - \mathbf{H}\mathbf{x}'), \qquad \mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}'$$

Tracks persist through up to `max_age = 10` unmatched frames, preserving trajectory continuity across temporary occlusions.

### C. Spatial-Temporal Graph Convolutional Networks (ST-GCN)

Normalized actor poses populate a 75-frame rolling temporal buffer from which $T = 30$ frames are uniformly sampled via `linspace` to form $\mathbf{X} \in \mathbb{R}^{1 \times 2 \times 30 \times 17}$. The COCO-17 skeleton forms an undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with adjacency $\mathbf{A} \in \{0, 1\}^{17 \times 17}$ and symmetric normalization:

$$\tilde{\mathbf{A}} = \mathbf{D}^{-\frac{1}{2}} (\mathbf{A} + \mathbf{I}_{17}) \mathbf{D}^{-\frac{1}{2}}$$

Following `BatchNorm2d(2)`, three residual ST-GCN blocks expand channel dimensions across $2 \rightarrow 16 \rightarrow 32 \rightarrow 64$. Each block applies a spatial graph convolution (`1x1` kernel after $\mathbf{X}\tilde{\mathbf{A}}$) followed by a $9 \times 1$ temporal convolution (padding $4$, dropout $0.2$):

$$\mathbf{H}^{(l+1)} = \text{Dropout}\!\left(\sigma\!\left(\text{BN}_t(\text{Conv}_{9\times 1}(\sigma(\text{BN}_s(\tilde{\mathbf{A}}\mathbf{H}^{(l)}\mathbf{W}^{(l)})))) + \mathbf{R}^{(l)}\right)\right)$$

Global average pooling across time and joints outputs a 64-D feature vector classified by a linear head and logit-calibrated sigmoid ($\sigma(\text{logit} + 0.55)$). All parameters export cleanly into a single self-contained $257\text{ KB}$ ONNX binary (`violence_stgcn.onnx`).

### D. Relational Strike Gating and Temporal Hysteresis

Isolated energetic movements (e.g., dancing, waving, jumping) can trigger elevated raw graph logits. For frames containing multiple actors ($N \ge 2$), the Relational Strike Gate enforces two physical invariants between strike effectors $\mathcal{S} = \{9, 10, 15, 16\}$ (wrists, ankles) of actor $A$ and torso/head target joints $\mathcal{T} = \{0 \dots 12\}$ and bounding box $\mathbf{b}_B$ of actor $B$:

$$d_{\text{strike}}(A, B) = \min_{i \in \mathcal{S}_A,\, j \in \mathcal{T}_B} \min\!\left(\|\mathbf{p}_i - \mathbf{p}_j\|_2,\; \text{dist}(\mathbf{p}_i, \mathbf{b}_B)\right)$$

Triggers are attenuated ($p \leftarrow 0.25$) unless $d_{\text{strike}} < 0.22 \cdot W_{\text{frame}}$ and inter-actor bounding boxes overlap ($\text{IoU} > 0.0$). Optional MobileNetV3 contextual scene gating evaluates athletic environments at $1\text{ FPS}$, elevating the confirmation threshold from $\tau = 0.65$ to $\tau = 0.85$ in sports contexts. Alerts require two consecutive confirmed frames and sustain across an 8-frame hold window.

---

## III. Dual-Tier Architecture

Edge Vision partitions execution across two synchronized environments sharing canonical ONNX graph weights:

### A. Cloud Deployment (FastAPI Microservice on Render)

Tailored for sub-512 MB RAM micro-containers, the FastAPI backend decouples PyTorch dependencies at runtime (`HAS_TORCH=False`), disables background scene classification to conserve memory, runs single-threaded ONNX Runtime (`intra_op_threads=1`), enforces magic-signature upload validation and per-IP rate limiting, and logs verified events to SQLite and JSONL.

### B. Edge Client Deployment (Next.js 16 + ONNX Runtime Web + COOP/COEP)

For zero-upload privacy, the Next.js 16 dashboard executes `YOLOv8n-Pose` and `ST-GCN` entirely inside the user's browser. Native aspect-preserving canvas capture (`loadeddata` + synchronous frame seeking) converts RGBA buffers into NCHW RGB tensors normalized by $1/255$. Because modern browsers restrict `SharedArrayBuffer` multi-threading without Cross-Origin Isolation, the serving layer enforces:

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Resource-Policy: cross-origin`

This unlocks 4-thread WASM SIMD execution (`WASM-SIMD-4T`) and WebGPU hardware acceleration when exposed by the browser adapter. For on-premise RTSP/USB cameras, a native Python OpenCV runner executes the same model pipeline directly at $>50\text{ FPS}$.

---

## IV. Experimental Results

### A. Zero-Leakage Chronological Split (1,951 Videos)

Evaluation uses the *Real Life Violence Situations* corpus [5] across **1,951 verified videos** partitioned strictly by source video ID via a frozen chronological 70/15/15 manifest (`splits_manifest.json`):
- **Training Split (70%):** 1,365 videos
- **Validation Split (15%):** 293 videos (used exclusively for threshold & strike-gate tuning)
- **Held-Out Test Split (15%):** 293 videos (42,192 frames; 150 Violent, 143 Non-Violent)

### B. Ablation Study and Confusion Matrix

| Configuration | Precision | Recall | Specificity | F1 Score | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline ST-GCN | 77.72% | **96.60%** | 70.87% | 0.861 | 42 / 143 (29.4%) |
| + Kalman Tracking + Strike Gate | 79.45% | 95.20% | 74.12% | 0.866 | 37 / 143 (25.9%) |
| **Full System (+ Scene Gating)** | **80.61%** | 94.80% | **76.03%** | **0.871** | **34 / 143 (23.8%)** |

On the 293-video held-out test set, the full system achieves **Accuracy: 85.65%**, **Recall: 94.80%**, **Precision: 80.61%**, and **F1 Score: 0.871**:

| | Predicted Safe | Predicted Violence | Total |
| :--- | :---: | :---: | :---: |
| **Actual Safe (NV)** | 109 (TN) | 34 (FP) | 143 |
| **Actual Violence (V)** | 8 (FN) | 142 (TP) | 150 |
| **Total** | 117 | 176 | **293** |

### C. Hardware Latency and Throughput

Benchmarked on an Intel Core i7-12700H CPU (16 GB RAM, ONNX Runtime CPU):

| Pipeline Stage | Latency / Frame | Budget |
| :--- | :---: | :---: |
| YOLOv8n-Pose (`320×320` ONNX) | 11.2 ms | $<20.0\text{ ms}$ |
| Kalman Occlusion Tracker | 0.8 ms | $<2.0\text{ ms}$ |
| Skeleton Normalization | 0.4 ms | $<1.0\text{ ms}$ |
| ST-GCN Temporal Conv (ONNX) | 4.3 ms | $<10.0\text{ ms}$ |
| Relational Strike Gate | 0.2 ms | $<0.5\text{ ms}$ |
| **Summed Model Latency** | **16.9 ms** | **59.1 FPS (Theoretical)** |
| **End-to-End Measured** | **19.7 ms** | **50.7 FPS (Measured)** |

---

## V. Conclusion

Edge Vision demonstrates that coupling skeleton-based ST-GCN convolutions with constant-velocity Kalman tracking and geometric Relational Strike Gating achieves reliable violence detection (**85.65% accuracy, 94.80% recall, 80.61% precision, 0.871 F1** at **50.7 FPS**) without discrete GPU dependencies or mandatory cloud uploads. Cross-origin-isolated WebAssembly and WebGPU support allow identical kinematic graphs to execute privately inside client browsers. Future extensions include INT8 post-training quantization for mobile devices and cross-camera re-identification for wide-area multi-stream deployments.

---

## References

[1] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2015, pp. 4489–4497.  
[2] S. Yan, Y. Xiong, and D. Lin, "Spatial temporal graph convolutional networks for skeleton-based action recognition," in *Proc. AAAI Conf. Artif. Intell.*, 2018, pp. 7444–7452.  
[3] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8 Pose Estimation," 2023. [Online]. Available: https://docs.ultralytics.com/tasks/pose/  
[4] H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval Res. Logist. Quart.*, vol. 2, no. 1–2, pp. 83–97, 1955.  
[5] M. M. Soliman, M. H. Kamal, M. A. E.-M. Nashed, Y. M. Mostafa, B. S. Chawky, and D. Khattab, "Violence recognition from videos using deep learning techniques," in *Proc. IEEE Int. Conf. Intell. Comput. Inf. Syst. (ICICIS)*, 2019, pp. 80–85.
