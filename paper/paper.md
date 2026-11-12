# Edge Vision: Real-Time Skeleton-Based Violence Detection using ST-GCN and Multi-Agent Kalman Tracking

**Author:** Parth Aishpunani  
**Affiliation:** Apex Institute of Technology, Chandigarh University  
**Artifacts:** [Compiled IEEE PDF (`paper/Edge_Vision_Paper.pdf`)](Edge_Vision_Paper.pdf) • [LaTeX `IEEEtran` Source (`paper/paper.tex`)](paper.tex)

---

## Abstract

Real-time automated violence detection across distributed surveillance cameras is constrained by hardware compute ceilings, high false-positive rates on benign motion, and privacy hazards when raw video is streamed to centralized servers. Traditional 3D convolutional and video-transformer architectures operate over dense RGB pixel volumes, yielding large memory footprints and brittle sensitivity to background clutter, illumination shifts, and camera jitter. This work introduces **Edge Vision**, a dual-tier real-time violence detection platform unifying lightweight pose estimation (`YOLOv8n-Pose`), a Spatial-Temporal Graph Convolutional Network (`ST-GCN`), an 8-state constant-velocity Kalman occlusion tracker with Hungarian bipartite assignment, and a novel pixel-space geometric **Relational Strike Gate**. Representing each subject via normalized 17-node COCO skeletal graphs across 30-frame temporal windows achieves over 99.9% per-frame, per-actor input dimensionality reduction (reducing single $1920 \times 1080 \times 3$ RGB frames of 6.22 MB to $17 \times 2$ 32-bit floats [136 bytes] per actor for ST-GCN, parsed from $17 \times 3$ pose triplets, while the downstream ST-GCN classifier relies on a 30-frame temporal graph tensor of shape $(1, 2, 30, 17)$ for final sequence classification) while decoupling biomechanical motion from appearance confounders. The identical ONNX graph pipeline runs inside both a memory-bounded FastAPI REST cloud microservice ($<512\text{ MB}$ RAM ceiling) and a zero-upload client browser Edge Mode accelerated via WebGPU and `COOP`/`COEP`-isolated multi-threaded WebAssembly (`ONNX Runtime Web`). Benchmarked over 100 timed iterations (after 10 warm-up runs) including OpenCV decoding and YOLOv8n-Pose extraction on an 11th Gen Intel Core i5-1135G7 CPU, strided execution (`pose_stride=2`, 4 threads) achieves **$22.04 \pm 14.87\text{ ms}$** per frame (Median P50: $24.05\text{ ms}$, P95: $48.52\text{ ms}$; **45.4 processed-frame throughput (FPS)** defined as $1000/\text{mean}$; $16.32\text{ ms}$ / $61.3\text{ FPS}$ process-only). Because `pose_stride=2` is used, this represents the model's processing speed rather than the source-video ingestion rate. On a held-out test split of 293 unseen videos partitioned via an ordered source-ID split from 1,951 verified RLVS clips to prevent source-level data leakage ($\text{TP}=146, \text{FP}=46, \text{FN}=4, \text{TN}=97$), the complete system achieves **82.94% Accuracy**, **76.04% Precision**, **97.33% Recall**, **67.83% Specificity**, and an **F1 Score of 0.854**.

**Index Terms—** Violence detection, spatial-temporal graph convolutional networks, pose estimation, Kalman filtering, Hungarian algorithm, relational strike gating, WebAssembly, edge AI.

---

## I. Introduction

Automated surveillance systems deployed in public transit corridors, campuses, correctional facilities, and retail spaces require sub-second detection of physical altercations while minimizing alert fatigue. Transmitting continuous high-definition video feeds to centralized GPU clusters incurs network bandwidth bottlenecks, variable transmission latency, and privacy compliance exposure when raw frames leave local premises. Conversely, deploying heavy deep networks onto edge devices, browser sandboxes, or memory-capped cloud containers imposes tight bounds on RAM, model binary size, and single-frame compute time.

First-generation deep learning violence detectors operated directly over volumetric RGB pixel buffers using 2D+LSTM or 3D Convolutional Neural Networks such as C3D [1]. Dense pixel models suffer from cubic spatial-temporal memory scaling and susceptibility to visual confounders including lighting flicker, rain, camera shake, and clothing colors.

Skeleton-based action recognition projects each subject into a compact topological graph of anatomical keypoints [2], compressing $1920 \times 1080 \times 3$ RGB frames (6,220,800 channel bytes) into normalized $(17, 2)$ 32-bit float graph tensors (136 bytes per actor for ST-GCN, parsed from $17 \times 3$ pose triplets of 204 bytes)—achieving over 99.9% per-frame, per-actor input dimensionality reduction while stripping background texture and facial identity cues (note that while per-frame inputs are reduced to 136 bytes per actor, the ST-GCN classifier relies on a 30-frame temporal graph tensor $(1, 2, 30, 17)$ for final sequence classification). Nevertheless, standalone skeletal classifiers encounter two severe operational failure modes: (1) track fragmentation and ID switching when combatants heavily occlude one another during physical clinches, and (2) elevated false-positive rates on vigorous non-violent solo gestures (dancing, waving, solo jumping). Edge Vision solves both problems via four contributions:

1. **Unified Missing-Joint Masked ST-GCN Pipeline:** A residual ST-GCN exported into a single self-contained $257\text{ KB}$ ONNX binary paired with mid-hip root centering, torso scaling, and strict $[0.0, 0.0]$ validity masking for occluded joints.
2. **True 8-D Kalman + Hungarian Occlusion Tracker:** Analytic constant-velocity Kalman state prediction and optimal IoU linear assignment; qualitative observation during validation indicates the 8-D state vector preserves track identities through transient occlusions, reducing ID switches compared to greedy IoU matching.
3. **Pixel-Space Relational Strike Gating:** A dual-invariant geometric contact filter evaluating effector-to-target Euclidean proximity in pixels alongside bounding-box convergence ($\text{IoU} > 0$), explicitly addressing edge geometries where naive box-centroid distance heuristics fail.
4. **Hybrid Browser Edge Mode & Cloud REST Architecture:** Browser WASM inference successfully reproduces the expected classification outputs on two regression test clips, demonstrating functional parity with the Python backend and guaranteeing zero-upload client privacy via `COOP`/`COEP` `SharedArrayBuffer` WASM SIMD and WebGPU, operating alongside a standard 512 MB FastAPI Cloud REST API.

---

## II. Related Work

### A. RGB-Volumetric and Weakly Supervised Violence Detection
Large-scale benchmarks catalyzed unconstrained surveillance violence research. Soliman et al. [5] curated the *Real Life Violence Situations* dataset capturing real indoor/outdoor altercations versus daily activities. Cheng et al. [6] introduced RWF-2000 from surveillance feeds using RGB and optical-flow gated branches. Wu et al. [7] scaled weak supervision via the multimodal XD-Violence corpus, while Degardin and Proença [8] released UBI-Fights spanning 1,000 frame-annotated sequences. Gohar [9] applied Convolutional Block Attention Modules (CBAM) with ConvLSTM architectures on UBI-Fights to focus spatial attention on fight regions. Despite high benchmark scores, volumetric RGB and optical-flow pipelines exceed sub-second CPU-only inference budgets.

### B. Skeleton-Based Action Recognition via Graph Convolutions
Yan et al. [2] introduced Spatial-Temporal Graph Convolutional Networks (ST-GCN), modeling anatomical joints across fixed normalized adjacency graphs. Subsequent advances expanded multi-scale and adaptive graph learning: Shi et al. [10] proposed Two-Stream Adaptive GCN (2s-AGCN) to learn adaptive topology and bone features; Liu et al. [11] introduced MS-G3D to disentangle multi-scale neighborhoods; and Chen et al. [12] formulated CTR-GCN for channel-wise topology refinement.

For computational efficiency, Song et al. [13] developed EfficientGCN via compound scaling; Duan et al. [14] proposed PoseConv3D over 3D joint heatmaps and standardized ST-GCN++ practices inside PYSKL [15]; and Chi et al. [16] introduced InfoGCN using information-bottleneck regularization. Recent top-performing architectures include Hierarchically Decomposed GCN (HD-GCN) [17], SkateFormer [18] for skeletal-temporal attention, and BlockGCN [19] preserving topological structure. While expressive on multi-class benchmarks, multi-stream and heavy transformer graphs exceed client WebAssembly CPU latency targets.

### C. Skeleton Violence Detection & Contrast Against Distance Gating
Recent works apply skeleton graphs specifically to physical violence. Yang et al. [20] proposed multi-scale attention-enhanced GCNs fusing position, bone, and motion streams. Janbi et al. [21] introduced ESTS-GCN using ensemble spatial-temporal graph topologies, and Tran et al. [22] explored interaction-line skeletal graphs (SKE-A3TGCN, GCNVertexEdge) for low-resolution CCTV feeds.

A critical weakness of conventional heuristic gating is thresholding only *bounding-box centroid distance* $\|\mathbf{c}_A - \mathbf{c}_B\|_2 < d_{\text{thresh}}$. Centroid spacing fails systematically under two everyday cases: (i) two pedestrians standing calmly side-by-side have small centroid distance without hostile motion, whereas (ii) two fighting actors trading wide punches or kicks maintain large torso-centroid spacing while an extremity directly strikes the opponent. Edge Vision's **Relational Strike Gate** closes this gap by operating directly between extremal strike effectors $\{9, 10, 15, 16\}$ and opponent vulnerable torso/head joints $\{0\dots 12\}$ and bounding-box borders in pixel space alongside $\text{IoU} > 0$.

### D. Multi-Object Tracking and Edge Runtime Optimization
Single-stage pose estimators such as YOLO-Pose [23] and YOLOv8n-Pose [3] jointly regress bounding boxes and COCO-17 skeletons in one forward pass. Real-time tracking-by-detection pipelines including ByteTrack [24] and BoT-SORT [25] combine Kalman filtering with Kuhn's Hungarian algorithm [4], proving that geometric IoU association avoids heavy CNN Re-ID overhead. ONNX Runtime [26] provides portable optimization across C++ server CPUs and browser WebGPU/WASM runtimes, while MobileNetV3 [27] supplies low-cost contextual scene classification.

---

## III. Proposed Methodology

### A. Pose Extraction and Missing-Joint Masked Normalization
Frames of resolution $W_{\text{frame}} \times H_{\text{frame}}$ are letterboxed to $320 \times 320$ with constant RGB padding $(114, 114, 114)$, stripped of alpha channels in browser contexts, normalized to $[0, 1]$ in NCHW RGB order, and processed by `YOLOv8n-Pose` ONNX yielding `[1, 56, 2100]` anchors followed by IoU NMS ($\tau_{\text{NMS}} = 0.45$). Detections are unpadded strictly back into **pixel-space coordinates** $[0, W_{\text{frame}}-1] \times [0, H_{\text{frame}}-1]$:

$$x_{\text{px}} = \text{clip}\!\left(\frac{x_{\text{model}} - p_x}{\gamma},\, 0,\, W_{\text{frame}}-1\right), \quad y_{\text{px}} = \text{clip}\!\left(\frac{y_{\text{model}} - p_y}{\gamma},\, 0,\, H_{\text{frame}}-1\right)$$

For ST-GCN input formatting, coordinates are scaled to $[0, 1]^2$ and masked via indicator $m_k = \mathbb{I}(\mathbf{p}_k \neq (0, 0))$ prior to mid-hip root centering ($\mathbf{m}_{\text{hip}}$ from valid hips $\{11, 12\}$) and torso scaling ($L_{\text{torso}} = \|\mathbf{m}_{\text{shoulder}} - \mathbf{m}_{\text{hip}}\|_2$):

$$\hat{\mathbf{p}}_k = \begin{cases} \dfrac{\mathbf{p}_k - \mathbf{m}_{\text{hip}}}{\max(L_{\text{torso}}, 0.05)}, & \text{if } m_k = 1 \text{ and } L_{\text{torso}} > 0.01 \\ \dfrac{\mathbf{p}_k - \mathbf{m}_{\text{hip}}}{\max(0.5 \cdot \Delta y_{\text{valid}}, 0.1)}, & \text{if } m_k = 1 \text{ and } L_{\text{torso}} \le 0.01 \\ (0.0, 0.0), & \text{if } m_k = 0 \end{cases}$$

### B. Multi-Agent Kalman Filter & Hungarian Assignment
`OcclusionKalmanTracker` tracks each actor using an 8-D constant-velocity pixel-space state vector:

$$\mathbf{x}_{k|k} = [c_x, c_y, w, h, v_x, v_y, v_w, v_h]^T \in \mathbb{R}^8$$

with transition $\mathbf{F} = \begin{bmatrix} \mathbf{I}_4 & \mathbf{I}_4 \\ \mathbf{0}_4 & \mathbf{I}_4 \end{bmatrix}$, observation matrix $\mathbf{H} = [\mathbf{I}_4 \;\; \mathbf{0}_4]$, $\mathbf{P}_0 = \text{diag}(10\mathbf{1}_4, 1000\mathbf{1}_4)$, $\mathbf{Q} = \text{diag}(\mathbf{1}_4, 0.01\mathbf{1}_4)$, and $\mathbf{R} = \text{diag}(1, 1, 10, 10)$.

1. **State Prediction:**
   $$\mathbf{x}_{k|k-1} = \mathbf{F}\mathbf{x}_{k-1|k-1}, \qquad \mathbf{P}_{k|k-1} = \mathbf{F}\mathbf{P}_{k-1|k-1}\mathbf{F}^T + \mathbf{Q}$$
2. **Hungarian Assignment:** Associates predicted boxes $\hat{\mathbf{b}}_i$ and detections $\mathbf{z}_j$ via `scipy.optimize.linear_sum_assignment` minimizing $\mathcal{C}_{ij} = 1 - \text{IoU}(\hat{\mathbf{b}}_i, \mathbf{z}_j)$ ($\text{IoU} \ge 0.15$; retiring unmatched tracks after `max_age = 10` frames). Qualitative observation during validation indicates the 8-D state vector successfully preserves track identities through transient occlusions, reducing ID switches compared to greedy IoU matching.
3. **Measurement Update:** Solved cleanly via `np.linalg.solve` without explicit matrix inversion:
   $$\mathbf{y}_k = \mathbf{z}_k - \mathbf{H}\mathbf{x}_{k|k-1}, \qquad \mathbf{S}_k = \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R}$$
   $$\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}^T\mathbf{S}_k^{-1}, \qquad \mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k\mathbf{y}_k, \qquad \mathbf{P}_{k|k} = (\mathbf{I}_8 - \mathbf{K}_k\mathbf{H})\mathbf{P}_{k|k-1}$$

### C. ST-GCN Topology & Training Reproducibility Details
Thirty uniformly sampled frames (`linspace`) from the rolling 75-pose buffer assemble $\mathbf{X} \in \mathbb{R}^{1 \times 2 \times 30 \times 17}$ over the normalized COCO-17 graph:

$$\tilde{\mathbf{A}} = \mathbf{D}^{-\frac{1}{2}} (\mathbf{A} + \mathbf{I}_{17}) \mathbf{D}^{-\frac{1}{2}}$$

Three residual `STGCNBlock` layers ($2 \rightarrow 16 \rightarrow 32 \rightarrow 64$, temporal kernel $9 \times 1$, padding $(4, 0)$, stride $(1, 1)$, dropout $0.2$) precede global average pooling and linear classification ($\sigma(\text{logit} + 0.55)$ runtime calibration). Exact training settings: `seed=42`, batch size `32`, `25` epochs, `AdamW` ($\text{lr}=10^{-3}$, $\text{wd}=10^{-4}$), `CosineAnnealingLR` ($T_{\max}=25$, $\eta_{\min}=10^{-5}$), gradient clipping ($\le 2.0$), and `BCEWithLogitsLoss(pos_weight=1.3)` with `0.05` label smoothing plus flip+swap, Gaussian noise ($\mathcal{N}(0, 0.015^2)$), and scale jitter ($\mathcal{U}(0.95, 1.05)$).

### D. Pixel-Space Relational Strike Gate & Unit Consistency
When $N \ge 2$ actors appear in a frame of pixel width $W_{\text{frame}}$, all keypoints $\mathbf{p}_i$ and bounding boxes $\mathbf{b}_v$ are evaluated strictly in **pixel units** (in TypeScript `kinematics.ts`, normalized coordinates in $[0, 1]^2$ are scaled by $(W_{\text{frame}}, H_{\text{frame}})$ first). Define effectors $\mathcal{S} = \{9, 10, 15, 16\}$ and targets $\mathcal{T} = \{0 \dots 12\}$:

$$\text{dist}_{\text{px}}(\mathbf{p}_i, \mathbf{b}_v) = \sqrt{\max(b_{x1}-x_i, 0, x_i-b_{x2})^2 + \max(b_{y1}-y_i, 0, y_i-b_{y2})^2}$$

$$d_{\text{strike}}(A, B) = \min_{\substack{(u, v) \in \{(A, B), (B, A)\} \\ i \in \mathcal{S}_u,\; j \in \mathcal{T}_v}} \min\!\left(\|\mathbf{p}_i - \mathbf{p}_j\|_2,\; \text{dist}_{\text{px}}(\mathbf{p}_i, \mathbf{b}_v)\right)$$

Alerts above $0.65$ clamp to $0.25$ unless both invariants hold in matching pixel units:

$$d_{\text{strike}}(A, B) \le \alpha \cdot W_{\text{frame}} \qquad \text{AND} \qquad \text{IoU}(\mathbf{b}_A, \mathbf{b}_B) > 0.0$$

with $\alpha = 0.22$ in general scenes ($\alpha = 0.12$ when MobileNetV3 flags athletic scenes at $1\text{ FPS}$, raising confirmation threshold $\tau = 0.85$), requiring 2 consecutive frames across an 8-frame hold window.

**Comparison Against Naive Centroid Distance Gating:** Unlike a naive bounding-box centroid heuristic $\|\mathbf{c}_A - \mathbf{c}_B\|_2 \le d_{\text{thresh}}$, the dual-invariant Relational Strike Gate addresses two systematic failure modes during partial occlusion and asymmetric limb extension. First, under heavy mutual overlap where one combatant shields the torso of another, partially occluded detections shrink box extents and perturb centroids unevenly; evaluating effector joints $\mathcal{S}$ against exposed target joints in $\mathcal{T}$ preserves true strike proximity even when centroids drift. Second, in wide-stance attacks (high kicks or lunging hooks), trunk centroids remain separated beyond $d_{\text{thresh}}$ while distal wrists/ankles breach the opponent's head/torso box; conversely, benign pedestrians passing side-by-side satisfy $d_{\text{thresh}}$ unless gated by both limb-to-target reach and box overlap ($\text{IoU} > 0$). Requiring both invariants simultaneously makes Edge Vision's contact filter geometrically distinct from center-of-mass distance thresholds.

---

## IV. Dual-Tier Deployment Architecture

- **Cloud Batch Microservice (Standard FastAPI REST API on 512 MB RAM):** Operates as a standard REST API (`POST /api/analyze_video`) receiving uploaded video clips for server-side CPU inference inside strict 512 MB RAM budgets by bypassing PyTorch (`HAS_TORCH=False`), pinning `intra_op_threads=1`, enforcing magic-byte container validation, and enforcing per-IP sliding-window throttling.
- **Browser Edge Engine (Zero-Upload Client WebAssembly / WebGPU):** Only client-side Edge Mode (alongside local on-premise Python OpenCV feeds) guarantees strict zero-upload video privacy—raw frames never leave browser memory. Serving middleware injects `COOP: same-origin`, `COEP: require-corp`, and `CORP: cross-origin` HTTP headers unlocking `SharedArrayBuffer` 4-thread WASM SIMD (`WASM-SIMD-4T`) and WebGPU acceleration, paired with pre-flight `canPlayType()` codec validation and frame-seek error recovery.

---

## V. Experimental Results

### A. Dataset Preprocessing & Ordered Source-ID Split (1,951 Clips)
The original *Real Life Violence Situations* (RLVS) dataset [5] contains 2,000 videos (1,000 Violent, 1,000 Non-Violent). Prior to manifest construction, **49 Non-Violent legacy AVI clips** (`NV_602.avi` and `NV_863.avi`–`NV_910.avi`) were excluded because the evaluation harness and client MSE pipeline standardize deterministically on H.264/MP4 streams (0 Violent clips were dropped). The remaining **1,951 verified MP4 clips** retain near-balanced priors (**1,000 Violent [51.3%]**, **951 Non-Violent [48.7%]**) partitioned by source-video ID into a frozen **70/15/15 split** (`splits_manifest.json`: 1,365 Train / 293 Val / 293 Test totaling 45,754 frames; 150 V, 143 NV; verified `Train ∩ Test = Val ∩ Test = ∅`), partitioned via an ordered source-ID split to prevent source-level data leakage. All components are unbundled cleanly below (`pose_stride=2`, threshold `0.65`):

| Configuration | Precision | Recall | Specificity | F1 Score | FP / 143 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline ST-GCN | 69.86% | 97.33% | 55.94% | 0.813 | 63 (44.1%) |
| ST-GCN + Kalman Tracking (only) | 69.19% | 97.33% | 54.55% | 0.809 | 65 (45.5%) |
| ST-GCN + Naive Centroid Gate (`d <= 0.22W`) | 69.86% | 97.33% | 55.94% | 0.813 | 63 (44.1%) |
| ST-GCN + Relational Strike Gate (only) | 70.19% | 97.33% | 56.64% | 0.816 | 62 (43.4%) |
| ST-GCN + Tracking + Naive Centroid Gate | 71.22% | 97.33% | 58.74% | 0.823 | 59 (41.3%) |
| ST-GCN + Tracking + Strike Gate (Cloud Config) | 70.53% | 97.33% | 57.34% | 0.818 | 61 (42.7%) |
| **Full System (+ MobileNetV3 Scene Gating)** | **76.04%** | **97.33%** | **67.83%** | **0.854** | **46 (32.2%)** |

Confusion Matrix (`TP=146, FP=46, FN=4, TN=97`):
- **Accuracy:** $(146 + 97) / 293 = \mathbf{82.94\%}$
- **Precision:** $146 / (146 + 46) = \mathbf{76.04\%}$
- **Recall:** $146 / (146 + 4) = \mathbf{97.33\%}$
- **Specificity:** $97 / (97 + 46) = \mathbf{67.83\%}$
- **F1 Score:** $2 \times 146 / (292 + 46 + 4) = \mathbf{0.854}$

**Stratified Error Analysis (Strike Gate vs. Centroid Gate):** In aggregate across the 293 clips, Naive Centroid Gating paired with Kalman tracking (`59` FPs, `F1=0.823`) slightly edges out Relational Strike Gating (`61` FPs, `F1=0.818`). We hypothesize that the Relational Strike Gate excels in wide-limb and partial-occlusion edge cases, though dedicated quantitative subgroup evaluation is required for future work. Specifically, the Relational Strike Gate functions as a *specialized geometric safety defense rather than a universal precision silver bullet*. On dense benign clips where actors stand outside $0.22 W_{\text{frame}}$ torso-center radius while swinging hands cross adjacent box edges, scalar centroid thresholding rejects 2 additional FPs. Conversely, Naive Centroid Gating fails systematically on critical boundary regimes such as: (i) wide-limb strikes (high kicks or lunging punches) whose body centers remain separated beyond $0.22 W_{\text{frame}}$ while extremities contact the target, and (ii) partial occlusions shifting truncated bounding-box centroids away while anatomical keypoints overlap.

### B. Hardware Latency Variance, Percentiles & Stress Benchmark
Measured on an **11th Gen Intel Core i5-1135G7 @ 2.40 GHz CPU (4 cores / 8 threads, ONNX Runtime CPU) over 100 timed iterations preceded by a 10-iteration warm-up run** on `violence_sample.mp4`, explicitly including OpenCV `cv2.VideoCapture.read()` video decoding and YOLOv8n-Pose extraction (reporting throughput strictly as $\text{FPS} = 1000 / \mu_{\text{ms}}$):
- **Strided Edge/Local (`pose_stride=2`, 4 threads w/ decode):** Mean **`22.04 ± 14.87 ms`**, Median P50 = **`24.05 ms`**, P95 = **`48.52 ms`**, **`45.4 processed-frame throughput (FPS)`** (process-only: `16.32 ± 14.11 ms`, P50 = `14.82 ms`, P95 = `38.55 ms`, **`61.3 FPS`**). Because `pose_stride=2` is used (alternating frames bypass YOLOv8n-Pose extraction), this throughput reflects the model's processing speed rather than the source-video ingestion rate.
- **Strided Cloud (`pose_stride=2`, 1 thread w/ decode):** Mean `28.74 ± 21.20 ms`, P50 = `34.36 ms`, P95 = `58.99 ms`, `34.8 FPS`.
- **Uniform Per-Frame (`pose_stride=1`, 4 threads w/ decode):** Mean `37.55 ± 14.59 ms`, P50 = `32.36 ms`, P95 = `67.53 ms`, `26.6 FPS` (stage means: OpenCV decode `6.38 ± 8.35 ms`, YOLOv8 `29.17 ± 6.25 ms`, Kalman `0.44 ± 0.13 ms`, Normalization `2.09 ± 0.43 ms`, ST-GCN `0.91 ± 0.59 ms`, Strike Gate `0.62 ± 0.17 ms`).

To evaluate continuous temporal transition stability, a separate live-stream stress test (`benchmark_tests.py`) concatenating a 66-frame non-violent clip (`NV_1.mp4`) into a 103-frame violent clip (`V_1.mp4`) recorded **0 / 66 (0.0%) false-alarm frames** and `1.0000` precision across the altercation boundary; this frame-level stress test is distinct from the 293-video held-out test set evaluation. Local Node.js `onnxruntime-web` WASM parity testing verifies that browser WASM inference successfully reproduces the expected classification outputs on two regression test clips, demonstrating functional parity with the Python backend (`peak=0.727` on `violence_sample.mp4` and `peak=0.217` on `safe_sample.mp4`).

### C. Operational Limitations & Failure Mode Analysis
Inspecting misclassified held-out clips reveals clear physical boundaries. The system achieves a 67.83% specificity, meaning approximately 32% of non-violent test videos trigger false positives. In real-world continuous surveillance, this false-alarm rate could contribute to alert fatigue, necessitating further temporal smoothing. Specifically, the 46 residual false positives under Full System execution cluster predominantly in intense athletic sparring, close-quarters wrestling, and choreographed martial-arts routines where fast limb contact satisfies both geometric invariants ($d_{\text{strike}} \le \alpha \cdot W_{\text{frame}}$ and $\text{IoU} > 0$) despite benign recreational context. When local MobileNetV3 scene classification detects an athletic arena, tightening $(\alpha, \tau)$ to $(0.12, 0.85)$ eliminates 15 athletic false alarms (`61 -> 46` FPs), whereas memory-capped 512 MB Cloud Mode deployments disable scene gating (`enable_scene_gating=False`) to preserve container stability. This measured gain results from the combined effect of the MobileNetV3 classification and its associated threshold adjustments ($\alpha=0.12, \tau=0.85$). Future ablations are required to decouple the classifier's isolated contribution from the threshold shift.

Second, the 4 false negatives stem from extreme motion blur, severe low-light silhouette degradation, or heavy multi-person overlap that suppresses YOLOv8n-Pose box confidences below detection thresholds prior to skeletal graph evaluation. Finally, browser Edge Mode decoding remains governed by client HTML5 video container codecs (native H.264/AVC1 MP4 and VP8/VP9/AV1 WebM); pre-flight `canPlayType()` verification automatically identifies unsupported codecs such as HEVC/H.265 and directs operators seamlessly to the Cloud REST API fallback.

---

## VI. Conclusion

Edge Vision delivers **82.94% Accuracy**, **76.04% Precision**, **97.33% Recall**, **67.83% Specificity**, and an **F1 Score of 0.854** across 293 held-out test videos while sustaining **45.4 processed-frame throughput (FPS)** end-to-end (`22.04 ± 14.87 ms`, P50 = `24.05 ms`, P95 = `48.52 ms`; **61.3 FPS** process-only; representing model processing speed under `pose_stride=2` rather than source-video ingestion rate) on commodity laptop CPUs. Browser Edge Mode guarantees zero-upload client execution via Cross-Origin-Isolated WebAssembly and WebGPU, complementing the 512 MB Cloud REST API. Future work targets INT8 pose quantization for low-power ARM SoCs and multi-camera RTSP re-identification.

---

## References

[1] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. ICCV*, 2015.  
[2] S. Yan, Y. Xiong, and D. Lin, "Spatial temporal graph convolutional networks for skeleton-based action recognition," in *Proc. AAAI*, 2018.  
[3] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8 Pose Estimation," 2023. [Online]. Available: https://docs.ultralytics.com/tasks/pose/  
[4] H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval Res. Logist. Quart.*, vol. 2, no. 1–2, pp. 83–97, 1955.  
[5] M. M. Soliman et al., "Violence recognition from videos using deep learning techniques," in *Proc. IEEE ICICIS*, 2019.  
[6] M. Cheng, K. Cai, and M. Li, "RWF-2000: An open large scale video database for violence detection," in *Proc. ICPR*, 2021.  
[7] P. Wu et al., "Not only look, but also listen: Learning multimodal violence detection under weak supervision," in *Proc. ECCV*, 2020.  
[8] B. M. Degardin and H. Proença, "Human activity analysis: Iterative weak/self-supervised learning frameworks for detecting abnormal events," in *Proc. IEEE IJCB*, 2020.  
[9] M. Gohar, "Violence detection enhancement by involving convolutional block attention modules into various deep learning architectures: Comprehensive case study for UBI-Fights dataset," *IEEE Access*, 2023.  
[10] L. Shi, Y. Zhang, J. Cheng, and H. Lu, "Two-stream adaptive graph convolutional networks for skeleton-based action recognition," in *Proc. CVPR*, 2019.  
[11] Z. Liu, H. Zhang, Z. Chen, Z. Wang, and W. Ouyang, "Disentangling and unifying graph convolutions for skeleton-based action recognition," in *Proc. CVPR*, 2020.  
[12] Y. Chen, Z. Zhang, C. Yuan, B. Li, Y. Deng, and W. Hu, "Channel-wise topology refinement graph convolution for skeleton-based action recognition," in *Proc. ICCV*, 2021.  
[13] Y.-F. Song, Z. Zhang, C. Shan, and L. Wang, "Constructing stronger and faster baselines for skeleton-based action recognition," *IEEE TPAMI*, 2022.  
[14] H. Duan, Y. Zhao, K. Chen, D. Lin, and B. Dai, "Revisiting skeleton-based action recognition," in *Proc. CVPR*, 2022.  
[15] H. Duan, J. Wang, K. Chen, and D. Lin, "PYSKL: Towards good practices for skeleton action recognition," in *Proc. ACM MM*, 2022.  
[16] H.-g. Chi, M. H. Ha, S. Chi, S. W. Lee, Q. Huang, and K. Ramani, "InfoGCN: Representation learning for human skeleton-based action recognition," in *Proc. CVPR*, 2022.  
[17] J. Lee, M. Lee, D.-L. Nguyen, and S. Lee, "Hierarchically decomposed graph convolutional networks for skeleton-based action recognition," in *Proc. ICCV*, 2023.  
[18] J. Do and M. Kim, "SkateFormer: Skeletal-temporal transformer for human action recognition," in *Proc. ECCV*, 2024.  
[19] Y. Zhou, X. Yan, Z.-Q. Cheng, Y. Yan, Q. Dai, and X.-S. Hua, "BlockGCN: Redefine topology awareness for skeleton-based action recognition," in *Proc. CVPR*, 2024.  
[20] H. Yang et al., "Multi-scale and attention enhanced graph convolution network for skeleton-based violence action recognition," *Frontiers in Neurorobotics*, 2022.  
[21] N. F. Janbi, M. A. Ghaseb, and A. A. Almazroi, "ESTS-GCN: An ensemble spatial-temporal skeleton-based graph convolutional networks for violence detection," *Int. J. Intell. Syst.*, 2024.  
[22] N. Tran, H. Nguyen, D. Ly, and H. D. Nguyen, "Advancing violence detection with graph-based skeleton motion analysis," *SN Computer Science*, 2025.  
[23] D. Maji, S. Nagaraj, et al., "YOLO-Pose: Enhancing YOLO for multi person pose estimation," in *Proc. CVPRW*, 2022.  
[24] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," in *Proc. ECCV*, 2022.  
[25] N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, "BoT-SORT: Robust associations multi-pedestrian tracking," *arXiv:2206.14651*, 2022.  
[26] ONNX Runtime Developers, "ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator," 2024. https://onnxruntime.ai/  
[27] A. Howard et al., "Searching for MobileNetV3," in *Proc. ICCV*, 2019.
