# Pose-Based Spatio-Temporal Violence Detection at the Edge

**IEEE-style engineering report**
**Authors:** Engineering Research Team  
**Affiliation:** Applied Systems & Vision Group  
**Artifact:** Kaggle-to-ONNX training pipeline, FastAPI service, and Next.js monitor

## Abstract

This work describes an edge-oriented violence classification system that combines YOLOv8 pose estimation with a compact, single-layer LSTM. Each frame contributes 17 COCO body keypoints; shared preprocessing centers them on the hip or shoulder root, normalizes by torso scale, and blends posture with frame-to-frame motion into a 30-by-34 sequence. The runtime tracks multiple people and requires both an active multi-entity scene and a biomechanical score of at least 0.56 before the fused probability can trigger an alert at 0.58. ONNX Runtime performs pose and temporal inference. An asyncio producer-consumer path holds at most two queued frames and caps local ingestion at 15 FPS.

The training pipeline used 2,000 labeled videos and temporal-window augmentation. The final all-data model's stratified 400-sequence split measured 0.891 precision, 0.895 recall, and 0.893 F1 at `> 0.58`. A separate candidate excluded all 15 end-to-end validation clips during training and was tested with simulated 6 FPS wall-clock sampling. That candidate detected 6/8 violence clips and correctly rejected 6/7 non-violence clips; it detected V_1, V_2, and V_3 and raised no alerts on NV_1 or NV_2. The final production weights were subsequently retrained on all sequences, including those 15 clips, so the clip benchmark is not a held-out evaluation of the final artifact. The small clip set is not deployment certification. The 22 older incident records predate the LSTM and contain no FPS or inference-latency fields.

**Index Terms—** violence detection, human pose estimation, LSTM, ONNX Runtime, edge inference, video analytics.

## I. Introduction

Violence monitoring has to run close to the camera when network links are constrained or a site cannot send video to a remote service. Large video classifiers are not always a practical fit for these deployments. Pose coordinates provide a smaller temporal representation: instead of storing pixel clips, the system evaluates a short sequence of body-joint locations.

This project uses a YOLOv8 pose model for per-frame keypoints and a 64-unit LSTM for temporal classification. The design aims to keep the temporal model small enough for CPU inference while retaining movement across frames. It does not establish that keypoint motion alone can distinguish every fight from sports, hurried movement, or camera shake. That question requires evaluation on labeled video.

## II. System Design

### A. Input Sources and Frame Timing

The service accepts browser webcam frames, RTSP/RTSPS URLs, and local video paths. The browser uses `getUserMedia` and streams JPEG frames through `/ws/stream`; the cloud backend never tries to open a local hardware webcam. RTSP and uploaded-video frames are read and resized before entering a bounded `asyncio.Queue(maxsize=2)`. When full, that server-ingestion queue drops the oldest pending frame. Local and Render ingestion are capped at 15 FPS. Inference runs off the event loop on a worker thread.

Local recordings are synchronized to wall-clock time using their native frame-rate metadata. When inference falls behind playback, the producer skips stale frames with `grab()` and decodes only the current target frame. Capture remains capped at 15 FPS; this is not a guarantee of sustained inference throughput. If inference falls behind, the two-frame queue discards its oldest waiting frame. Wall-clock synchronization still depends on usable frame-rate metadata.

### B. Pose Sequence and Classifier

For each frame, the exported YOLOv8-Pose ONNX model estimates 17 keypoints using a 320-by-320 input. The backend decodes its 56-by-2,100 output, applies confidence filtering and non-maximum suppression, then tracks up to four people by nearest box center. Keypoints are mapped through the letterbox transform and scaled to [0, 1]. Shared preprocessing interpolates missing frames, centers the joints on the hip root (or shoulder root as a fallback), normalizes by torso scale, and combines 0.3 centered posture with 0.7 frame-to-frame change. The flattened 34 features form the classifier input:

$$
X_t \in \mathbb{R}^{30 \times 34}.
$$

The recurrent layer is a one-layer, batch-first LSTM with input dimension 34 and hidden dimension 64. Its final time-step representation passes through a linear layer and sigmoid to produce a probability $p_t$. The runtime scores the two most active tracks. A violence alert requires at least two people, or one person and a secondary entity, a biomechanical score of at least 0.56, and a fused score of at least 0.58. The fusion rule is `max(bio_score, 0.75 * bio_score + 0.25 * p_t)` above the biomechanical floor; below the floor, the reported probability is capped at `0.85 * bio_score`. A four-frame hold keeps qualifying alerts visible. Pose inference defaults to every second frame. Intermediate coordinates reuse the latest detected box and extrapolate keypoints from estimated motion.

Training uses binary cross-entropy with logits, Adam, and a stratified 80/20 train/test split. It applies the shared pose normalization and temporal-window augmentation: contiguous 10-to-20-frame slices are resampled to 30 frames. The training script reports held-out per-sequence precision, recall, and F1 at strict $p_t > 0.58$ and at 0.50. These classifier-only metrics do not model scene gating or the online fusion rule.

### C. Persistence and User Interface

FastAPI manages ingestion, inference, upload handling, incident persistence, and `/ws/stream`. For browser capture, the WebSocket reader runs alongside a frame processor. A one-frame queue replaces stale waiting images, while each response contains an annotated base64 JPEG, `is_violence`, probability, FPS, and inference latency. Preview frames are resized to at most 480 pixels wide and encoded at JPEG quality 60 to reduce transfer size. The frontend offers webcam, MP4 upload, and RTSP controls. It waits for an inference response before sending the next webcam frame, and stops all camera tracks when the source changes. The status badge changes from green SAFE to a red violence alert when the message flag is true.

Positive predictions are written to SQLite and JSON Lines with timestamps, severity, probability, confidence, FPS, and measured inference latency. A cooldown limits repeated log rows during a continuous event. Uploads are size-limited, stored temporarily, and played once; switching sources resets the LSTM history.

## III. Training Data and Reproducibility

The download script targets the Kaggle dataset `mohamedmustafa/real-life-violence-situations-dataset`. For this run, the user supplied `archive.zip`; its two top-level dataset copies had matching sizes and CRCs for all 2,000 video paths, so only one copy was used. The extracted set contains 1,000 Violence and 1,000 NonViolence videos. Dataset preparation uses a thread-local YOLO instance in a bounded thread pool. It decodes each video sequentially, batches its 30 sampled frames for pose inference, and writes sequences in deterministic source-path order to `X_train.npy` and `y_train.npy`.

One sequence is produced per source video, which keeps clips from the same source together for the provided sample-level split. If the dataset contains near-duplicate or related videos, a source-aware grouping audit is still needed to rule out train/test leakage. Extraction failures are reported and skipped; the script fails if no labeled videos can be processed.

Training used 30 epochs, batch size 64, seed 42, and CPU-only PyTorch. The stratified split contained 1,600 training and 400 test sequences, with 200 examples of each class in the test set. The split is repeatable because input sequences are saved in sorted source-path order. A calibration candidate was retrained with the 15 named runtime clips excluded, then scored end-to-end at 6 FPS before the production artifact was retrained on the full dataset. The final production artifact therefore includes those 15 videos; only the candidate benchmark is clip-held-out. Related-video leakage and performance on an independent dataset have not been evaluated.

## IV. Local Logs and Benchmark Results

The local `backend/data/incidents.jsonl` was parsed at authoring time before release cleanup. It contained 22 records in the pre-LSTM velocity schema; runtime log files are not included in the production repository:

| Severity reported by the earlier logger | Records |
|---|---:|
| HIGH | 13 |
| CRITICAL | 7 |
| MEDIUM | 2 |
| **Total** | **22** |

None of those records contains an FPS or inference-latency field. They were written by the earlier pose-velocity implementation, not by the ONNX LSTM. The records describe classifier outputs and are not verified incident labels.

The earlier implementation's 130-frame controlled benchmark reported 65.43 ms mean pose latency, 0.095 ms heuristic latency, 66.95 ms mean pipeline latency, 74.31 ms p95 latency, and 14.9 FPS. The harness fed synthetic keypoint sequences to the velocity classifier. These historical values are retained here for context only; they must not be attributed to the LSTM or described as real-world violence accuracy. The benchmark files are excluded from the production repository.

The current classifier-only metrics come from the final normalized model's stratified 400-sequence split. The separate clip-level results come from the calibration candidate trained with all 15 listed clips excluded; they do not describe the final all-data production weights.

| Pose-LSTM test or runtime measurement | Result |
|---|---:|
| Unique videos used | 2,000 (1,000 per class) |
| Held-out sequences | 400 (200 per class) |
| Final model precision / recall / F1 at raw `> 0.58` | 0.891 / 0.895 / 0.893 |
| Final model precision / recall / F1 at raw `>= 0.50` | 0.871 / 0.915 / 0.893 |
| Excluded-clip candidate: violence clips detected | 6 / 8 (recall 0.750) |
| Excluded-clip candidate: non-violence clips correctly clear | 6 / 7 (specificity 0.857) |
| Excluded-clip candidate: V_1, V_2, V_3 detected | 3 / 3 |
| Excluded-clip candidate: NV_1, NV_2 with no alerts | 2 / 2 |
| Excluded-clip candidate clip confusion matrix (TN, FP; FN, TP) | [[6, 1], [2, 6]] |
| Single-clip mean processing time, stride one / two | 31.98 / 16.75 ms per frame |
| Historical no-stride clip runtime | 66 frames, `NV_1.mp4`; 15.89 wall-clock FPS |
| Historical no-stride mean / p95 pose-plus-ONNX call time | 57.00 / 53.49 ms |
| Historical incident rows with FPS/latency | 0 of 22 |

The end-to-end benchmark sampled videos at six frames per second using native frame-rate metadata and `cap.grab()` to skip stale frames. A clip counted as detected if at least one frame raised the alert. The excluded-clip candidate detected the three named primary violence clips and cleared both named primary non-violence clips. Across all 15 clips, however, it missed two of eight violent clips and raised one false alert among seven non-violent clips. That is useful calibration evidence, not a broad accuracy claim. The final production model was retrained on all 2,000 sequences, including those 15 clips, so it has not had an independent 15-clip end-to-end test.

Earlier timing checks processed 66 decoded frames from one non-violent clip using a local CPU, one ONNX thread, and no websocket encoding. Mean per-frame time was 31.98 ms at stride one and 16.75 ms at stride two. These historical measurements are not benchmarks of the normalized model or Render 0.1-vCPU hardware. Render ingestion is capped at 15 FPS; this cap does not guarantee the service sustains that inference rate. The historical velocity-heuristic rows and 130-frame synthetic-motion benchmark remain separate from these LSTM measurements.

## V. Deployment and Limitations

The Render Docker service binds to the platform-provided port (default 10000) and starts without a camera source. Both `backend/yolov8n-pose.onnx` and `backend/violence_lstm.onnx` must be present in the image. Runtime dependencies exclude PyTorch and Ultralytics; training-only packages are listed separately. The free service has limited CPU and ephemeral local storage, so its incident database and uploaded files are not durable across restarts.

The runtime tracks up to four people by nearest box center, but crossings and occlusions can still cause identity switches. The low-confidence pose proposals and motion contours used as secondary entities are heuristics; they do not identify an animal class semantically. Some non-violent high-motion scenes can score as violence, and some violent clips remain undetected. The small excluded-clip benchmark and classifier-only random split do not establish production safety or generalization to new cameras and populations.

The dataset split and reported metrics should be checked for subject/camera leakage, class imbalance, and threshold calibration. Field testing should include nonviolent high-motion activity, occlusion, poor lighting, camera movement, and multiple people. This system is a screening aid, not a substitute for human review or emergency response procedures.

## VI. Conclusion

The repository defines a route from Kaggle video to normalized pose sequences, a compact LSTM, two-model ONNX inference, hybrid ingestion, event logging, and a live dashboard. The low-resource runtime avoids PyTorch and Ultralytics, pins compute-library threads, and uses stride-two pose inference; Render and local ingestion are capped at 15 FPS. The excluded-clip candidate met the requested V_1–V_3 and NV_1–NV_2 checks but had 0.75 violence recall and 0.857 non-violence specificity across the full 15-clip sample. The final production model was trained on those clips as part of the full dataset, and Render hardware behavior remains unmeasured. Results should be treated as preliminary engineering evidence.

## References

[1] M. Mustafa, “Real Life Violence Situations Dataset,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
[2] Ultralytics, “Pose Estimation.” [Online]. Available: https://docs.ultralytics.com/tasks/pose/
[3] PyTorch, “LSTM.” [Online]. Available: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
[4] ONNX Runtime, “Python API.” [Online]. Available: https://onnxruntime.ai/docs/api/python/api_summary.html
