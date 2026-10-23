# Pose-Based Spatio-Temporal Violence Detection at the Edge

**IEEE-style engineering report**
**Authors:** Engineering Research Team  
**Affiliation:** Applied Systems & Vision Group  
**Artifact:** Kaggle-to-ONNX training pipeline, FastAPI service, and Next.js monitor

## Abstract

This work describes an edge-oriented violence classification system that combines YOLOv8 pose estimation with a compact, single-layer LSTM. Each frame contributes the normalized x/y coordinates of 17 COCO body keypoints. A rolling 30-frame window forms a 30-by-34 input sequence. The classifier returns a fighting probability; the local-edge service requires three consecutive results above 0.75 before raising an event. The Render Blueprint pins compute-library threads to one, uses pose stride two, and retains a 0.85 probability threshold. ONNX Runtime performs both pose and temporal inference. An asyncio producer-consumer path bounds queued frames to 30 and caps local ingestion at 15 FPS.

The pipeline was run on 2,000 unique videos from the downloaded dataset, producing balanced 30-frame pose sequences and a trained ONNX artifact. On a stratified 400-video test split, raw `> 0.65` thresholding yielded 0.944 precision, 0.670 recall, and 0.784 F1; the prior `> 0.85` threshold yielded 1.000 precision, 0.330 recall, and 0.496 F1. In a simulation using extracted held-out keypoints, stride two changed precision/recall/F1 at `> 0.75` from 0.952/0.500/0.656 to 0.971/0.495/0.656. A local 66-frame CPU test averaged 31.98 ms per frame at stride one and 16.75 ms at stride two. These measurements are limited: the sequence simulation is not end-to-end validation, and the timing test does not represent Render's 0.1 vCPU. The 22 older local incident records predate the LSTM and contain no FPS or inference-latency fields.

**Index Terms—** violence detection, human pose estimation, LSTM, ONNX Runtime, edge inference, video analytics.

## I. Introduction

Violence monitoring has to run close to the camera when network links are constrained or a site cannot send video to a remote service. Large video classifiers are not always a practical fit for these deployments. Pose coordinates provide a smaller temporal representation: instead of storing pixel clips, the system evaluates a short sequence of body-joint locations.

This project uses a YOLOv8 pose model for per-frame keypoints and a 64-unit LSTM for temporal classification. The design aims to keep the temporal model small enough for CPU inference while retaining movement across frames. It does not establish that keypoint motion alone can distinguish every fight from sports, hurried movement, or camera shake. That question requires evaluation on labeled video.

## II. System Design

### A. Input Sources and Frame Timing

The ingestion process accepts local webcam indices, RTSP/RTSPS URLs, and local video paths. OpenCV reads and resizes frames before they enter a bounded `asyncio.Queue(maxsize=30)`. When full, the queue drops the oldest pending frame. Local and Render ingestion are capped at 15 FPS. Inference runs off the event loop on a worker thread.

Local recordings are not decoded as fast as the hardware permits. The producer targets up to 15 frames per second and sleeps for the remaining frame interval after accounting for read and queue time. This cap is aligned with the approximately 15 FPS measured in one offline CPU smoke test, not a guarantee of sustained inference throughput. If inference falls behind, pending frames are discarded after the 30-frame buffer fills. This cannot reconstruct original capture timing when a source has a variable frame rate or unreliable metadata.

### B. Pose Sequence and Classifier

For each frame, the exported YOLOv8-Pose ONNX model estimates 17 keypoints using a 320-by-320 input. The backend decodes its 56-by-2,100 output, applies confidence filtering and non-maximum suppression, then selects the highest-confidence person. Keypoints are mapped through the letterbox transform, scaled to [0, 1], and flattened into 34 features. Missing detections are represented by zeros. The latest 30 feature vectors form the classifier input:

$$
X_t \in \mathbb{R}^{30 \times 34}.
$$

The recurrent layer is a one-layer, batch-first LSTM with input dimension 34 and hidden dimension 64. Its final time-step representation passes through a linear layer and sigmoid to produce a probability $p_t$. The local-edge service uses a default threshold of 0.75 and sets `is_violence` only after three consecutive processed frames exceed it; a below-threshold result resets the counter. The Render Blueprint retains its stricter `VIOLENCE_THRESHOLD=0.85` override. Pose inference defaults to every second frame. Intermediate coordinates reuse the latest detected box and extrapolate the keypoints by half the estimated per-frame velocity. This reduces pose-model calls, but modifies the classifier input and is not equivalent to measuring every frame.

Training uses binary cross-entropy with logits, Adam, and a stratified 80/20 train/test split. The training script reports held-out precision, recall, and F1 at strict $p_t > 0.75$, then exports a probability-output ONNX model. These per-sequence reports do not model the online three-result debounce.

### C. Persistence and User Interface

FastAPI manages ingestion, inference, upload handling, incident persistence, and `/ws/stream`. WebSocket frame messages contain a base64 JPEG, `is_violence`, probability, FPS, and inference time. Preview frames are resized to at most 480 pixels wide and encoded at JPEG quality 60 to reduce transfer size. The frontend offers webcam, MP4 upload, and RTSP controls. The status badge changes from green SAFE to a red violence alert when the message flag is true.

Positive predictions are written to SQLite and JSON Lines with timestamps, severity, probability, confidence, FPS, and measured inference latency. A cooldown limits repeated log rows during a continuous event. Uploads are size-limited, stored temporarily, and played once; switching sources resets the LSTM history.

## III. Training Data and Reproducibility

The download script targets the Kaggle dataset `mohamedmustafa/real-life-violence-situations-dataset`. For this run, the user supplied `archive.zip`; its two top-level dataset copies had matching sizes and CRCs for all 2,000 video paths, so only one copy was used. The extracted set contains 1,000 Violence and 1,000 NonViolence videos. Dataset preparation uses a thread-local YOLO instance in a bounded thread pool. It decodes each video sequentially, batches its 30 sampled frames for pose inference, and writes sequences in deterministic source-path order to `X_train.npy` and `y_train.npy`.

One sequence is produced per source video, which keeps clips from the same source together for the provided sample-level split. If the dataset contains near-duplicate or related videos, a source-aware grouping audit is still needed to rule out train/test leakage. Extraction failures are reported and skipped; the script fails if no labeled videos can be processed.

Training used 30 epochs, batch size 64, seed 42, and CPU-only PyTorch on an Intel Core i5-1135G7. The split was stratified by video and contained 1,600 training and 400 test sequences, with 200 examples of each class in the test set. The split is repeatable because input sequences are saved in sorted source-path order. Related-video leakage and performance on an independent dataset have not been evaluated.

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

The trained LSTM results below use its held-out split and report raw per-sequence threshold comparisons only. They do not evaluate the three-result debounce; the earlier `> 0.85` and reference 0.50 rows provide context for the measured threshold trade-off.

| Pose-LSTM test or runtime measurement | Result |
|---|---:|
| Unique videos used | 2,000 (1,000 per class) |
| Held-out sequences | 400 (200 per class) |
| Precision / recall / F1 at raw `> 0.65` (no debounce) | 0.944 / 0.670 / 0.784 |
| Precision / recall / F1 at raw `> 0.85` (no debounce) | 1.000 / 0.330 / 0.496 |
| Precision / recall / F1 at `>= 0.50` | 0.847 / 0.775 / 0.809 |
| Stride-one held-out precision / recall / F1 at `> 0.75` | 0.952 / 0.500 / 0.656 |
| Stride-two keypoint-simulation precision / recall / F1 at `> 0.75` | 0.971 / 0.495 / 0.656 |
| Single-clip mean processing time, stride one / two | 31.98 / 16.75 ms per frame |
| Historical no-stride clip runtime | 66 frames, `NV_1.mp4`; 15.89 wall-clock FPS |
| Historical no-stride mean / p95 pose-plus-ONNX call time | 57.00 / 53.49 ms |
| Historical incident rows with FPS/latency | 0 of 22 |

Threshold metrics are computed from the existing ONNX classifier on the same deterministic held-out split: at `> 0.65`, confusion counts are 192 true negatives, 8 false positives, 66 false negatives, and 134 true positives. Lowering the threshold raises recall by 0.34 over 0.85, while precision falls from 1.000 to 0.944. For the stride experiment, the held-out pose arrays were transformed to mimic causal stride-two processing (detect every other frame and extrapolate skipped points from prior observed velocity), then classified sequence by sequence. At the Render threshold of `> 0.85`, precision remained 1.000 in this split, but accuracy decreased from 0.665 to 0.6475 and recall from 0.330 to 0.295; therefore the experiment does not support a claim of unchanged classification behavior at that setting. At `> 0.75`, precision/recall/F1 changed from 0.952/0.500/0.656 to 0.971/0.495/0.656. This simulation does not include YOLO errors, nearest-neighbor image resizing, frame-drop patterns, or the three-result debounce.

The updated runtime timing check processed 66 decoded frames from one non-violent clip using a local CPU, one ONNX thread, nearest-neighbor pose resizing, and no websocket encoding. Mean per-frame time was 31.98 ms at stride one and 16.75 ms at stride two; the detector ran pose inference on 66 and 33 frames, respectively. The earlier separate 66-frame test reported 15.89 wall-clock FPS and 57.00 ms mean pose-plus-classifier time. These are single-clip local measurements, not Render 0.1-vCPU benchmarks. Render ingestion is now capped at 15 FPS; this cap does not guarantee the service sustains that inference rate. Results from one split and one clip do not establish performance on other cameras, scenes, or populations. The historical velocity-heuristic rows and 130-frame synthetic-motion benchmark remain separate from these LSTM measurements.

## V. Deployment and Limitations

The Render Docker service binds to the platform-provided port (default 10000) and starts without a camera source. Both `backend/yolov8n-pose.onnx` and `backend/violence_lstm.onnx` must be present in the image. Runtime dependencies exclude PyTorch and Ultralytics; training-only packages are listed separately. The free service has limited CPU and ephemeral local storage, so its incident database and uploaded files are not durable across restarts.

The current feature vector contains coordinates but not keypoint confidence, body scale, or explicit person tracking. Selecting the highest-confidence person independently in each frame may switch identities when several people are present. Zero-filling missing poses also creates a signal the model may not see often enough during training. These choices need targeted validation and may need to change before a safety-critical deployment.

The dataset split and reported metrics should be checked for subject/camera leakage, class imbalance, and threshold calibration. Field testing should include nonviolent high-motion activity, occlusion, poor lighting, camera movement, and multiple people. This system is a screening aid, not a substitute for human review or emergency response procedures.

## VI. Conclusion

The repository defines a route from Kaggle video to normalized pose sequences, a compact LSTM, two-model ONNX inference, hybrid ingestion, event logging, and a live dashboard. The low-resource runtime avoids PyTorch and Ultralytics, pins compute-library threads, and uses stride-two pose inference; Render and local ingestion are capped at 15 FPS. Dataset and single-clip results are reported with their limitations; this model is not yet independently validated or characterized on Render hardware.

## References

[1] M. Mustafa, “Real Life Violence Situations Dataset,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
[2] Ultralytics, “Pose Estimation.” [Online]. Available: https://docs.ultralytics.com/tasks/pose/
[3] PyTorch, “LSTM.” [Online]. Available: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
[4] ONNX Runtime, “Python API.” [Online]. Available: https://onnxruntime.ai/docs/api/python/api_summary.html
