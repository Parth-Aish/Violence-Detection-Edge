# Pose-Based Spatio-Temporal Violence Detection at the Edge

**IEEE-style engineering report**
**Authors:** Engineering Research Team  
**Affiliation:** Applied Systems & Vision Group  
**Artifact:** Kaggle-to-ONNX training pipeline, FastAPI service, and Next.js monitor

## Abstract

This work specifies an edge-oriented violence classification system that combines YOLOv8 pose estimation with a compact, single-layer LSTM. Each frame contributes the normalized x/y coordinates of 17 COCO body keypoints. A rolling 30-frame window forms a 30-by-34 input sequence. The classifier returns a fighting probability; the service raises an event when that probability exceeds 0.85. A bounded multiprocessing queue connects webcam, RTSP, or paced local-video ingestion to inference.

The pipeline was run on 2,000 unique videos from the downloaded dataset, producing balanced 30-frame pose sequences and a trained ONNX artifact. On a stratified 400-video test split, the configured `> 0.85` threshold yielded 1.00 precision, 0.33 recall, and 0.496 F1; at 0.50, the corresponding scores were 0.847, 0.775, and 0.809. A 66-frame offline CPU smoke test on one non-violent clip ran at 15.89 FPS, with 57.00 ms mean pose-plus-classifier time. These are limited dataset and single-clip measurements, not external or live-site validation. The 22 older local incident records predate the LSTM and contain no FPS or inference-latency fields.

**Index Terms—** violence detection, human pose estimation, LSTM, ONNX Runtime, edge inference, video analytics.

## I. Introduction

Violence monitoring has to run close to the camera when network links are constrained or a site cannot send video to a remote service. Large video classifiers are not always a practical fit for these deployments. Pose coordinates provide a smaller temporal representation: instead of storing pixel clips, the system evaluates a short sequence of body-joint locations.

This project uses a YOLOv8 pose model for per-frame keypoints and a 64-unit LSTM for temporal classification. The design aims to keep the temporal model small enough for CPU inference while retaining movement across frames. It does not establish that keypoint motion alone can distinguish every fight from sports, hurried movement, or camera shake. That question requires evaluation on labeled video.

## II. System Design

### A. Input Sources and Frame Timing

The ingestion process accepts local webcam indices, RTSP/RTSPS URLs, and local video paths. Webcam capture uses OpenCV; network streams and files use the FFmpeg-backed producer. A one-slot multiprocessing queue discards stale frames when the consumer falls behind, limiting queue growth.

Local recordings are not decoded as fast as the hardware permits. The producer targets 30 frames per second and sleeps for the remaining frame interval after accounting for read and queue time. This preserves approximately real-time spacing for the temporal sequence. It cannot reconstruct original capture timing when the source has a variable frame rate or unreliable metadata.

### B. Pose Sequence and Classifier

For each frame, YOLOv8-Pose estimates 17 keypoints at `imgsz=320` in both dataset preparation and online inference. The implementation selects the person with the highest detection confidence, scales x and y coordinates to the range [0, 1], and flattens them into 34 features. Missing detections are represented by zeros. The latest 30 feature vectors form the classifier input:

$$
X_t \in \mathbb{R}^{30 \times 34}.
$$

The recurrent layer is a one-layer, batch-first LSTM with input dimension 34 and hidden dimension 64. Its final time-step representation passes through a linear layer and sigmoid to produce a probability $p_t$. The service sets `is_violence` when $p_t > 0.85$. This threshold is a configured operating point, not a threshold selected from a completed validation study.

Training uses binary cross-entropy with logits, Adam, and a stratified 80/20 train/test split. The training script reports held-out precision, recall, and F1 at the same strict $p_t > 0.85$ operating threshold used by the service, then exports a probability-output ONNX model.

### C. Persistence and User Interface

FastAPI manages ingestion, inference, upload handling, incident persistence, and `/ws/stream`. WebSocket frame messages contain a base64 JPEG, `is_violence`, probability, FPS, and inference time. The frontend offers webcam, MP4 upload, and RTSP controls. The status badge changes from green SAFE to a red violence alert when the message flag is true.

Positive predictions are written to SQLite and JSON Lines with timestamps, severity, probability, confidence, FPS, and measured inference latency. A cooldown limits repeated log rows during a continuous event. Uploads are size-limited, stored temporarily, and played once; switching sources resets the LSTM history.

## III. Training Data and Reproducibility

The download script targets the Kaggle dataset `mohamedmustafa/real-life-violence-situations-dataset`. For this run, the user supplied `archive.zip`; its two top-level dataset copies had matching sizes and CRCs for all 2,000 video paths, so only one copy was used. The extracted set contains 1,000 Violence and 1,000 NonViolence videos. Dataset preparation uses a thread-local YOLO instance in a bounded thread pool. It decodes each video sequentially, batches its 30 sampled frames for pose inference, and writes sequences in deterministic source-path order to `X_train.npy` and `y_train.npy`.

One sequence is produced per source video, which keeps clips from the same source together for the provided sample-level split. If the dataset contains near-duplicate or related videos, a source-aware grouping audit is still needed to rule out train/test leakage. Extraction failures are reported and skipped; the script fails if no labeled videos can be processed.

Training used 30 epochs, batch size 64, seed 42, and CPU-only PyTorch on an Intel Core i5-1135G7. The split was stratified by video and contained 1,600 training and 400 test sequences, with 200 examples of each class in the test set. The split is repeatable because input sequences are saved in sorted source-path order. Related-video leakage and performance on an independent dataset have not been evaluated.

## IV. Local Logs and Benchmark Results

The local `backend/data/incidents.jsonl` was parsed at authoring time. It contains 22 records in the pre-LSTM velocity schema:

| Severity reported by the earlier logger | Records |
|---|---:|
| HIGH | 13 |
| CRITICAL | 7 |
| MEDIUM | 2 |
| **Total** | **22** |

None of those records contains an FPS or inference-latency field. They were written by the earlier pose-velocity implementation, not by the ONNX LSTM. The records describe classifier outputs and are not verified incident labels.

The checked-in `backend/data/benchmark_summary.json` is also from the earlier implementation. Its 130-frame controlled benchmark reports 65.43 ms mean pose latency, 0.095 ms heuristic latency, 66.95 ms mean pipeline latency, 74.31 ms p95 latency, and 14.9 FPS. The associated benchmark harness feeds synthetic keypoint sequences to the velocity classifier. These values must not be attributed to the LSTM, nor described as real-world violence accuracy.

The trained LSTM results below use its held-out split and the same `> 0.85` threshold as the backend. The 0.50 row is included to show the precision/recall trade-off; it is not the deployed setting.

| Pose-LSTM test or runtime measurement | Result |
|---|---:|
| Unique videos used | 2,000 (1,000 per class) |
| Held-out sequences | 400 (200 per class) |
| Precision / recall / F1 at `> 0.85` | 1.000 / 0.330 / 0.496 |
| Precision / recall / F1 at `>= 0.50` | 0.847 / 0.775 / 0.809 |
| Offline clip runtime | 66 frames, `NV_1.mp4`; 15.89 wall-clock FPS |
| Mean / p95 pose-plus-ONNX call time | 57.00 / 53.49 ms |
| Historical incident rows with FPS/latency | 0 of 22 |

The runtime check used direct frame-by-frame inference on one downloaded non-violent clip, without ingestion, WebSocket, or frontend overhead. Its mean includes a slow first-frame outlier, which is why the p95 is lower than the mean. Throughput was below the configured 30 FPS target on this CPU. The test metrics come from one split of one dataset; they do not establish performance on other cameras, scenes, or populations. The historical velocity-heuristic rows and 130-frame synthetic-motion benchmark remain separate from these LSTM measurements.

## V. Deployment and Limitations

The Docker image exposes port 7860 for Hugging Face Spaces. The ONNX model must be generated at `backend/violence_lstm.onnx` before building the production image. The backend checks for that artifact during startup and reports a clear error if it is absent.

The current feature vector contains coordinates but not keypoint confidence, body scale, or explicit person tracking. Selecting the highest-confidence person independently in each frame may switch identities when several people are present. Zero-filling missing poses also creates a signal the model may not see often enough during training. These choices need targeted validation and may need to change before a safety-critical deployment.

The dataset split and reported metrics should be checked for subject/camera leakage, class imbalance, and threshold calibration. Field testing should include nonviolent high-motion activity, occlusion, poor lighting, camera movement, and multiple people. This system is a screening aid, not a substitute for human review or emergency response procedures.

## VI. Conclusion

The repository now defines a complete route from Kaggle video to normalized pose sequences, a compact LSTM, ONNX inference, hybrid ingestion, event logging, and a live dashboard. The implementation is reproducible in structure, but the model has not yet been trained in this environment. No LSTM performance number is reported until data access, training, and target-device measurement have been completed.

## References

[1] M. Mustafa, “Real Life Violence Situations Dataset,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
[2] Ultralytics, “Pose Estimation.” [Online]. Available: https://docs.ultralytics.com/tasks/pose/
[3] PyTorch, “LSTM.” [Online]. Available: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
[4] ONNX Runtime, “Python API.” [Online]. Available: https://onnxruntime.ai/docs/api/python/api_summary.html
