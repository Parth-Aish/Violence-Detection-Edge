# Edge-Optimized Spatio-Temporal Violence Detection for Legacy Surveillance Infrastructure

**Authors:** Engineering Research Team  
**Affiliation:** Applied Systems & Vision Group  
**Target:** Systems & Computer Vision Engineering Track  
**Code & Data:** `backend/data/` (benchmark summary and JSONL incident log)

---

## Abstract

This paper describes an edge-oriented violence-monitoring pipeline for existing cameras and uploaded video. It combines YOLOv8n-pose with a temporal velocity heuristic and accepts RTSP streams, a local webcam, or MP4 playback. A one-frame queue bounds ingestion backlog, and local files are paced to the configured frame rate so their motion is not evaluated as an artificial high-speed stream.

The checked-in CPU benchmark summary reports 65.43 ms mean pose latency, 0.095 ms mean heuristic latency, 66.95 ms mean pipeline latency, and 14.9 frames per second across a 130-frame controlled run. Its JSONL incident log contains 11 records: six CRITICAL, four HIGH, and one MEDIUM. The harness constructs synthetic keypoint sequences for its motion scenarios; these figures measure implementation behavior under controlled inputs, not detection accuracy on independently labeled surveillance footage.

---

## 1. The Deployment Problem in Real Surveillance

Many surveillance deployments already expose an RTSP stream, but a prototype or test setup may instead use a USB webcam or a short local clip. A useful monitoring service needs to handle each source without changing the inference interface. This implementation uses 2D pose estimation and frame-to-frame kinematics to keep the temporal stage small; the benchmark here measures runtime and logging behavior rather than comparative recognition accuracy.

| Architecture | Model Parameters | GFLOPs (Clip) | Indicative Compute Target | Reported Frame Latency |
| :--- | :--- | :--- | :--- | :--- |
| **SlowFast R-101** | 53.8M | 213.0 | RTX 2080 / T4 | ~120 ms |
| **I3D (RGB + Flow)** | 25.0M | 108.0 | GTX 1080 Ti | ~95 ms |
| **TimeSformer-L** | 121.4M | 196.0 | A100 / RTX 3090 | ~80 ms |
| **Our System (YOLOv8n-pose + Heuristic)** | **3.2M** | **~3.3** | **None (Pure CPU)** | **66.95 ms** |

The comparison values above are indicative figures drawn from prior reports, not measurements from a matched experiment. Only the final row is measured by the benchmark described in this paper.

The method separates pose extraction from temporal classification. YOLOv8n-pose supplies joint coordinates; the classifier evaluates the movement of selected limbs over a short history and uses proximity as an escalation cue. This design is inexpensive to compute, but its limits are important: fast motion is only a proxy for violence and can produce false alarms.

---

## 2. System Architecture

The pipeline consists of three asynchronous stages designed around one core rule: **never let camera latency block the inference loop, and never let slow inference buffer frames in memory.**

```
┌─────────────────┐      raw bytes      ┌──────────────────────┐
│ RTSP / MP4      ├────────────────────►│ FFmpeg Subprocess    │
│ Camera / Upload │                     │ (paced local files)  │
└─────────────────┘                     └──────────┬───────────┘
                                                   │ Raw BGR
                                                   ▼
                                        ┌──────────────────────┐
                                        │ Multiprocessing      │
                                        │ Queue (maxsize=1)    │
                                        └──────────┬───────────┘
                                                   │ Drop old frames
                                                   ▼
┌─────────────────┐    WebSocket JSON   ┌──────────────────────┐
│ Next.js Web UI  │◄────────────────────┤ Main FastAPI Server  │
│ (Live Monitor)  │   Base64 JPEGs      │ (asyncio event loop) │
└─────────────────┘                     └──────────┬───────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        │                                                     │
                        ▼                                                     ▼
             ┌─────────────────────┐                               ┌─────────────────────┐
             │ YOLOv8n-Pose Engine │                               │ Dual-Sink Logger    │
             │ + Velocity Math     │                               │ SQLite WAL + JSONL  │
             └─────────────────────┘                               └─────────────────────┘
```

### 2.1 Source-specific ingestion with a bounded queue

RTSP streams and MP4 files use an isolated producer process that reads FFmpeg raw frames. The local webcam uses OpenCV's `cv2.VideoCapture(0)` in that producer. A FastAPI upload endpoint stores MP4 input in a temporary file and replaces the active producer; a webcam endpoint can switch back to camera index zero. The consumer receives the same BGR frame format from every source.

For RTSP input, the producer launches FFmpeg with low-latency flags:

```bash
ffmpeg -rtsp_transport tcp -i <STREAM_URL> \
       -an -sn -dn \
       -vf scale=640:480,fps=15 \
       -pix_fmt bgr24 -vcodec rawvideo \
       -fflags nobuffer -flags low_delay \
       -f rawvideo pipe:1
```

The worker reads exactly `640 * 480 * 3 = 921,600` bytes per frame from stdout. For local video, FFmpeg converts frames to the configured resolution and rate. The producer sleeps for only the remaining portion of each target frame interval, keeping playback close to real time while compensating for time already spent reading and queueing the frame.

Frames are pushed to a `multiprocessing.Queue(maxsize=1)`. If inference is busy, the producer discards the stale frame and inserts the newest one rather than building an unbounded backlog. RTSP reconnection uses exponential backoff. Uploaded videos play once; the producer does not silently restart the clip at EOF.

---

## 3. Kinematic Velocity Heuristic

Rather than passing 16 or 32 frame stacks into a neural net, we track coordinate histories for 17 COCO keypoints per detected person across a 15-frame ring buffer.

### 3.1 Joint Selection

Violent physical acts rely on rapid limb movement. We isolate six specific joints:
- Left / Right Wrists (COCO indices 9, 10) for punches, open-hand strikes, and grabs.
- Left / Right Elbows (COCO indices 7, 8) for strikes and elbow drive.
- Left / Right Ankles (COCO indices 15, 16) for kicks and sudden lunges.

Torso and facial keypoints are ignored during velocity scoring because full-body translation (like someone running past the camera) shouldn't automatically trigger a violence alarm.

### 3.2 Velocity and Noise Filtering

For every tracked joint $j$ between frame $t$ and frame $t-1$, we compute the pixel displacement:

$$v_j^t = \sqrt{(x_j^t - x_j^{t-1})^2 + (y_j^t - y_j^{t-1})^2}$$

We discard any joint where detection confidence drops below 0.3 in either frame. We take the maximum velocity among confident joints:

$$V_{max}^t = \max_{j \in \mathcal{J}} v_j^t$$

Because 2D keypoint detectors suffer from single-frame spatial jitter, feeding raw $V_{max}^t$ directly into a threshold triggers frequent false positives. We smooth the velocity signal using an Exponential Moving Average (EMA):

$$\hat{V}^t = \alpha V_{max}^t + (1 - \alpha) \hat{V}^{t-1}$$

Through empirical testing against baseline movements, setting $\alpha = 0.4$ provides the right balance: it filters single-frame keypoint jumps while reacting within 2 to 3 frames of a genuine punch.

### 3.3 Escalation Tiers and Proximity Boosting

We map smoothed velocity $\hat{V}^t$ (measured in pixels/frame at 640x480 resolution) to four severity categories:

- **NONE** (< 18.0 px/frame): Normal human movement, walking, casual gestures.
- **LOW** (>= 18.0 px/frame): Agitated movement, aggressive arm waving.
- **MEDIUM** (>= 35.0 px/frame): Forceful shoving, rapid grabbing motions.
- **HIGH** (>= 55.0 px/frame): High-speed strikes, punching, kicking.
- **CRITICAL** (>= 80.0 px/frame): Extreme multi-limb velocity altercations.

We apply two critical heuristic constraints:

1. **Consecutive Frame Counter:** A person must sustain velocity above 18.0 px/frame for at least 3 consecutive frames (`min_trigger_frames = 3`) to trigger an alarm. If a person drops below threshold, the counter decrements gradually rather than resetting instantly, allowing the system to bridge brief micro-pauses between strikes.
2. **Proximity Boost:** Fast movement by a solitary individual (like someone waving to catch a bus or stretching) isn't violence. When two detected persons have bounding box centers closer than 150 pixels, the system elevates their severity tier by +1. An agitated shove (MEDIUM) in close quarters immediately escalates to HIGH.

---

## 4. Real-World Benchmarks & Evaluation

The checked-in benchmark summary identifies the test hardware only as a local x86_64 CPU machine with PyTorch CPU. The harness warms up YOLOv8n-pose, times 30 pose calls, and runs a separate 130-frame controlled kinematics sequence. It is not a field trial on recorded CCTV.

### 4.1 Latency Breakdown and Real Metrics

The 130-frame sequence contains 20 baseline-motion frames, 25 moderate-motion frames, 30 simulated close-proximity punching frames, 35 simulated sustained-strike frames, and 20 cooldown frames. The script supplies synthetic keypoints directly to the temporal classifier; it does not ask YOLO to recognize a real fight in those phases.

```
+-------------------------------------------------------------------------+
| Pipeline Component Latencies (130 Evaluated Frames)                     |
+---------------------------------------------------+---------------------+
| YOLOv8n-Pose Forward Pass (Mean)                  | 65.43 ms            |
| YOLOv8n-Pose Forward Pass (Median)                | 63.95 ms            |
| YOLOv8n-Pose Forward Pass (95th Percentile)       | 90.67 ms            |
| Kinematic Velocity Classifier (Mean)              |  0.095 ms (95 us)   |
| End-to-End Pipeline Latency (Mean)                | 66.95 ms            |
| End-to-End Pipeline Latency (95th Percentile)     | 74.31 ms            |
| Sustained Real-World Throughput                   | 14.90 FPS           |
+---------------------------------------------------+---------------------+
```

The benchmark shows that the temporal classifier is inexpensive relative to pose extraction: its reported mean is below one tenth of a millisecond. Pose estimation accounts for most measured inference time.

The reported 14.9 FPS is close to the configured 15 FPS target on this test machine. It should not be taken as a throughput guarantee for other CPUs, camera resolutions, or deployment conditions.

### 4.2 Incident Log Analysis

The checked-in `incidents.jsonl` contains 11 records written by the controlled test harness. Their severity counts and velocity statistics are:

| Severity Level | Event Count | Minimum Velocity (px/frame) | Mean Velocity (px/frame) | Peak Recorded Velocity (px/frame) | Primary Trigger Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CRITICAL** | 6 | 8.38 | 43.52 | 80.35 | Sustained strikes + Proximity Boost |
| **HIGH** | 4 | 17.54 | 55.62 | 93.70 | Fast punches & lunges |
| **MEDIUM** | 1 | 60.00 | 60.00 | 60.00 | Rapid arm extension |

The lowest velocity in the CRITICAL rows is 8.38 px/frame. This is possible because the classifier combines its smoothed movement score, consecutive-frame counter, and proximity escalation. It is a concrete example of the heuristic's behavior, not evidence that this threshold correctly identifies grappling in real footage. The log records classifier outputs; it contains no ground-truth labels or false-positive/false-negative counts.

The harness defines the first 20 frames as baseline walking motion. This controlled check exercises the consecutive-frame filter, but does not replace evaluation on independently labeled videos.

### 4.3 Storage and Memory Footprint

The checked-in benchmark summary does not contain memory, database-size, or WebSocket-bandwidth measurements, so no values are reported for those metrics. The single-slot queue bounds queued frames; separate long-duration memory and network tests remain necessary before making deployment capacity claims.

---

## 5. Engineering Constraints & Failure Modes

We don't claim this approach solves violence detection under all circumstances. Decomposing the problem this way introduces distinct physical trade-offs:

1. **Heavy Occlusion:** If two people grapple on the ground or fall behind a counter, YOLOv8-Pose loses extremity keypoints. When confidence drops below 0.3, velocity falls to zero, causing the system to lose track of the fight until limbs reappear.
2. **Global Camera Motion:** The math assumes stationary camera geometry. If a pan-tilt-zoom (PTZ) camera pans quickly, every static person appears to move at 40+ px/frame, triggering false alarms unless global motion compensation (like optical flow background subtraction) is applied.
3. **Weapon-Based Non-Kinematic Violence:** A person brandishing a firearm or knife often moves slowly and deliberately. Because our heuristic looks strictly for rapid limb movement, weapon threats without fast swinging motions won't trigger the velocity thresholds.

---

## 6. Conclusion

This implementation combines a 2D pose model, short-window velocity smoothing, and proximity-based escalation, with RTSP, webcam, and paced MP4 input paths. The recorded benchmark reports 14.9 FPS and 66.95 ms mean pipeline latency on its local CPU test machine. Those runtime numbers do not establish detection accuracy: that requires evaluation on a representative, labeled video set, including non-violent fast movement, occlusion, camera motion, and varied lighting. The current logs provide a reproducible check of the classifier and logger under synthetic motion inputs, not a substitute for that study.

---

## References

1. IHS Markit, "Video Surveillance Installed Base Report," Tech. Rep., 2024.
2. J. Carreira and A. Zisserman, "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset," *CVPR*, pp. 6299-6308, 2017.
3. C. Feichtenhofer, H. Fan, J. Malik, and K. He, "SlowFast Networks for Video Recognition," *ICCV*, pp. 6202-6211, 2019.
4. G. Bertasius, H. Wang, and L. Torresani, "Is Space-Time Attention All You Need for Video Understanding?" *ICML*, pp. 813-824, 2021.
5. S. Yan, Y. Xiong, and D. Lin, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," *AAAI*, vol. 32, no. 1, 2018.
6. Ultralytics, "YOLOv8 Real-Time Object Detection and Pose Estimation," https://github.com/ultralytics/ultralytics, 2023.
