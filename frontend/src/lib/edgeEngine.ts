/**
 * edgeEngine.ts
 * High-performance WebAssembly client-side Pose + ST-GCN Inference Engine.
 *
 * Implements the exact same pipeline as Python backend/core/inference.py:
 * 1. YOLOv8n-pose ONNX feature extraction with letterbox normalization
 * 2. Continuous primary-actor pose sequence buffer (75-frame rolling window)
 * 3. Uniform temporal linspace sampling (30 frames) for ST-GCN ONNX inference
 * 4. Calibrated logit mapping (logit + 0.55) & sigmoid activation
 * 5. Multi-person Relational Strike Gate (strike joints vs target body & bounding box)
 * 6. Temporal alert hold and consecutive-trigger confirmation
 */

import * as ort from "onnxruntime-web";
import {
  SEQUENCE_LENGTH,
  KEYPOINT_COUNT,
  FEATURE_COUNT,
  POSE_IMAGE_SIZE,
  PRIMARY_PERSON_CONFIDENCE,
  SECONDARY_PROPOSAL_CONFIDENCE,
  POSE_NMS_THRESHOLD,
  VIOLENCE_THRESHOLD,
  CONSECUTIVE_TRIGGERS_REQUIRED,
  ALERT_HOLD_FRAMES,
  POSE_STRIDE,
  Detection,
  BBox,
  rawToStgcnTensor,
  computeRelationalStrikeDistance,
  nms,
  boxIou,
  resetTrackIdCounter,
} from "./kinematics";

// ─── Types ────────────────────────────────────────────────────────────────────
export interface EdgePrediction {
  isFighting: boolean;
  probability: number;
  inferMs: number;
  personCount: number;
  secondaryCount: number;
  interactionMode: string;
  bioScore: number;
  turbulence: number;
  backend: string;
  actors: ActorInfo[];
}

export interface ActorInfo {
  bbox: BBox;
  /** normalised [0,1] keypoints */
  keypoints: Float32Array;
  trackId: number;
}

export type EngineState = "idle" | "loading" | "ready" | "error";

export interface LoadProgress { stage: string }

// ─── EdgeEngine ───────────────────────────────────────────────────────────────
export class EdgeEngine {
  private poseSession: ort.InferenceSession | null = null;
  private stgcnSession: ort.InferenceSession | null = null;
  private _backend = "—";
  private _state: EngineState = "idle";

  // Rolling temporal pose buffer (exact match of Python self._poses deque)
  private poses: Float32Array[] = [];
  private frameIndex = 0;

  // Alert hold and gating (exact match of Python inference.py)
  private consecutiveTriggers = 0;
  private alertHoldFrames = 0;

  // Cached detections across frame stride
  private cachedDetections: Detection[] = [];

  // Reusable canvas and buffers to eliminate garbage-collection pauses
  private letterboxCanvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  private letterboxCtx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null = null;
  private srcCanvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  private srcCtx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null = null;
  private inputTensorBuffer = new Float32Array(3 * POSE_IMAGE_SIZE * POSE_IMAGE_SIZE);

  get backend(): string { return this._backend; }
  get state(): EngineState { return this._state; }

  // ─── Init ───────────────────────────────────────────────────────────────────
  async load(
    onProgress?: (p: LoadProgress) => void,
  ): Promise<void> {
    this._state = "loading";
    try {
      onProgress?.({ stage: "Allocating Client WebAssembly SIMD Runtime..." });
      ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency ?? 4, 4);
      ort.env.wasm.simd = true;
      ort.env.wasm.wasmPaths = "/ort-wasm/";

      onProgress?.({ stage: "Compiling ONNX WASM Graph Kernels..." });
      const opts: ort.InferenceSession.SessionOptions = {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      };

      [this.poseSession, this.stgcnSession] = await Promise.all([
        ort.InferenceSession.create("/models/yolov8n-pose.onnx", opts),
        ort.InferenceSession.create("/models/violence_stgcn.onnx", opts),
      ]);

      this._backend = "WASM";
      onProgress?.({ stage: "Edge Models Ready" });
      this._state = "ready";
    } catch (err) {
      this._state = "error";
      throw err;
    }
  }

  reset(): void {
    this.poses = [];
    this.frameIndex = 0;
    this.consecutiveTriggers = 0;
    this.alertHoldFrames = 0;
    this.cachedDetections = [];
    resetTrackIdCounter();
  }

  // ─── Main inference entry ───────────────────────────────────────────────────
  async processFrame(frameData: ImageData): Promise<EdgePrediction> {
    if (!this.poseSession || !this.stgcnSession) {
      throw new Error("EdgeEngine not loaded");
    }
    const t0 = performance.now();
    const { width, height } = frameData;
    const isPoseFrame = this.frameIndex % POSE_STRIDE === 0;

    // 1. Pose extraction (strided for throughput)
    let detections: Detection[];
    if (isPoseFrame) {
      detections = await this._detectPose(frameData);
      this.cachedDetections = detections;
    } else {
      detections = this.cachedDetections;
    }
    this.frameIndex++;

    const personCount = detections.length;

    // 2. Select primary actor (highest confidence) & update poses deque (matches Python inference.py)
    if (personCount > 0) {
      let bestDet = detections[0];
      for (let i = 1; i < detections.length; i++) {
        if (detections[i].confidence > bestDet.confidence) {
          bestDet = detections[i];
        }
      }
      this.poses.push(bestDet.keypoints.slice());
    } else {
      if (this.poses.length > 0) {
        this.poses.push(this.poses[this.poses.length - 1].slice());
      } else {
        this.poses.push(new Float32Array(FEATURE_COUNT));
      }
    }
    if (this.poses.length > 75) {
      this.poses.shift();
    }

    // 3. ST-GCN probability evaluation
    let onnxProbability = 0;
    if (this.poses.length >= 6) {
      onnxProbability = await this._evalStgcn();
    }

    // 4. Relational Strike Gate (matches Python inference.py)
    let prob = onnxProbability;
    let gateTriggered = false;
    let minStrikeDist = Infinity;
    let maxIou = 0;

    if (personCount >= 2) {
      for (let i = 0; i < personCount; i++) {
        for (let j = i + 1; j < personCount; j++) {
          const iou = boxIou(detections[i].bbox, detections[j].bbox);
          if (iou > maxIou) maxIou = iou;

          const dist = computeRelationalStrikeDistance(
            detections[i].keypoints,
            detections[j].bbox,
            detections[j].keypoints,
            detections[i].bbox,
            width,
            height,
          );
          if (dist < minStrikeDist) minStrikeDist = dist;
        }
      }

      // Strike threshold: 0.22 * frameWidth (suppresses distant non-contact false alarms)
      const strikeThreshold = 0.22 * width;
      if (prob > 0.65) {
        if (minStrikeDist > strikeThreshold || maxIou === 0) {
          prob = 0.25;
          gateTriggered = true;
        }
      }
    }

    // 5. Alert hold & consecutive triggers (matches Python inference.py)
    let isFighting = false;
    if (gateTriggered) {
      this.consecutiveTriggers = 0;
      isFighting = false;
    } else {
      if (prob >= VIOLENCE_THRESHOLD) {
        this.consecutiveTriggers++;
      } else {
        this.consecutiveTriggers = 0;
      }

      if (this.consecutiveTriggers >= CONSECUTIVE_TRIGGERS_REQUIRED) {
        this.alertHoldFrames = ALERT_HOLD_FRAMES;
        isFighting = true;
      } else if (this.alertHoldFrames > 0) {
        this.alertHoldFrames--;
        isFighting = true;
      } else {
        isFighting = false;
      }
    }

    const inferMs = performance.now() - t0;
    return {
      isFighting,
      probability: prob,
      inferMs,
      personCount,
      secondaryCount: Math.max(0, personCount - 1),
      interactionMode: personCount >= 2 ? "Multi-Person Interaction" : "Single Person",
      bioScore: prob,
      turbulence: 0,
      backend: this._backend,
      actors: detections.map((d, idx) => ({
        bbox: d.bbox,
        keypoints: d.keypoints,
        trackId: idx + 1,
      })),
    };
  }

  // ─── ST-GCN evaluation with linspace sampling ──────────────────────────────
  private async _evalStgcn(): Promise<number> {
    if (!this.stgcnSession || this.poses.length < 6) return 0;

    // Uniform linspace sampling across this.poses matching Python:
    // indices = np.linspace(0, len(self._poses) - 1, SEQUENCE_LENGTH, dtype=int)
    const seqLen = SEQUENCE_LENGTH;
    const sample30: Float32Array[] = [];
    const n = this.poses.length;
    for (let i = 0; i < seqLen; i++) {
      const idx = Math.min(n - 1, Math.round((i / (seqLen - 1)) * (n - 1)));
      sample30.push(this.poses[idx]);
    }

    // Transform into (1, 2, 30, 17) ST-GCN tensor
    const stgcnData = rawToStgcnTensor(sample30);
    const inputTensor = new ort.Tensor("float32", stgcnData, [1, 2, seqLen, KEYPOINT_COUNT]);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.stgcnSession.inputNames[0]] = inputTensor;
    const out = await this.stgcnSession.run(feeds);
    const outData = out[this.stgcnSession.outputNames[0]].data as Float32Array;
    const rawVal = outData[0] ?? 0;

    // Calibrate raw logit / output matching backend/core/inference.py
    let logit = rawVal;
    if (rawVal >= 0 && rawVal <= 1) {
      const eps = 1e-6;
      const valC = Math.max(eps, Math.min(1 - eps, rawVal));
      logit = Math.log(valC / (1 - valC));
    }
    const calLogit = Math.max(-15, Math.min(15, logit + 0.55));
    const prob = 1 / (1 + Math.exp(-calLogit));
    return Math.max(0, Math.min(1, isFinite(prob) ? prob : 0));
  }

  // ─── YOLOv8-pose detection ──────────────────────────────────────────────────
  private async _detectPose(frameData: ImageData): Promise<Detection[]> {
    const { width, height } = frameData;

    // Letterbox to POSE_IMAGE_SIZE
    const scale = Math.min(POSE_IMAGE_SIZE / width, POSE_IMAGE_SIZE / height);
    const rw = Math.round(width * scale), rh = Math.round(height * scale);
    const padX = Math.floor((POSE_IMAGE_SIZE - rw) / 2);
    const padY = Math.floor((POSE_IMAGE_SIZE - rh) / 2);

    // Reuse letterbox canvas
    if (!this.letterboxCanvas) {
      if (typeof OffscreenCanvas !== "undefined") {
        this.letterboxCanvas = new OffscreenCanvas(POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);
      } else {
        const c = document.createElement("canvas");
        c.width = POSE_IMAGE_SIZE;
        c.height = POSE_IMAGE_SIZE;
        this.letterboxCanvas = c;
      }
      this.letterboxCtx = this.letterboxCanvas.getContext("2d") as any;
    }

    const ctx = this.letterboxCtx!;
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);

    // Reuse source canvas
    if (!this.srcCanvas || this.srcCanvas.width !== width || this.srcCanvas.height !== height) {
      if (typeof OffscreenCanvas !== "undefined") {
        this.srcCanvas = new OffscreenCanvas(width, height);
      } else {
        const c = document.createElement("canvas");
        c.width = width;
        c.height = height;
        this.srcCanvas = c;
      }
      this.srcCtx = this.srcCanvas.getContext("2d") as any;
    }
    this.srcCtx!.putImageData(frameData, 0, 0);
    ctx.drawImage(this.srcCanvas, padX, padY, rw, rh);
    const imgData = ctx.getImageData(0, 0, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);

    // Pre-allocated RGB float tensor [1,3,320,320]
    const tensor = this.inputTensorBuffer;
    const planeSize = POSE_IMAGE_SIZE * POSE_IMAGE_SIZE;
    for (let i = 0; i < planeSize; i++) {
      tensor[i]                 = imgData.data[i * 4]     / 255;
      tensor[planeSize + i]     = imgData.data[i * 4 + 1] / 255;
      tensor[2 * planeSize + i] = imgData.data[i * 4 + 2] / 255;
    }

    const inputTensor = new ort.Tensor("float32", tensor, [1, 3, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE]);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.poseSession!.inputNames[0]] = inputTensor;
    const out = await this.poseSession!.run(feeds);
    const raw = out[this.poseSession!.outputNames[0]];
    return this._decodeYolo(raw, width, height, scale, padX, padY);
  }

  private _decodeYolo(
    output: ort.Tensor,
    origW: number, origH: number,
    scale: number, padX: number, padY: number,
  ): Detection[] {
    const data = output.data as Float32Array;
    const dims = output.dims; // [1, 56, N]
    const N = dims[2];
    const ATTRS = dims[1]; // 56 = 4 bbox + 1 conf + 17*3 kps
    if (ATTRS < 56) return [];

    const primary: Detection[] = [];
    for (let b = 0; b < N; b++) {
      const conf = data[4 * N + b];
      // Use SECONDARY_PROPOSAL_CONFIDENCE (0.18) matching Python backend
      if (conf < SECONDARY_PROPOSAL_CONFIDENCE) continue;

      const cx = data[0 * N + b], cy = data[1 * N + b];
      const bw = data[2 * N + b], bh = data[3 * N + b];

      const x1 = Math.max(0, Math.min(origW, (cx - bw / 2 - padX) / scale));
      const y1 = Math.max(0, Math.min(origH, (cy - bh / 2 - padY) / scale));
      const x2 = Math.max(0, Math.min(origW, (cx + bw / 2 - padX) / scale));
      const y2 = Math.max(0, Math.min(origH, (cy + bh / 2 - padY) / scale));
      const bbox: BBox = { x1, y1, x2, y2, conf };

      // Keypoints — normalised to [0,1] in original frame
      const kps = new Float32Array(FEATURE_COUNT);
      for (let k = 0; k < KEYPOINT_COUNT; k++) {
        const base = (5 + k * 3);
        const kpx = (data[base * N + b] - padX) / scale;
        const kpy = (data[(base + 1) * N + b] - padY) / scale;
        kps[k * 2]     = Math.max(0, Math.min(1, kpx / Math.max(origW, 1)));
        kps[k * 2 + 1] = Math.max(0, Math.min(1, kpy / Math.max(origH, 1)));
      }
      primary.push({ keypoints: kps, confidence: conf, bbox });
    }
    return nms(primary, POSE_NMS_THRESHOLD).slice(0, 4);
  }
}

// Singleton for browser use
export const edgeEngine = new EdgeEngine();
