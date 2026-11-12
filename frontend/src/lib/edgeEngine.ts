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
  POSE_CONFIDENCE,
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
      onProgress?.({ stage: "Probing hardware acceleration (WebGPU → WebGL → WASM SIMD)..." });

      // Configure WASM fallback: SIMD + multi-threading
      ort.env.wasm.numThreads = Math.min(
        typeof navigator !== "undefined" && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 4,
        4,
      );
      ort.env.wasm.simd = true;
      ort.env.wasm.wasmPaths = "/ort-wasm/";

      // Execution provider fallback chain: WebGPU (GPU) → WebGL (GPU) → WASM SIMD (CPU)
      const options: ort.InferenceSession.SessionOptions = {
        executionProviders: ["webgpu", "webgl", "wasm"],
        graphOptimizationLevel: "all",
        executionMode: "sequential",
      };
      if (typeof navigator !== "undefined" && navigator.hardwareConcurrency) {
        (options as any).intraOpNumThreads = Math.min(4, navigator.hardwareConcurrency);
      }

      onProgress?.({ stage: "Compiling ONNX graph kernels (GPU-accelerated if available)..." });

      try {
        [this.poseSession, this.stgcnSession] = await Promise.all([
          ort.InferenceSession.create("/models/yolov8n-pose.onnx", options),
          ort.InferenceSession.create("/models/violence_stgcn.onnx", options),
        ]);
      } catch (gpuErr) {
        // Fallback strictly to multi-threaded WASM if WebGPU/WebGL initialization fails on driver
        console.warn("GPU execution provider initialization failed, falling back to WASM:", gpuErr);
        const wasmOpts: ort.InferenceSession.SessionOptions = {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
          executionMode: "sequential",
        };
        if (typeof navigator !== "undefined" && navigator.hardwareConcurrency) {
          (wasmOpts as any).intraOpNumThreads = Math.min(4, navigator.hardwareConcurrency);
        }
        [this.poseSession, this.stgcnSession] = await Promise.all([
          ort.InferenceSession.create("/models/yolov8n-pose.onnx", wasmOpts),
          ort.InferenceSession.create("/models/violence_stgcn.onnx", wasmOpts),
        ]);
      }

      // Detect which execution provider was actually selected
      this._backend = "WASM-SIMD";
      try {
        // WebGPU sessions expose an internal property; probe it
        const poseProvider = (this.poseSession as any)?.handler?.backend;
        if (typeof poseProvider === "string") {
          if (poseProvider.toLowerCase().includes("webgpu")) this._backend = "WebGPU";
          else if (poseProvider.toLowerCase().includes("webgl")) this._backend = "WebGL";
        }
      } catch { /* detection is best-effort */ }

      onProgress?.({ stage: `Edge Models Ready (${this._backend})` });
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
    const outTensor = out[this.stgcnSession.outputNames[0]];
    const outData = (await outTensor.getData()) as Float32Array;
    const rawVal = outData[0] ?? 0;
    try { inputTensor.dispose?.(); outTensor.dispose?.(); } catch { /* noop */ }

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

  /**
   * Letterbox RGBA ImageData into [1, 3, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE] Float32 NCHW tensor.
   * Strips the Alpha channel, scales pixels to [0.0, 1.0], and uses (114, 114, 114) padding
   * matching Python's OpenCV _letterbox + cv2.dnn.blobFromImage preprocessing.
   */
  private _preprocessFrameToTensor(frameData: ImageData): {
    tensor: Float32Array;
    scale: number;
    padX: number;
    padY: number;
  } {
    const { width, height } = frameData;

    const scale = Math.min(POSE_IMAGE_SIZE / width, POSE_IMAGE_SIZE / height);
    const rw = Math.round(width * scale);
    const rh = Math.round(height * scale);
    const padX = Math.floor((POSE_IMAGE_SIZE - rw) / 2);
    const padY = Math.floor((POSE_IMAGE_SIZE - rh) / 2);

    if (!this.letterboxCanvas) {
      if (typeof OffscreenCanvas !== "undefined") {
        this.letterboxCanvas = new OffscreenCanvas(POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);
      } else {
        const c = document.createElement("canvas");
        c.width = POSE_IMAGE_SIZE;
        c.height = POSE_IMAGE_SIZE;
        this.letterboxCanvas = c;
      }
      this.letterboxCtx = this.letterboxCanvas.getContext("2d", { willReadFrequently: true }) as any;
    }

    const ctx = this.letterboxCtx!;
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);

    if (!this.srcCanvas || this.srcCanvas.width !== width || this.srcCanvas.height !== height) {
      if (typeof OffscreenCanvas !== "undefined") {
        this.srcCanvas = new OffscreenCanvas(width, height);
      } else {
        const c = document.createElement("canvas");
        c.width = width;
        c.height = height;
        this.srcCanvas = c;
      }
      this.srcCtx = this.srcCanvas.getContext("2d", { willReadFrequently: true }) as any;
    }
    this.srcCtx!.putImageData(frameData, 0, 0);
    ctx.drawImage(this.srcCanvas, padX, padY, rw, rh);
    const imgData = ctx.getImageData(0, 0, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);

    // Transpose HWC (RGBA) -> NCHW (RGB), strip Alpha channel, divide by 255.0
    const tensor = this.inputTensorBuffer;
    const rgba = imgData.data;
    const planeSize = POSE_IMAGE_SIZE * POSE_IMAGE_SIZE;
    const inv255 = 1.0 / 255.0;
    for (let i = 0; i < planeSize; i++) {
      const idx4 = i << 2;
      tensor[i]                 = rgba[idx4]     * inv255; // R plane
      tensor[planeSize + i]     = rgba[idx4 + 1] * inv255; // G plane
      tensor[2 * planeSize + i] = rgba[idx4 + 2] * inv255; // B plane
    }

    return { tensor, scale, padX, padY };
  }

  // ─── YOLOv8-pose detection ──────────────────────────────────────────────────
  private async _detectPose(frameData: ImageData): Promise<Detection[]> {
    const { width, height } = frameData;
    const { tensor, scale, padX, padY } = this._preprocessFrameToTensor(frameData);

    const inputTensor = new ort.Tensor("float32", tensor, [1, 3, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE]);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.poseSession!.inputNames[0]] = inputTensor;
    const out = await this.poseSession!.run(feeds);
    const raw = out[this.poseSession!.outputNames[0]];
    const data = (await raw.getData()) as Float32Array;
    const dims = raw.dims;
    try { inputTensor.dispose?.(); raw.dispose?.(); } catch { /* noop */ }
    return this._decodeYolo(data, dims, width, height, scale, padX, padY);
  }

  private _decodeYolo(
    data: Float32Array,
    dims: readonly number[],
    origW: number, origH: number,
    scale: number, padX: number, padY: number,
  ): Detection[] {
    if (dims.length < 3) return [];

    // Standard YOLOv8-Pose layout is [1, 56, 2100] (channels-first); handle transposed [1, 2100, 56] too
    const channelsFirst = dims[1] <= dims[2];
    const ATTRS = channelsFirst ? dims[1] : dims[2];
    const N = channelsFirst ? dims[2] : dims[1];
    if (ATTRS < 55) return [];

    const hasBoxConf = ATTRS >= 56;
    const kpStart = hasBoxConf ? 5 : 4;
    const readAttr = (attrIdx: number, anchorIdx: number): number =>
      channelsFirst ? data[attrIdx * N + anchorIdx] : data[anchorIdx * ATTRS + attrIdx];

    const primary: Detection[] = [];
    for (let b = 0; b < N; b++) {
      let conf: number;
      if (hasBoxConf) {
        conf = readAttr(4, b);
      } else {
        let maxKpConf = 0;
        for (let k = 0; k < KEYPOINT_COUNT; k++) {
          const kc = readAttr(kpStart + k * 3 + 2, b);
          if (kc > maxKpConf) maxKpConf = kc;
        }
        conf = maxKpConf;
      }

      if (conf < POSE_CONFIDENCE) continue;

      const cx = readAttr(0, b);
      const cy = readAttr(1, b);
      const bw = readAttr(2, b);
      const bh = readAttr(3, b);

      const x1 = Math.max(0, Math.min(origW, (cx - bw / 2 - padX) / scale));
      const y1 = Math.max(0, Math.min(origH, (cy - bh / 2 - padY) / scale));
      const x2 = Math.max(0, Math.min(origW, (cx + bw / 2 - padX) / scale));
      const y2 = Math.max(0, Math.min(origH, (cy + bh / 2 - padY) / scale));
      const bbox: BBox = { x1, y1, x2, y2, conf };

      // Keypoints — normalised to [0,1] in original frame, masking low-confidence (< 0.5) as [0, 0]
      const kps = new Float32Array(FEATURE_COUNT);
      for (let k = 0; k < KEYPOINT_COUNT; k++) {
        const base = kpStart + k * 3;
        const rawKx = readAttr(base, b);
        const rawKy = readAttr(base + 1, b);
        const kpConf = readAttr(base + 2, b);
        if (kpConf < 0.5 || (rawKx === 0 && rawKy === 0)) {
          kps[k * 2]     = 0.0;
          kps[k * 2 + 1] = 0.0;
          continue;
        }
        const kpx = (rawKx - padX) / scale;
        const kpy = (rawKy - padY) / scale;
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
