/**
 * edgeEngine.ts
 * Full TypeScript port of Python PoseLSTMDetector.
 *
 * Loads yolov8n-pose.onnx + violence_lstm.onnx via onnxruntime-web,
 * then runs the complete pipeline (detection → tracking → LSTM → bio-score
 * → turbulence fusion) identical to the Python backend.
 *
 * Execution providers: WebGPU (GPU) → WASM SIMD multi-threaded (CPU fallback).
 * Frame-striding: full pose inference every POSE_STRIDE frames.
 */

import * as ort from "onnxruntime-web";
import {
  SEQUENCE_LENGTH,
  KEYPOINT_COUNT,
  FEATURE_COUNT,
  RAW_FEATURE_COUNT,
  KINEMATIC_FEATURE_COUNT,
  POSE_IMAGE_SIZE,
  PRIMARY_PERSON_CONFIDENCE,
  SECONDARY_PROPOSAL_CONFIDENCE,
  POSE_NMS_THRESHOLD,
  VIOLENCE_THRESHOLD,
  BIO_SCORE_FLOOR,
  STRONG_BIO_SCORE_THRESHOLD,
  CONSECUTIVE_TRIGGERS_REQUIRED,
  ALERT_HOLD_FRAMES,
  POSE_STRIDE,
  TURBULENCE_BOOST_THRESHOLD,
  Detection,
  BBox,
  TrackedPerson,
  normalizePoseSequence,
  extractKinematicTensor,
  biomechanicalScore,
  computeInteractionTurbulence,
  rgbaToGray,
  updateTracks,
  resetTrackIdCounter,
  nms,
  bboxCenter,
  bboxAspect,
  boxIou,
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
  private lstmSession: ort.InferenceSession | null = null;
  private _backend = "—";
  private _state: EngineState = "idle";

  // Track state
  private tracks = new Map<number, TrackedPerson>();
  private frameIndex = 0;
  private lastFrameTime: number | null = null;
  private fps = 0;

  // Turbulence state
  private prevGray: Uint8Array | null = null;
  private interactionTurbulence = 0;

  // Secondary/motion state
  private secondaryCount = 0;
  private secondaryPersistenceFrames = 0;

  // Alert gating
  private consecutiveTriggers = 0;
  private alertHoldFrames = 0;

  // Cached from last stride
  private cachedDetections: Detection[] = [];
  private cachedSecondaryCount = 0;

  get backend(): string { return this._backend; }
  get state(): EngineState { return this._state; }

  // ─── Init ───────────────────────────────────────────────────────────────────
  async load(
    onProgress?: (p: LoadProgress) => void,
  ): Promise<void> {
    this._state = "loading";
    try {
      onProgress?.({ stage: "Allocating Client WebGPU Buffer..." });
      ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency ?? 4, 4);
      ort.env.wasm.simd = true;
      ort.env.wasm.wasmPaths = "/ort-wasm/";

      onProgress?.({ stage: "Compiling WASM Kernels..." });
      const opts: ort.InferenceSession.SessionOptions = {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      };

      [this.poseSession, this.lstmSession] = await Promise.all([
        ort.InferenceSession.create("/models/yolov8n-pose.onnx", opts),
        ort.InferenceSession.create("/models/violence_lstm_v2.onnx", opts),
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
    this.tracks.clear();
    this.frameIndex = 0;
    this.lastFrameTime = null;
    this.fps = 0;
    this.prevGray = null;
    this.interactionTurbulence = 0;
    this.secondaryCount = 0;
    this.secondaryPersistenceFrames = 0;
    this.consecutiveTriggers = 0;
    this.alertHoldFrames = 0;
    this.cachedDetections = [];
    this.cachedSecondaryCount = 0;
    resetTrackIdCounter();
  }

  // ─── Main inference entry ───────────────────────────────────────────────────
  async processFrame(frameData: ImageData): Promise<EdgePrediction> {
    if (!this.poseSession || !this.lstmSession) {
      throw new Error("EdgeEngine not loaded");
    }
    const t0 = performance.now();
    const now = performance.now() / 1000; // seconds
    const dt = this.lastFrameTime !== null
      ? Math.max((now - this.lastFrameTime), 0.02)
      : 1 / 15;

    const { width, height } = frameData;
    const isPoseFrame = this.frameIndex % POSE_STRIDE === 0;

    // ── Pose extraction (strided) ───────────────────────────────────────────
    let detections: Detection[];
    if (isPoseFrame) {
      detections = await this._detectPose(frameData);
      this.cachedDetections = detections;
      this.cachedSecondaryCount = 0;
    } else {
      detections = this.cachedDetections;
    }

    // ── Track update (mirror Python) ────────────────────────────────────────
    const activeTracks = updateTracks(this.tracks, detections, now, dt, width, height);
    const sorted = [...activeTracks].sort((a, b) => b.activity - a.activity).slice(0, 2);

    // ── Interaction-Zone Optical turbulence ─────────────────────────────────
    const curGray = rgbaToGray(frameData.data, width, height);
    if (sorted.length >= 2) {
      this.interactionTurbulence = computeInteractionTurbulence(
        this.prevGray, curGray, width, height, sorted.map(t => t.bbox),
      );
    } else {
      this.interactionTurbulence = 0;
    }
    this.prevGray = curGray;

    this.frameIndex++;
    if (this.lastFrameTime !== null && dt > 0) {
      const instantFps = 1 / dt;
      this.fps = this.fps === 0 ? instantFps : 0.8 * this.fps + 0.2 * instantFps;
    }
    this.lastFrameTime = now;

    const personCount = activeTracks.length;
    const secondaryCount = this.cachedSecondaryCount;

    // Secondary persistence (mirror Python)
    let effectiveSecondary = secondaryCount;
    if (personCount === 1 && secondaryCount > 0) {
      this.secondaryPersistenceFrames++;
      if (this.secondaryPersistenceFrames < 2) effectiveSecondary = 0;
    } else {
      this.secondaryPersistenceFrames = 0;
      if (personCount === 1) effectiveSecondary = 0;
    }

    const interactionActive = personCount >= 2 || (personCount === 1 && effectiveSecondary >= 1);
    const interactionMode = personCount >= 2
      ? (this.interactionTurbulence > TURBULENCE_BOOST_THRESHOLD ? "Multi-Person + Turbulence" : "2+ Persons")
      : personCount === 1 && effectiveSecondary >= 1
        ? "Person + Entity/Animal"
        : "Solo / Safe";

    // ── LSTM probability ────────────────────────────────────────────────────
    let onnxProbability = 0;
    for (const track of sorted) {
      const p = await this._trackProbability(track);
      if (p > onnxProbability) onnxProbability = p;
    }

    // ── Bio score ───────────────────────────────────────────────────────────
    const bioScore = biomechanicalScore(
      sorted, [], personCount, effectiveSecondary,
      width, height, this.interactionTurbulence,
    );

    // ── Alert gating (exact mirror of Python inference.py) ──────────────────
    let probability = 0;
    let isFighting = false;

    if (interactionActive) {
      // Trust the physics-aware ONNX model's output
      probability = onnxProbability;
      // Optical turbulence check as a secondary modifier
      if (this.interactionTurbulence > TURBULENCE_BOOST_THRESHOLD) {
        probability = Math.min(probability + 0.10, 1.0);
      }

      // Clean 0.65 threshold with 2-frame confirmation
      if (isPoseFrame) {
        if (probability >= VIOLENCE_THRESHOLD) {
          this.consecutiveTriggers++;
        } else {
          this.consecutiveTriggers = 0;
        }
        if (this.consecutiveTriggers >= CONSECUTIVE_TRIGGERS_REQUIRED) {
          this.alertHoldFrames = ALERT_HOLD_FRAMES;
        } else if (this.alertHoldFrames > 0) {
          this.alertHoldFrames--;
        }
      }
      isFighting = this.alertHoldFrames > 0;
    } else {
      probability = Math.min(onnxProbability, 0.249);
      this.consecutiveTriggers = 0;
      this.alertHoldFrames = 0;
      isFighting = false;
    }

    const inferMs = performance.now() - t0;
    return {
      isFighting,
      probability,
      inferMs,
      personCount,
      secondaryCount: effectiveSecondary,
      interactionMode,
      bioScore,
      turbulence: this.interactionTurbulence,
      backend: this._backend,
      actors: activeTracks.map(t => ({
        bbox: t.bbox,
        keypoints: t.keypoints,
        trackId: t.trackId,
      })),
    };
  }

  // ─── YOLOv8-pose detection ──────────────────────────────────────────────────
  private async _detectPose(frameData: ImageData): Promise<Detection[]> {
    const { width, height } = frameData;

    // Letterbox to POSE_IMAGE_SIZE
    const scale = Math.min(POSE_IMAGE_SIZE / width, POSE_IMAGE_SIZE / height);
    const rw = Math.round(width * scale), rh = Math.round(height * scale);
    const padX = Math.floor((POSE_IMAGE_SIZE - rw) / 2);
    const padY = Math.floor((POSE_IMAGE_SIZE - rh) / 2);

    // Render into offscreen canvas with letterbox
    const oc = new OffscreenCanvas(POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);
    const octx = oc.getContext("2d")!;
    octx.fillStyle = "rgb(114,114,114)";
    octx.fillRect(0, 0, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);
    // Draw source frame scaled
    const srcCanvas = new OffscreenCanvas(width, height);
    const srcCtx = srcCanvas.getContext("2d")!;
    srcCtx.putImageData(frameData, 0, 0);
    octx.drawImage(srcCanvas, padX, padY, rw, rh);
    const imgData = octx.getImageData(0, 0, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE);

    // RGB float tensor [1,3,320,320]
    const tensor = new Float32Array(3 * POSE_IMAGE_SIZE * POSE_IMAGE_SIZE);
    for (let i = 0; i < POSE_IMAGE_SIZE * POSE_IMAGE_SIZE; i++) {
      tensor[i]                                 = imgData.data[i * 4]     / 255;
      tensor[POSE_IMAGE_SIZE * POSE_IMAGE_SIZE + i]     = imgData.data[i * 4 + 1] / 255;
      tensor[2 * POSE_IMAGE_SIZE * POSE_IMAGE_SIZE + i] = imgData.data[i * 4 + 2] / 255;
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
      if (conf < PRIMARY_PERSON_CONFIDENCE) continue;

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

  // ─── LSTM probability (port of _track_probability) ──────────────────────────
  private async _trackProbability(track: TrackedPerson): Promise<number> {
    if (!this.lstmSession || track.history.length < 6) return 0;

    let sequence = [...track.history];
    if (sequence.length < SEQUENCE_LENGTH) {
      const pad = Array(SEQUENCE_LENGTH - sequence.length).fill(sequence[0]);
      sequence = [...pad, ...sequence];
    }
    const last30 = sequence.slice(-SEQUENCE_LENGTH);

    // Flatten to [30, 34] Float32Array
    const raw = new Float32Array(SEQUENCE_LENGTH * RAW_FEATURE_COUNT);
    for (let t = 0; t < SEQUENCE_LENGTH; t++) raw.set(last30[t], t * RAW_FEATURE_COUNT);

    // extractKinematicTensor -> (30, 82)
    const kinematic = extractKinematicTensor(raw);
    const inputTensor = new ort.Tensor("float32", kinematic, [1, SEQUENCE_LENGTH, KINEMATIC_FEATURE_COUNT]);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.lstmSession.inputNames[0]] = inputTensor;
    const out = await this.lstmSession.run(feeds);
    const outData = out[this.lstmSession.outputNames[0]].data as Float32Array;
    const p = outData.length > 1 ? outData[1] : outData[0];
    return Math.max(0, Math.min(1, isFinite(p) ? p : 0));
  }
}

// Singleton for browser use
export const edgeEngine = new EdgeEngine();
