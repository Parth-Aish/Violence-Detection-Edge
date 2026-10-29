/**
 * kinematics.ts
 * Exact TypeScript port of backend/pose_features.py + biomechanical scoring
 * from backend/inference.py (_biomechanical_score, _update_tracks, etc.)
 *
 * All maths mirrors the Python implementation so that the Edge engine
 * produces mathematically identical output to the Python FastAPI engine.
 */

// ─── Constants (mirror Python) ────────────────────────────────────────────────
export const SEQUENCE_LENGTH = 30;
export const KEYPOINT_COUNT = 17;
export const RAW_FEATURE_COUNT = KEYPOINT_COUNT * 2; // 34
export const FEATURE_COUNT = RAW_FEATURE_COUNT; // 34 (backward compatibility)
export const KINEMATIC_FEATURE_COUNT = 82;
export const POSE_IMAGE_SIZE = 320;
export const POSE_CONFIDENCE = 0.25;
export const PRIMARY_PERSON_CONFIDENCE = 0.30;
export const SECONDARY_PROPOSAL_CONFIDENCE = 0.18;
export const POSE_NMS_THRESHOLD = 0.45;
export const VIOLENCE_THRESHOLD = 0.65;
export const BIO_SCORE_FLOOR = 0.56;
export const STRONG_BIO_SCORE_THRESHOLD = 0.60;
export const CONSECUTIVE_TRIGGERS_REQUIRED = 2;
export const ALERT_HOLD_FRAMES = 3;
export const POSE_STRIDE = 2;
export const TURBULENCE_BOOST_THRESHOLD = 0.08;
export const TURBULENCE_BOOST_AMOUNT = 0.18;

// COCO keypoint indices
export const KP = {
  NOSE: 0, L_EYE: 1, R_EYE: 2, L_EAR: 3, R_EAR: 4,
  L_SHOULDER: 5, R_SHOULDER: 6,
  L_ELBOW: 7, R_ELBOW: 8,
  L_WRIST: 9, R_WRIST: 10,
  L_HIP: 11, R_HIP: 12,
  L_KNEE: 13, R_KNEE: 14,
  L_ANKLE: 15, R_ANKLE: 16,
} as const;

// COCO skeleton pairs for rendering
export const SKELETON_PAIRS: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

// ─── Types ────────────────────────────────────────────────────────────────────
export interface RawKeypoint {
  /** normalised 0-1 in original frame coords */
  x: number;
  y: number;
  conf: number;
}

export interface BBox {
  /** pixel coords in original frame */
  x1: number; y1: number; x2: number; y2: number;
  conf: number;
}

export interface TrackedPerson {
  trackId: number;
  bbox: BBox;
  /** normalised [0,1] keypoints, shape [17, 2] flattened row-major */
  keypoints: Float32Array;      // length 34
  /** smoothed model_pose used for LSTM history, shape [17,2] */
  modelPose: Float32Array;      // length 34
  velocity: Float32Array;       // length 34
  acceleration: Float32Array;   // length 34
  history: Float32Array[];      // rolling SEQUENCE_LENGTH buffer of 34-dim features
  lastDetectionKps: Float32Array;
  lastDetectionTime: number;
  lastSeen: number;
  confidence: number;
  centerY: number;
  aspectRatio: number;
  centerDrop: number;
  aspectCollapse: number;
  activity: number;
  prevBbox: BBox | null;
}

export interface Detection {
  /** normalised [0,1] keypoints flat array, length 34 */
  keypoints: Float32Array;
  confidence: number;
  bbox: BBox;
}

// ─── Utility maths ────────────────────────────────────────────────────────────

export function boxIou(a: BBox, b: BBox): number {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const aA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const aB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = aA + aB - inter;
  return union > 0 ? inter / union : 0;
}

export function bboxCenter(b: BBox): [number, number] {
  return [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2];
}

export function bboxAspect(b: BBox): number {
  return (b.y2 - b.y1) / Math.max(b.x2 - b.x1, 1);
}

export function pointToBoxDist(px: number, py: number, b: BBox): number {
  const dx = Math.max(b.x1 - px, 0, px - b.x2);
  const dy = Math.max(b.y1 - py, 0, py - b.y2);
  return Math.hypot(dx, dy);
}

/** NMS on BBox array, returns kept indices */
export function nms(dets: Detection[], iouThresh: number): Detection[] {
  const sorted = [...dets].sort((a, b) => b.confidence - a.confidence);
  const keep: Detection[] = [];
  const suppressed = new Set<number>();
  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    keep.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (boxIou(sorted[i].bbox, sorted[j].bbox) > iouThresh) suppressed.add(j);
    }
  }
  return keep;
}

// ─── extractKinematicTensor (exact port of kinematics_extractor.py) ───────────
/**
 * Transform a (30, 34) raw pose sequence into a physics-aware (30, 82) kinematic tensor.
 *
 * Breakdown of 82 features:
 *   0..33 (34): Root-centered, torso-scaled normalized joint coordinates (17 * 2)
 *   34..67 (34): 1st-order frame-to-frame normalized velocities (17 * 2)
 *   68..75  (8): 2nd-order acceleration magnitudes (4) + velocity magnitudes (4) for wrists & ankles
 *   76..81  (6): Cosine angles of elbows (2) & knees (2), torso aspect ratio (1), torso vertical tilt (1)
 *
 * Output: Float32Array of length 30 * 82 = 2460 floats.
 */
export function extractKinematicTensor(
  sequence: Float32Array | number[] | Float32Array[] | number[][],
): Float32Array {
  const SL = SEQUENCE_LENGTH; // 30
  const KN = KEYPOINT_COUNT;  // 17
  const RF = RAW_FEATURE_COUNT; // 34

  // Parse input into [30][17][2]
  const pts: number[][][] = [];
  if (
    Array.isArray(sequence) &&
    sequence.length === SL &&
    (Array.isArray(sequence[0]) || sequence[0] instanceof Float32Array)
  ) {
    for (let t = 0; t < SL; t++) {
      const frame: number[][] = [];
      const row = sequence[t] as ArrayLike<number>;
      for (let k = 0; k < KN; k++) {
        frame.push([row[k * 2] ?? 0, row[k * 2 + 1] ?? 0]);
      }
      pts.push(frame);
    }
  } else {
    const flat = sequence as ArrayLike<number>;
    for (let t = 0; t < SL; t++) {
      const frame: number[][] = [];
      for (let k = 0; k < KN; k++) {
        frame.push([flat[t * RF + k * 2] ?? 0, flat[t * RF + k * 2 + 1] ?? 0]);
      }
      pts.push(frame);
    }
  }

  // Handle zero/missing frames via linear interpolation
  const frameValid = pts.map(f => f.some(kp => kp[0] !== 0 || kp[1] !== 0));
  const validIdx = frameValid.map((v, i) => v ? i : -1).filter(i => i >= 0);
  if (validIdx.length === 0) {
    return new Float32Array(SL * KINEMATIC_FEATURE_COUNT);
  }

  for (let k = 0; k < KN; k++) {
    for (let c = 0; c < 2; c++) {
      const v = pts.map(f => f[k][c]);
      const jv = frameValid.map((valid, t) => valid && isFinite(v[t]));
      const kn = jv.map((val, i) => val ? i : -1).filter(i => i >= 0);
      if (kn.length > 0) {
        const knownVals = kn.map(i => v[i]);
        for (let t = 0; t < SL; t++) {
          pts[t][k][c] = linearInterp(t, kn, knownVals);
        }
      }
    }
  }

  for (let fi = 0; fi < SL; fi++) {
    if (!frameValid[fi]) {
      const nearest = validIdx.reduce((prev, cur) =>
        Math.abs(cur - fi) < Math.abs(prev - fi) ? cur : prev
      );
      pts[fi] = pts[nearest].map(kp => [...kp]);
    }
  }

  // 1. Mid-hip root-centering & Torso scale normalization (34 features)
  const roots: [number, number][] = [];
  for (let t = 0; t < SL; t++) {
    const lhip = pts[t][KP.L_HIP], rhip = pts[t][KP.R_HIP];
    const lsh  = pts[t][KP.L_SHOULDER], rsh = pts[t][KP.R_SHOULDER];
    const hipsValid = (lhip[0] !== 0 || lhip[1] !== 0) && (rhip[0] !== 0 || rhip[1] !== 0);
    const shouldersValid = (lsh[0] !== 0 || lsh[1] !== 0) && (rsh[0] !== 0 || rsh[1] !== 0);
    if (hipsValid) {
      roots.push([(lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2]);
    } else if (shouldersValid) {
      roots.push([(lsh[0] + rsh[0]) / 2, (lsh[1] + rsh[1]) / 2]);
    } else {
      const xs = pts[t].map(kp => kp[0]).sort((a, b) => a - b);
      const ys = pts[t].map(kp => kp[1]).sort((a, b) => a - b);
      roots.push([xs[Math.floor(KN / 2)], ys[Math.floor(KN / 2)]]);
    }
  }

  const centered: number[][][] = [];
  for (let t = 0; t < SL; t++) {
    const frame: number[][] = [];
    for (let k = 0; k < KN; k++) {
      frame.push([pts[t][k][0] - roots[t][0], pts[t][k][1] - roots[t][1]]);
    }
    centered.push(frame);
  }

  const midShoulders: [number, number][] = [];
  const torsoLens: number[] = [];
  const fallbacks: number[] = [];
  for (let t = 0; t < SL; t++) {
    const msX = (pts[t][KP.L_SHOULDER][0] + pts[t][KP.R_SHOULDER][0]) / 2;
    const msY = (pts[t][KP.L_SHOULDER][1] + pts[t][KP.R_SHOULDER][1]) / 2;
    midShoulders.push([msX, msY]);
    torsoLens.push(Math.hypot(msX - roots[t][0], msY - roots[t][1]));

    const norms = centered[t].map(kp => Math.hypot(kp[0], kp[1])).sort((a, b) => a - b);
    // np.percentile(..., 90) on length 17: (17 - 1) * 0.9 = 14.4 -> norms[14] + 0.4 * (norms[15] - norms[14])
    const idx = 14.4;
    const iLow = Math.floor(idx);
    const iHigh = Math.ceil(idx);
    const fb = norms[iLow] + (idx - iLow) * (norms[iHigh] - norms[iLow]);
    fallbacks.push(fb);
  }

  const scales: number[] = [];
  for (let t = 0; t < SL; t++) {
    const s = torsoLens[t] > 1e-4 ? torsoLens[t] : fallbacks[t];
    scales.push(Math.max(s, 1e-3));
  }

  const normJoints: number[][][] = [];
  for (let t = 0; t < SL; t++) {
    const frame: number[][] = [];
    const sc = scales[t];
    for (let k = 0; k < KN; k++) {
      const nx = Math.max(-2.5, Math.min(2.5, centered[t][k][0] / sc));
      const ny = Math.max(-2.5, Math.min(2.5, centered[t][k][1] / sc));
      frame.push([nx, ny]);
    }
    normJoints.push(frame);
  }

  // 2. 1st-Order Velocity (34 features)
  const velocity: number[][][] = [];
  for (let t = 0; t < SL; t++) {
    const frame: number[][] = [];
    for (let k = 0; k < KN; k++) {
      if (t === 0) {
        frame.push([0, 0]);
      } else {
        const vx = Math.max(-2.5, Math.min(2.5, normJoints[t][k][0] - normJoints[t - 1][k][0]));
        const vy = Math.max(-2.5, Math.min(2.5, normJoints[t][k][1] - normJoints[t - 1][k][1]));
        frame.push([vx, vy]);
      }
    }
    velocity.push(frame);
  }

  // 3. 2nd-Order Acceleration (8 features: 4 acc mags + 4 vel mags for wrists & ankles)
  const accel: number[][][] = [];
  for (let t = 0; t < SL; t++) {
    const frame: number[][] = [];
    for (let k = 0; k < KN; k++) {
      if (t === 0) {
        frame.push([0, 0]);
      } else {
        const ax = Math.max(-5.0, Math.min(5.0, velocity[t][k][0] - velocity[t - 1][k][0]));
        const ay = Math.max(-5.0, Math.min(5.0, velocity[t][k][1] - velocity[t - 1][k][1]));
        frame.push([ax, ay]);
      }
    }
    accel.push(frame);
  }

  const STRIKE_JOINTS = [KP.L_WRIST, KP.R_WRIST, KP.L_ANKLE, KP.R_ANKLE]; // [9, 10, 15, 16]
  const dyn8: number[][] = [];
  for (let t = 0; t < SL; t++) {
    const accMags = STRIKE_JOINTS.map(j => Math.hypot(accel[t][j][0], accel[t][j][1]));
    const velMags = STRIKE_JOINTS.map(j => Math.hypot(velocity[t][j][0], velocity[t][j][1]));
    const row = [...accMags, ...velMags].map(v => Math.max(0.0, Math.min(5.0, v)));
    dyn8.push(row);
  }

  // 4. Angles & Posture (6 features)
  function cosAngle(p1: [number, number], p2: [number, number], p3: [number, number]): number {
    const v1x = p1[0] - p2[0], v1y = p1[1] - p2[1];
    const v2x = p3[0] - p2[0], v2y = p3[1] - p2[1];
    const dot = v1x * v2x + v1y * v2y;
    const norm1 = Math.hypot(v1x, v1y);
    const norm2 = Math.hypot(v2x, v2y);
    return Math.max(-1.0, Math.min(1.0, dot / (norm1 * norm2 + 1e-6)));
  }

  const angles6: number[][] = [];
  for (let t = 0; t < SL; t++) {
    const cosLE = cosAngle(
      pts[t][KP.L_SHOULDER] as [number, number],
      pts[t][KP.L_ELBOW] as [number, number],
      pts[t][KP.L_WRIST] as [number, number],
    );
    const cosRE = cosAngle(
      pts[t][KP.R_SHOULDER] as [number, number],
      pts[t][KP.R_ELBOW] as [number, number],
      pts[t][KP.R_WRIST] as [number, number],
    );
    const cosLK = cosAngle(
      pts[t][KP.L_HIP] as [number, number],
      pts[t][KP.L_KNEE] as [number, number],
      pts[t][KP.L_ANKLE] as [number, number],
    );
    const cosRK = cosAngle(
      pts[t][KP.R_HIP] as [number, number],
      pts[t][KP.R_KNEE] as [number, number],
      pts[t][KP.R_ANKLE] as [number, number],
    );

    const shW = Math.hypot(
      pts[t][KP.L_SHOULDER][0] - pts[t][KP.R_SHOULDER][0],
      pts[t][KP.L_SHOULDER][1] - pts[t][KP.R_SHOULDER][1],
    );
    const torsoH = Math.abs(midShoulders[t][1] - roots[t][1]);
    const aspect = Math.max(0.0, Math.min(5.0, torsoH / (shW + 1e-4)));

    const torsoVecX = midShoulders[t][0] - roots[t][0];
    const torsoVecY = midShoulders[t][1] - roots[t][1];
    const cosTilt = Math.max(
      -1.0,
      Math.min(1.0, -torsoVecY / (Math.hypot(torsoVecX, torsoVecY) + 1e-6)),
    );

    angles6.push([cosLE, cosRE, cosLK, cosRK, aspect, cosTilt]);
  }

  // Assemble complete (30, 82) feature matrix
  const out = new Float32Array(SL * KINEMATIC_FEATURE_COUNT);
  for (let t = 0; t < SL; t++) {
    const base = t * KINEMATIC_FEATURE_COUNT;
    // 0..33 (34): normJoints
    for (let k = 0; k < KN; k++) {
      out[base + k * 2]     = normJoints[t][k][0];
      out[base + k * 2 + 1] = normJoints[t][k][1];
    }
    // 34..67 (34): velocity
    for (let k = 0; k < KN; k++) {
      out[base + 34 + k * 2]     = velocity[t][k][0];
      out[base + 34 + k * 2 + 1] = velocity[t][k][1];
    }
    // 68..75 (8): dyn8 (4 acc_mags + 4 vel_mags)
    for (let i = 0; i < 8; i++) {
      out[base + 68 + i] = dyn8[t][i];
    }
    // 76..81 (6): angles6
    for (let i = 0; i < 6; i++) {
      out[base + 76 + i] = angles6[t][i];
    }
  }
  return out;
}

/**
 * Backward compatibility aliases.
 */
export const normalizePoseSequence = extractKinematicTensor;
export const extract_kinematic_features = extractKinematicTensor;
export const extractKinematicFeatures = extractKinematicTensor;

/**
 * Calculate the interior angle at joint B between vectors BA and BC in degrees.
 */
export function calculateJointAngle(
  a: [number, number],
  b: [number, number],
  c: [number, number],
): number {
  const v1x = a[0] - b[0];
  const v1y = a[1] - b[1];
  const v2x = c[0] - b[0];
  const v2y = c[1] - b[1];
  const dot = v1x * v2x + v1y * v2y;
  const mag1 = Math.hypot(v1x, v1y);
  const mag2 = Math.hypot(v2x, v2y);
  if (mag1 < 1e-6 || mag2 < 1e-6) return 0;
  const cos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return Math.acos(cos) * (180 / Math.PI);
}

/**
 * Extract key kinematic angles (elbows, knees, shoulders) from normalized keypoints [17, 2].
 */
export function extractPoseAngles(kps: Float32Array): {
  leftElbow: number;
  rightElbow: number;
  leftKnee: number;
  rightKnee: number;
  leftShoulder: number;
  rightShoulder: number;
} {
  const getPt = (idx: number): [number, number] => [kps[idx * 2], kps[idx * 2 + 1]];
  return {
    leftElbow: calculateJointAngle(getPt(KP.L_SHOULDER), getPt(KP.L_ELBOW), getPt(KP.L_WRIST)),
    rightElbow: calculateJointAngle(getPt(KP.R_SHOULDER), getPt(KP.R_ELBOW), getPt(KP.R_WRIST)),
    leftKnee: calculateJointAngle(getPt(KP.L_HIP), getPt(KP.L_KNEE), getPt(KP.L_ANKLE)),
    rightKnee: calculateJointAngle(getPt(KP.R_HIP), getPt(KP.R_KNEE), getPt(KP.R_ANKLE)),
    leftShoulder: calculateJointAngle(getPt(KP.L_HIP), getPt(KP.L_SHOULDER), getPt(KP.L_ELBOW)),
    rightShoulder: calculateJointAngle(getPt(KP.R_HIP), getPt(KP.R_SHOULDER), getPt(KP.R_ELBOW)),
  };
}

function linearInterp(x: number, xs: number[], ys: number[]): number {
  if (xs.length === 0) return 0;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 0; i < xs.length - 1; i++) {
    if (x >= xs[i] && x <= xs[i + 1]) {
      const t = (x - xs[i]) / (xs[i + 1] - xs[i]);
      return ys[i] + t * (ys[i + 1] - ys[i]);
    }
  }
  return ys[ys.length - 1];
}

// ─── Biomechanical score (exact port of _biomechanical_score) ──────────────────
export function biomechanicalScore(
  activeTracks: TrackedPerson[],
  secondaryBoxes: BBox[],
  personCount: number,
  secondaryCount: number,
  frameWidth: number,
  frameHeight: number,
  interactionTurbulence: number,
): number {
  if (activeTracks.length === 0) return 0;

  const pairScores: [number, number, number][] = [];
  for (const track of activeTracks.slice(0, 2)) {
    // velocity/acceleration of wrists + ankles (kps 9,10,15,16)
    const strikeIndices = [9, 10, 15, 16];
    const strikeVels = strikeIndices.map(k => Math.hypot(track.velocity[k * 2], track.velocity[k * 2 + 1]));
    const strikeAccs = strikeIndices.map(k => Math.hypot(track.acceleration[k * 2], track.acceleration[k * 2 + 1]));
    const strikeSpeed = Math.max(...strikeVels);
    const strikeAcc   = Math.max(...strikeAccs);
    const strike   = Math.min(strikeSpeed / 2.0, 1.0);
    const burst    = Math.min(strikeAcc / 12.0, 1.0);
    const knockdown = Math.max(
      Math.min(track.centerDrop / 0.12, 1.0),
      Math.min(track.aspectCollapse / 0.7, 1.0),
    );
    pairScores.push([strike, burst, knockdown]);
  }

  let proximity = 0;
  if (personCount >= 2 && activeTracks.length >= 2) {
    const [first, second] = activeTracks;
    proximity = boxIou(first.bbox, second.bbox) > 0.10 ? 1.0 : 0.0;
    const [scx, scy] = bboxCenter(second.bbox);
    const scxN = scx / Math.max(frameWidth, 1);
    const scyN = scy / Math.max(frameHeight, 1);
    // wrist points for first (kps 9, 10) — normalised 0-1
    const w9x = first.keypoints[9 * 2], w9y = first.keypoints[9 * 2 + 1];
    const w10x = first.keypoints[10 * 2], w10y = first.keypoints[10 * 2 + 1];
    const wd = Math.min(
      Math.hypot(w9x - scxN, w9y - scyN),
      Math.hypot(w10x - scxN, w10y - scyN),
    );
    if (wd < 0.25 * frameWidth / Math.max(frameWidth, 1)) proximity = Math.max(proximity, 1.0);
  } else if (personCount === 1 && secondaryCount > 0) {
    const track = activeTracks[0];
    for (const box of secondaryBoxes) {
      if (boxIou(track.bbox, box) > 0.10) { proximity = 1.0; break; }
      const [bcx, bcy] = bboxCenter(box);
      const bcxN = bcx / Math.max(frameWidth, 1), bcyN = bcy / Math.max(frameHeight, 1);
      const w9x = track.keypoints[9 * 2], w9y = track.keypoints[9 * 2 + 1];
      const w10x = track.keypoints[10 * 2], w10y = track.keypoints[10 * 2 + 1];
      const wd = Math.min(
        Math.hypot(w9x - bcxN, w9y - bcyN),
        Math.hypot(w10x - bcxN, w10y - bcyN),
      );
      if (wd < 0.25) { proximity = 1.0; break; }
    }
    if (proximity === 0) proximity = 0.5;
  }

  const strike    = Math.max(...pairScores.map(s => s[0]));
  const burst     = Math.max(...pairScores.map(s => s[1]));
  const knockdown = Math.max(...pairScores.map(s => s[2]));
  let score = Math.max(0, Math.min(1,
    0.35 * strike + 0.20 * burst + 0.30 * proximity + 0.15 * knockdown,
  ));

  if (personCount >= 2 && activeTracks.length >= 2) {
    const [first, second] = activeTracks;
    const hGap = Math.max(0, first.bbox.x1 - second.bbox.x2, second.bbox.x1 - first.bbox.x2);
    const vGap = Math.max(0, first.bbox.y1 - second.bbox.y2, second.bbox.y1 - first.bbox.y2);
    const minBoxGap = Math.hypot(hGap, vGap);

    const strikeKps = [9, 10, 15, 16];
    const limbDists: number[] = [];
    for (const [src, tgt] of [[first, second], [second, first]] as [TrackedPerson, TrackedPerson][]) {
      for (const k of strikeKps) {
        const px = src.keypoints[k * 2] * frameWidth;
        const py = src.keypoints[k * 2 + 1] * frameHeight;
        limbDists.push(pointToBoxDist(px, py, tgt.bbox));
      }
    }
    const minLimbDist = Math.min(...limbDists);
    const hasStrongStrike = pairScores.some(([s, b]) => s >= 0.75 || b >= 0.75);

    if (minBoxGap > 0.25 * frameWidth && minLimbDist > 0.28 * frameWidth && !hasStrongStrike) {
      return Math.min(score, 0.48);
    }

    // Interaction-zone turbulence boost (+0.18)
    const iouVal = boxIou(first.bbox, second.bbox);
    const limbClose = minLimbDist < 0.18 * frameWidth;
    if (interactionTurbulence > TURBULENCE_BOOST_THRESHOLD && (iouVal > 0.05 || limbClose)) {
      score = Math.min(score + TURBULENCE_BOOST_AMOUNT, 1.0);
    }
  }
  return score;
}

// ─── Optical turbulence in interaction zone ────────────────────────────────────
/**
 * Compute mean absolute difference inside the union bbox of the two actors.
 * Returns a value in [0, 1]. Mirrors the Python absdiff logic.
 */
export function computeInteractionTurbulence(
  prevGray: Uint8Array | null,
  curGray: Uint8Array,
  width: number,
  height: number,
  boxes: BBox[],
): number {
  if (!prevGray || boxes.length < 2) return 0;
  const ux1 = Math.max(0, Math.floor(Math.min(...boxes.map(b => b.x1))));
  const uy1 = Math.max(0, Math.floor(Math.min(...boxes.map(b => b.y1))));
  const ux2 = Math.min(width, Math.ceil(Math.max(...boxes.map(b => b.x2))));
  const uy2 = Math.min(height, Math.ceil(Math.max(...boxes.map(b => b.y2))));
  if (ux2 <= ux1 || uy2 <= uy1) return 0;

  let sum = 0, count = 0;
  for (let y = uy1; y < uy2; y++) {
    for (let x = ux1; x < ux2; x++) {
      const idx = y * width + x;
      sum += Math.abs(curGray[idx] - prevGray[idx]);
      count++;
    }
  }
  return count > 0 ? (sum / count) / 255 : 0;
}

/** Convert RGBA ImageData to grayscale Uint8Array (luma = 0.299R + 0.587G + 0.114B) */
export function rgbaToGray(data: Uint8ClampedArray, width: number, height: number): Uint8Array {
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = Math.round(0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]);
  }
  return gray;
}

// ─── Tracker state update (port of _update_tracks) ────────────────────────────
let nextTrackId = 1;

export function createTrack(det: Detection, now: number, frameW: number, frameH: number): TrackedPerson {
  const [cx, cy] = bboxCenter(det.bbox);
  return {
    trackId: nextTrackId++,
    bbox: det.bbox,
    keypoints: det.keypoints.slice() as Float32Array,
    modelPose: det.keypoints.slice() as Float32Array,
    velocity: new Float32Array(FEATURE_COUNT),
    acceleration: new Float32Array(FEATURE_COUNT),
    history: [],
    lastDetectionKps: det.keypoints.slice() as Float32Array,
    lastDetectionTime: now,
    lastSeen: now,
    confidence: det.confidence,
    centerY: cy / Math.max(frameH, 1),
    aspectRatio: bboxAspect(det.bbox),
    centerDrop: 0,
    aspectCollapse: 0,
    activity: 0,
    prevBbox: null,
  };
}

export function updateTracks(
  tracks: Map<number, TrackedPerson>,
  detections: Detection[],
  now: number,
  dt: number,
  frameWidth: number,
  frameHeight: number,
): TrackedPerson[] {
  const available = [...tracks.values()];
  const maxDist = Math.max(frameWidth, frameHeight) * 0.35;

  // nearest-centroid assignment
  const candidates: [number, number, number][] = []; // [dist, trackIdx, detIdx]
  for (let ti = 0; ti < available.length; ti++) {
    const [tcx, tcy] = bboxCenter(available[ti].bbox);
    for (let di = 0; di < detections.length; di++) {
      const [dcx, dcy] = bboxCenter(detections[di].bbox);
      const dist = Math.hypot(tcx - dcx, tcy - dcy);
      if (dist <= maxDist) candidates.push([dist, ti, di]);
    }
  }
  candidates.sort((a, b) => a[0] - b[0]);

  const matchedTracks = new Set<number>();
  const matchedDets   = new Set<number>();
  const assignments   = new Map<number, number>(); // detIdx -> trackIdx

  for (const [, ti, di] of candidates) {
    if (matchedTracks.has(ti) || matchedDets.has(di)) continue;
    matchedTracks.add(ti); matchedDets.add(di);
    assignments.set(di, ti);
  }

  const timeScale = (1 / 15) / Math.max(dt, 0.02);

  for (let di = 0; di < detections.length; di++) {
    const det = detections[di];
    const [dcx, dcy] = bboxCenter(det.bbox);

    if (assignments.has(di)) {
      const track = available[assignments.get(di)!];
      const prevVel = track.velocity.slice() as Float32Array;
      const detDt = Math.max(now - track.lastDetectionTime, 0.02);
      const newVel  = new Float32Array(FEATURE_COUNT);
      const newAcc  = new Float32Array(FEATURE_COUNT);
      for (let i = 0; i < FEATURE_COUNT; i++) {
        const delta = det.keypoints[i] - track.lastDetectionKps[i];
        newVel[i]  = delta / detDt;
        newAcc[i]  = (newVel[i] - prevVel[i]) / detDt;
      }
      track.velocity     = newVel;
      track.acceleration = newAcc;

      // smoothed model pose
      for (let i = 0; i < FEATURE_COUNT; i++) {
        const correction = det.keypoints[i] - track.keypoints[i];
        track.modelPose[i] = Math.max(0, Math.min(1, track.modelPose[i] + correction * timeScale));
      }

      const prevCenterY = track.centerY;
      const prevAspect  = track.aspectRatio;
      track.prevBbox       = track.bbox;
      track.bbox           = det.bbox;
      track.centerY        = dcy / Math.max(frameHeight, 1);
      track.aspectRatio    = bboxAspect(det.bbox);
      track.centerDrop     = Math.max(0, track.centerY - prevCenterY);
      track.aspectCollapse = Math.max(0, prevAspect - track.aspectRatio);

      const strikeVels = [9, 10, 15, 16].map(k => Math.hypot(newVel[k * 2], newVel[k * 2 + 1]));
      track.activity = Math.max(
        track.activity * 0.65,
        Math.max(...strikeVels),
        track.centerDrop / Math.max(dt, 0.02),
        track.aspectCollapse / Math.max(dt, 0.02),
      );

      track.keypoints          = det.keypoints.slice() as Float32Array;
      track.lastDetectionKps   = det.keypoints.slice() as Float32Array;
      track.lastDetectionTime  = now;
      track.confidence         = det.confidence;
      track.lastSeen           = now;
      track.history.push(track.modelPose.slice() as Float32Array);
      if (track.history.length > SEQUENCE_LENGTH) track.history.shift();
    } else {
      const newTrack = createTrack(det, now, frameWidth, frameHeight);
      newTrack.history.push(newTrack.modelPose.slice() as Float32Array);
      tracks.set(newTrack.trackId, newTrack);
    }
  }

  // Propagate unmatched tracks
  for (let ti = 0; ti < available.length; ti++) {
    if (matchedTracks.has(ti)) continue;
    const track = available[ti];
    const age = now - track.lastSeen;
    if (age > 1.0) { tracks.delete(track.trackId); continue; }
    for (let i = 0; i < FEATURE_COUNT; i++) {
      track.keypoints[i] = Math.max(0, Math.min(1, track.keypoints[i] + track.velocity[i] * dt));
      track.modelPose[i] = Math.max(0, Math.min(1, track.modelPose[i] + track.velocity[i] * timeScale * dt));
    }
    track.history.push(track.modelPose.slice() as Float32Array);
    if (track.history.length > SEQUENCE_LENGTH) track.history.shift();
    track.activity *= 0.85;
  }

  // Retain top-4 by (lastSeen, activity, confidence)
  const active = [...tracks.values()].filter(t => now - t.lastSeen <= 1.0);
  active.sort((a, b) =>
    (b.lastSeen - a.lastSeen) || (b.activity - a.activity) || (b.confidence - a.confidence)
  );
  const retained = new Set(active.slice(0, 4).map(t => t.trackId));
  for (const id of tracks.keys()) if (!retained.has(id)) tracks.delete(id);

  return active.slice(0, 4);
}

export function resetTrackIdCounter(): void { nextTrackId = 1; }
