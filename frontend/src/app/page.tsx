"use client";

import { useEffect, useRef, useState } from "react";
import {
  UploadCloud,
  Film,
  ShieldCheck,
  AlertTriangle,
  Activity,
  Clock,
  Cpu,
  Layers,
  Play,
  Check,
  Copy,
  Terminal,
  RefreshCw,
  Flame,
  CheckCircle2,
  Shield,
  Zap,
  Cloud,
  BookOpen,
} from "lucide-react";
import { edgeEngine } from "@/lib/edgeEngine";
import Documentation from "@/components/Documentation";

// ─── Data Types ─────────────────────────────────────────────────────────────
interface TimelinePoint {
  timestamp: number;
  probability: number;
  is_fighting: boolean;
  person_count?: number;
  bio_score?: number;
  min_strike_dist?: number;
  max_iou?: number;
  gate_triggered?: boolean;
}

interface AnalysisReport {
  overall_verdict: "Violence" | "Safe";
  confidence: number;
  incident_timestamps: number[];
  processing_time_sec: number;
  frame_count?: number;
  video_duration_sec?: number;
  video_fps?: number;
  peak_probability?: number;
  is_sports_context?: boolean;
  runtime_backend?: string;
  timeline?: TimelinePoint[];
}

interface SampleItem {
  id: string;
  name: string;
  label: string;
  category: "Safe" | "Violence";
  description: string;
  videoUrl: string;
}

const PRESET_SAMPLES: SampleItem[] = [
  {
    id: "NV_1.mp4",
    name: "Normal Activity (NV_1)",
    label: "Safe Benchmark",
    category: "Safe",
    description: "Multi-person gestures and normal walking without hostile convergence.",
    videoUrl: "/samples/safe_sample.mp4",
  },
  {
    id: "V_1.mp4",
    name: "Physical Altercation (V_1)",
    label: "Violence Benchmark",
    category: "Violence",
    description: "Rapid strike acceleration, close inter-actor proximity, and violent kinematics.",
    videoUrl: "/samples/violence_sample.mp4",
  },
];

const SCAN_STEPS = [
  "YOLOv8-Pose skeletal extraction...",
  "Kalman multi-actor association...",
  "ST-GCN spatial-temporal convolution...",
  "Relational strike distance gating...",
  "Aggregating sequence telemetry...",
];

function inferVideoMimeType(file?: File | null, url?: string): string {
  if (file?.type) return file.type;
  const target = (file?.name || url || "").toLowerCase();
  if (target.endsWith(".avi")) return "video/x-msvideo";
  if (target.endsWith(".mov")) return "video/quicktime";
  if (target.endsWith(".webm")) return "video/webm";
  if (target.endsWith(".mp4") || target.endsWith(".m4v")) return "video/mp4";
  return "";
}

async function runClientSideEdgeInference(
  videoUrl: string,
  mimeType?: string,
  onProgress?: (status: string, percent: number) => void
): Promise<AnalysisReport> {
  if (typeof window === "undefined") {
    throw new Error("Edge inference requires a browser environment.");
  }

  // Create and configure hidden <video> element BEFORE setting src so browsers do not throttle decoding
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = "anonymous";
  video.preload = "auto";
  video.style.position = "fixed";
  video.style.width = "1px";
  video.style.height = "1px";
  video.style.opacity = "0.001";
  video.style.pointerEvents = "none";
  video.style.left = "-9999px";

  // Pre-flight codec validation
  if (mimeType) {
    const canPlay = video.canPlayType(mimeType);
    if (canPlay === "") {
      throw new Error(
        "Unsupported video codec for local processing. Please use an H.264 MP4, or switch to Cloud API mode."
      );
    }
  }

  if (edgeEngine.state !== "ready") {
    onProgress?.("Loading ONNX Runtime & models...", 10);
    await edgeEngine.load((p) => onProgress?.(p.stage, 20));
  }
  edgeEngine.reset();

  document.body.appendChild(video);

  try {
    video.src = videoUrl;
    video.load();

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        video.removeEventListener("loadeddata", onReady);
        video.removeEventListener("canplay", onReady);
        video.removeEventListener("error", onErr);
        clearTimeout(loadTimer);
      };
      const onReady = () => {
        if (!settled && video.readyState >= 2) {
          settled = true;
          cleanup();
          resolve();
        }
      };
      const onErr = () => {
        if (!settled) {
          settled = true;
          cleanup();
          const code = video.error?.code;
          if (code === 3 || code === 4) {
            reject(
              new Error(
                "Unsupported video codec for local processing. Please use an H.264 MP4, or switch to Cloud API mode."
              )
            );
          } else {
            reject(new Error("Unable to decode video stream in browser."));
          }
        }
      };
      const loadTimer = setTimeout(() => {
        if (!settled) {
          if (video.readyState >= 2) {
            settled = true;
            cleanup();
            resolve();
          } else {
            settled = true;
            cleanup();
            reject(
              new Error(
                "Timed out decoding video stream in browser. Please use an H.264 MP4, or switch to Cloud API mode."
              )
            );
          }
        }
      }, 10000);

      video.addEventListener("loadeddata", onReady);
      video.addEventListener("canplay", onReady);
      video.addEventListener("error", onErr);
      if (video.readyState >= 2) onReady();
    });

    const duration = video.duration && isFinite(video.duration) ? video.duration : 4.0;

    // Preserve native video aspect ratio (capped at 640px) so letterboxing matches Python backend
    const maxDim = 640;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const downscale = Math.min(1, maxDim / Math.max(vw, vh));
    const targetWidth = Math.round(vw * downscale);
    const targetHeight = Math.round(vh * downscale);
    const canvas = document.createElement("canvas");
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Canvas 2D context unavailable.");

    // Ensure >= 30 sequential frames across clip so ST-GCN temporal window receives full motion trajectory
    const sampleFps = 6;
    const totalFrames = Math.max(30, Math.min(Math.round(duration * sampleFps), 40));
    const maxSeekTime = Math.max(0, duration - 0.02);
    const stepTime = totalFrames > 1 ? maxSeekTime / (totalFrames - 1) : 0;
    const timeline: TimelinePoint[] = [];
    const incidentTimestamps: number[] = [];
    let peakProb = 0;
    let processedFrames = 0;
    let activeBackend = edgeEngine.backend;
    const tStart = performance.now();

    for (let i = 0; i < totalFrames; i++) {
      const currentTime = Math.min(maxSeekTime, i * stepTime);
      if (Math.abs(video.currentTime - currentTime) > 0.005) {
        await new Promise<void>((resolve) => {
          let done = false;
          const cleanup = () => {
            video.removeEventListener("seeked", onSeeked);
            clearTimeout(timer);
          };
          const onSeeked = () => {
            if (!done) {
              done = true;
              cleanup();
              resolve();
            }
          };
          const timer = setTimeout(() => {
            if (!done) {
              done = true;
              cleanup();
              resolve();
            }
          }, 1500);
          video.addEventListener("seeked", onSeeked);
          video.currentTime = currentTime;
        });
      }

      let frameData: ImageData;
      try {
        ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
        frameData = ctx.getImageData(0, 0, targetWidth, targetHeight);
      } catch (drawErr) {
        console.warn(`Skipping corrupted/unready frame ${i} at ${currentTime.toFixed(2)}s:`, drawErr);
        continue;
      }

      // Force stride=1 on sequentially subsampled client frames to avoid duplicating stale poses
      const pred = await edgeEngine.processFrame(frameData, 1);
      processedFrames++;
      activeBackend = pred.backend;
      const prob = pred?.probability ?? 0;
      if (prob > peakProb) peakProb = prob;

      const isFighting = pred?.isFighting ?? false;
      if (isFighting && (incidentTimestamps.length === 0 || currentTime - incidentTimestamps[incidentTimestamps.length - 1] > 0.5)) {
        incidentTimestamps.push(Number(currentTime.toFixed(2)));
      }

      timeline.push({
        timestamp: Number(currentTime.toFixed(2)),
        probability: Number(prob.toFixed(3)),
        is_fighting: isFighting,
        person_count: pred?.personCount ?? 0,
        bio_score: Number((pred?.bioScore ?? 0).toFixed(2)),
      });

      const percent = Math.min(95, Math.round(20 + (i / totalFrames) * 75));
      if (i % 2 === 0 || i === totalFrames - 1) {
        onProgress?.(`Frame ${i + 1}/${totalFrames} [${activeBackend}]`, percent);
      }
    }

    if (processedFrames === 0) {
      throw new Error(
        "Unsupported video codec for local processing. Please use an H.264 MP4, or switch to Cloud API mode."
      );
    }

    const isViolent = incidentTimestamps.length > 0 || peakProb >= 0.70;
    const processingTime = Number(((performance.now() - tStart) / 1000).toFixed(2));

    return {
      overall_verdict: isViolent ? "Violence" : "Safe",
      confidence: isViolent ? peakProb : Math.max(0.05, 1 - peakProb),
      incident_timestamps: incidentTimestamps,
      processing_time_sec: processingTime,
      frame_count: processedFrames,
      video_duration_sec: Number(duration.toFixed(1)),
      video_fps: Number((processedFrames / Math.max(duration, 0.1)).toFixed(1)),
      peak_probability: Number(peakProb.toFixed(3)),
      is_sports_context: false,
      runtime_backend: activeBackend,
      timeline,
    };
  } finally {
    video.pause();
    video.removeAttribute("src");
    video.load();
    if (video.parentNode) {
      video.parentNode.removeChild(video);
    }
  }
}

export default function VisionKinematicsConsole() {
  const [inferenceMode, setInferenceMode] = useState<"cloud" | "edge">("cloud");
  const [activeScanStatus, setActiveScanStatus] = useState<string | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(PRESET_SAMPLES[0]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string>(PRESET_SAMPLES[0].videoUrl);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisReport, setAnalysisReport] = useState<AnalysisReport | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [wasmIsolated, setWasmIsolated] = useState<boolean>(false);
  const [copiedSnippet, setCopiedSnippet] = useState<boolean>(false);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [scanStepIndex, setScanStepIndex] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setWasmIsolated(
      typeof window !== "undefined" &&
      Boolean((window as any).crossOriginIsolated) &&
      typeof SharedArrayBuffer !== "undefined"
    );
  }, []);

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch("/api/health");
        setApiConnected(res.ok);
      } catch {
        setApiConnected(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!isAnalyzing) {
      setScanStepIndex(0);
      return;
    }
    const timer = setInterval(() => {
      setScanStepIndex((prev) => (prev + 1) % SCAN_STEPS.length);
    }, 1000);
    return () => clearInterval(timer);
  }, [isAnalyzing]);

  const processVideoFile = (file: File, sampleMeta: SampleItem | null = null) => {
    setUploadedFile(file);
    setSelectedSample(sampleMeta);
    setAnalysisReport(null);
    setErrorMsg(null);

    const objectUrl = URL.createObjectURL(file);
    setPreviewVideoUrl(objectUrl);
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
    }
  };

  const handleQuickSelect = async (sample: SampleItem) => {
    const isViolence = sample.category === "Violence";
    const sampleUrl = isViolence ? "/samples/violence_sample.mp4" : "/samples/safe_sample.mp4";
    const fileName = isViolence ? "violence_sample.mp4" : "safe_sample.mp4";

    try {
      setSelectedSample(sample);
      setAnalysisReport(null);
      setErrorMsg(null);

      const response = await fetch(sampleUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${sampleUrl} (${response.status})`);
      }
      const blob = await response.blob();
      const file = new File([blob], fileName, { type: "video/mp4" });
      processVideoFile(file, sample);
    } catch (err: any) {
      console.warn("Could not preload sample blob:", err);
      setSelectedSample(sample);
      setPreviewVideoUrl(sampleUrl);
    }
  };

  useEffect(() => {
    handleQuickSelect(PRESET_SAMPLES[0]);
  }, []);

  const isAllowedVideoFile = (file: File): boolean => {
    const lower = file.name.toLowerCase();
    return (
      file.type.startsWith("video/") ||
      lower.endsWith(".mp4") ||
      lower.endsWith(".avi") ||
      lower.endsWith(".mov") ||
      lower.endsWith(".webm")
    );
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!isAllowedVideoFile(file)) {
      setErrorMsg("Unsupported format: please provide an MP4, AVI, MOV, or WebM video file.");
      return;
    }
    processVideoFile(file, null);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (!isAllowedVideoFile(file)) {
      setErrorMsg("Unsupported format: please provide an MP4, AVI, MOV, or WebM video file.");
      return;
    }
    processVideoFile(file, null);
  };

  const handleRunAnalysis = async () => {
    setIsAnalyzing(true);
    setErrorMsg(null);
    setAnalysisReport(null);
    setActiveScanStatus(null);

    try {
      if (inferenceMode === "edge") {
        setActiveScanStatus("Checking browser video codec compatibility...");
        const videoSourceUrl = previewVideoUrl;
        if (!videoSourceUrl) {
          throw new Error("No video source selected for client inference.");
        }

        const mimeType = inferVideoMimeType(uploadedFile, videoSourceUrl);
        const report = await runClientSideEdgeInference(videoSourceUrl, mimeType, (stage, percent) => {
          setActiveScanStatus(`${stage} (${percent}%)`);
        });
        setAnalysisReport(report);

        const triggers = report?.incident_timestamps ?? [];
        if (Array.isArray(triggers) && triggers.length > 0 && videoRef.current) {
          videoRef.current.currentTime = Math.max(0, (triggers[0] ?? 0) - 0.5);
        }
      } else {
        setActiveScanStatus("Uploading stream to FastAPI engine...");
        let fileToSend = uploadedFile;
        if (!fileToSend && selectedSample) {
          try {
            const isViolence = selectedSample.category === "Violence";
            const sampleUrl = isViolence ? "/samples/violence_sample.mp4" : "/samples/safe_sample.mp4";
            const fileName = isViolence ? "violence_sample.mp4" : "safe_sample.mp4";
            const sRes = await fetch(sampleUrl);
            if (sRes.ok) {
              const blob = await sRes.blob();
              fileToSend = new File([blob], fileName, { type: "video/mp4" });
              setUploadedFile(fileToSend);
            }
          } catch (e) {
            console.warn("Could not fetch sample blob during analysis:", e);
          }
        }

        const formData = new FormData();
        if (fileToSend) {
          formData.append("file", fileToSend);
        } else if (selectedSample) {
          formData.append("sample_name", selectedSample.id);
        } else {
          throw new Error("No video selected for analysis.");
        }

        const res = await fetch("/api/analyze_video", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          let errorDetail = `HTTP ${res.status}`;
          try {
            const errJson = await res.json();
            if (errJson?.detail) {
              errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
            }
          } catch {
            const text = await res.text().catch(() => "");
            if (res.status === 404 || text.includes("404")) {
              errorDetail = "Cloud API unreachable (404). Switch to Edge Mode for client-side inference.";
            }
          }
          throw new Error(errorDetail);
        }

        const report: AnalysisReport = await res.json();
        if (!report || typeof report !== "object") {
          throw new Error("Malformed JSON payload from inference service.");
        }

        setAnalysisReport({ ...report, runtime_backend: "Cloud C++ ORT" });

        const triggers = report?.incident_timestamps ?? [];
        if (Array.isArray(triggers) && triggers.length > 0 && videoRef.current) {
          videoRef.current.currentTime = Math.max(0, (triggers[0] ?? 0) - 0.5);
        }
      }
    } catch (err: any) {
      console.error("Analysis error:", err);
      if (err?.message?.includes("Failed to load chunk") || err?.name === "ChunkLoadError") {
        setErrorMsg("Stale client bundle detected. Hard-refresh (Ctrl+F5) to load the latest build.");
        setAnalysisReport(null);
        return;
      }
      const isEdge = inferenceMode === "edge";
      setErrorMsg(
        isEdge
          ? `Edge Runtime Error: ${err?.message || "Inference failed"}. Try toggling to Cloud API mode.`
          : (err?.message || "Cloud analysis request failed. Check API connectivity.")
      );
      setAnalysisReport(null);
    } finally {
      setIsAnalyzing(false);
      setActiveScanStatus(null);
    }
  };

  const seekToTimestamp = (sec: number | undefined | null) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, sec ?? 0);
      videoRef.current.play().catch(() => {});
    }
  };

  const edgeSnippet = `git clone https://github.com/Parth-Aish/Violence-Detection-Edge.git
cd Violence-Detection-Edge
pip install -r requirements.txt
python backend/run_live.py --source 0`;

  const copySnippet = () => {
    navigator.clipboard.writeText(edgeSnippet);
    setCopiedSnippet(true);
    setTimeout(() => setCopiedSnippet(false), 2500);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 flex flex-col font-sans selection:bg-zinc-700 selection:text-white">
      {/* ── Technical Status Header ───────────────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-zinc-800 bg-[#09090b]/95 px-6 py-3">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-md bg-zinc-900 border border-zinc-700 flex items-center justify-center">
              <Activity className="h-4 w-4 text-zinc-200" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
                  EDGE VISION // KINEMATICS CONSOLE
                </h1>
                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-400 border border-zinc-800">
                  ST-GCN v4
                </span>
              </div>
              <p className="text-[11px] font-mono text-zinc-500">
                COCO-17 Skeleton Graph • Kalman Occlusion Tracker • Relational Strike Gate
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
            {/* Cross-Origin Isolation Status Badge */}
            <div
              title="COOP/COEP headers enable SharedArrayBuffer for multi-threaded WebAssembly"
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 border ${
                wasmIsolated
                  ? "border-emerald-800/70 bg-emerald-950/30 text-emerald-400"
                  : "border-amber-800/70 bg-amber-950/30 text-amber-400"
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${wasmIsolated ? "bg-emerald-400" : "bg-amber-400"}`} />
              <span>{wasmIsolated ? "COOP/COEP: ISOLATED (MT)" : "ISOLATION: 1T FALLBACK"}</span>
            </div>

            {/* Active Execution Posture Badge */}
            <div
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 border ${
                inferenceMode === "edge"
                  ? "border-emerald-800/70 bg-emerald-950/30 text-emerald-300"
                  : "border-indigo-800/70 bg-indigo-950/30 text-indigo-300"
              }`}
            >
              {inferenceMode === "edge" ? (
                <>
                  <Shield className="h-3 w-3 text-emerald-400" />
                  <span>PRIVACY: LOCAL ISOLATED</span>
                </>
              ) : (
                <>
                  <Cloud className="h-3 w-3 text-indigo-400" />
                  <span>MODE: CLOUD BATCH API</span>
                </>
              )}
            </div>

            {/* Backend Health Status */}
            <div className="flex items-center gap-1.5 rounded border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-zinc-300">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  apiConnected === true
                    ? "bg-emerald-400"
                    : apiConnected === false
                    ? "bg-rose-500"
                    : "bg-amber-400"
                }`}
              />
              <span>
                {apiConnected === true
                  ? "API: ONLINE"
                  : apiConnected === false
                  ? "API: OFFLINE"
                  : "API: PROBING"}
              </span>
            </div>

            {/* In-App Documentation Anchor */}
            <a
              href="#about-documentation-section"
              className="flex items-center gap-1.5 rounded border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 px-2.5 py-1 text-zinc-200 transition"
            >
              <BookOpen className="h-3 w-3 text-zinc-400" />
              <span>DOCS &amp; SPECS</span>
            </a>
          </div>
        </div>
      </header>

      {/* ── Main Engineering Workbench ────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* ── Two-Column Console ──────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Execution Config, Input Stream & Video Viewport (7 cols) */}
          <div className="lg:col-span-7 space-y-5">
            {/* Execution Architecture Selector */}
            <section className="rounded-lg border border-zinc-800 bg-[#111114] p-4 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4 text-zinc-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-zinc-200">
                      Inference Execution Engine
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase border ${
                        inferenceMode === "edge"
                          ? "bg-emerald-950/50 text-emerald-400 border-emerald-800/70"
                          : "bg-indigo-950/50 text-indigo-300 border-indigo-800/70"
                      }`}
                    >
                      {inferenceMode === "edge" ? "Zero-Upload Client" : "FastAPI Server"}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 mt-1">
                    {inferenceMode === "cloud"
                      ? "Executes YOLOv8-Pose + ST-GCN on native C++ ONNX Runtime (~50 FPS batch throughput)."
                      : "Executes YOLOv8-Pose + ST-GCN locally inside the browser via WebGPU / multi-threaded WASM SIMD."}
                  </p>
                </div>

                {/* Crisp Mode Switcher */}
                <div className="flex items-center rounded-md bg-zinc-950 p-1 border border-zinc-800 self-start sm:self-auto">
                  <button
                    type="button"
                    id="toggle-cloud-mode-btn"
                    onClick={() => {
                      setInferenceMode("cloud");
                      setErrorMsg(null);
                    }}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition ${
                      inferenceMode === "cloud"
                        ? "bg-indigo-600 text-white"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    <Cloud className="h-3.5 w-3.5" />
                    <span>Cloud API</span>
                  </button>
                  <button
                    type="button"
                    id="toggle-edge-mode-btn"
                    onClick={() => {
                      setInferenceMode("edge");
                      setErrorMsg(null);
                    }}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition ${
                      inferenceMode === "edge"
                        ? "bg-emerald-600 text-white"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    <Cpu className="h-3.5 w-3.5" />
                    <span>Edge WASM</span>
                  </button>
                </div>
              </div>

              {/* Pro-Tip Callout for Long-Form Videos */}
              <div
                id="local-execution-protip"
                className="rounded-md border border-amber-700/60 bg-amber-950/20 px-3.5 py-2.5 text-xs text-zinc-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
              >
                <div className="flex items-start gap-2.5 leading-relaxed">
                  <Zap className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <p className="text-[11px] text-zinc-300">
                    <span className="font-semibold text-amber-300">
                      Pro-Tip for Long-Form Videos:
                    </span>{" "}
                    For videos longer than 30 seconds, explicitly select{" "}
                    <span className="font-mono text-emerald-300 font-semibold">
                      Privacy: High (Local WASM)
                    </span>
                    . Cloud mode requires uploading heavy payloads over the network, whereas Local
                    Mode runs instantly on your device&apos;s multi-core CPU/GPU via WebAssembly,
                    resulting in significantly faster processing and zero network bottleneck.
                  </p>
                </div>
                {inferenceMode !== "edge" && (
                  <button
                    type="button"
                    onClick={() => {
                      setInferenceMode("edge");
                      setErrorMsg(null);
                    }}
                    className="flex-shrink-0 px-2.5 py-1 rounded text-[11px] font-mono bg-emerald-950/80 hover:bg-emerald-900 text-emerald-300 border border-emerald-700/80 transition self-start sm:self-center"
                  >
                    Select Local WASM
                  </button>
                )}
              </div>
            </section>

            {/* Input Source & Benchmark Presets */}
            <section className="rounded-lg border border-zinc-800 bg-[#111114] p-4 space-y-4">
              <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2.5">
                <div className="flex items-center gap-2">
                  <Film className="h-4 w-4 text-zinc-400" />
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-200">
                    Input Stream Source
                  </h2>
                </div>
                <span className="text-[11px] font-mono text-zinc-500">MP4 / AVI / MOV (MAX 150MB)</span>
              </div>

              {/* File Dropzone */}
              <div
                id="video-dropzone"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border border-dashed rounded-md p-5 text-center cursor-pointer transition flex items-center justify-center gap-4 ${
                  isDragOver
                    ? "border-indigo-500 bg-indigo-950/20"
                    : uploadedFile
                    ? "border-emerald-700/60 bg-zinc-950/60"
                    : "border-zinc-700 bg-zinc-950/40 hover:border-zinc-500"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime"
                  onChange={handleFileUpload}
                  className="hidden"
                />

                <div className="h-10 w-10 rounded bg-zinc-900 border border-zinc-800 flex items-center justify-center flex-shrink-0">
                  {uploadedFile ? (
                    <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                  ) : (
                    <UploadCloud className="h-5 w-5 text-zinc-400" />
                  )}
                </div>

                <div className="text-left">
                  {uploadedFile ? (
                    <>
                      <p className="text-xs font-mono font-semibold text-zinc-100">{uploadedFile.name}</p>
                      <p className="text-[11px] font-mono text-emerald-400 mt-0.5">
                        {(((uploadedFile?.size ?? 0) / (1024 * 1024))).toFixed(2)} MB • STAGED FOR ANALYSIS
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-xs font-medium text-zinc-200">
                        Drop video payload here, or <span className="text-indigo-400 underline">select file</span>
                      </p>
                      <p className="text-[11px] text-zinc-500 mt-0.5">
                        Or load a verified validation benchmark below
                      </p>
                    </>
                  )}
                </div>
              </div>

              {/* Validation Benchmark Presets */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {PRESET_SAMPLES.map((sample) => {
                  const isSelected = selectedSample?.id === sample.id;
                  const isViolence = sample.category === "Violence";
                  return (
                    <button
                      key={sample.id}
                      id={`sample-btn-${sample.id.replace(/\W/g, "-")}`}
                      onClick={() => handleQuickSelect(sample)}
                      disabled={isAnalyzing}
                      className={`text-left p-3 rounded-md border transition flex flex-col justify-between ${
                        isSelected
                          ? isViolence
                            ? "border-rose-500/80 bg-rose-950/20"
                            : "border-emerald-500/80 bg-emerald-950/20"
                          : "border-zinc-800 bg-zinc-950/50 hover:border-zinc-700"
                      } disabled:opacity-50`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-mono font-semibold text-xs text-zinc-100 flex items-center gap-1.5">
                          {isViolence ? (
                            <Flame className="h-3.5 w-3.5 text-rose-400" />
                          ) : (
                            <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                          )}
                          {sample.name}
                        </span>
                        <span
                          className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border ${
                            isViolence
                              ? "bg-rose-950/60 text-rose-300 border-rose-800/80"
                              : "bg-emerald-950/60 text-emerald-300 border-emerald-800/80"
                          }`}
                        >
                          {sample.category}
                        </span>
                      </div>
                      <p className="text-[11px] text-zinc-400 leading-snug">{sample.description}</p>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* Video Canvas & Execution Control */}
            <section className="rounded-lg border border-zinc-800 bg-[#111114] p-4 space-y-3.5">
              <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
                <div className="flex items-center gap-2">
                  <Play className="h-3.5 w-3.5 text-zinc-400" />
                  <span className="text-xs font-semibold uppercase tracking-wider text-zinc-200">
                    Video Viewport
                  </span>
                </div>
                <span className="text-[11px] font-mono text-zinc-400">
                  {uploadedFile ? uploadedFile.name : selectedSample?.name}
                </span>
              </div>

              <div className="relative aspect-video w-full rounded bg-black border border-zinc-800 overflow-hidden">
                <video
                  ref={videoRef}
                  id="preview-video-player"
                  src={previewVideoUrl}
                  controls
                  playsInline
                  className="w-full h-full object-contain"
                />
              </div>

              <button
                id="start-analysis-btn"
                onClick={handleRunAnalysis}
                disabled={isAnalyzing}
                className={`w-full py-3 px-4 rounded-md font-mono font-semibold text-xs uppercase tracking-wider transition flex items-center justify-center gap-2.5 border ${
                  isAnalyzing
                    ? "bg-zinc-900 text-zinc-400 cursor-not-allowed border-zinc-700"
                    : inferenceMode === "edge"
                    ? "bg-emerald-600 hover:bg-emerald-500 text-white border-emerald-500"
                    : "bg-zinc-100 hover:bg-white text-zinc-950 border-zinc-200"
                }`}
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    <span>{activeScanStatus || "EXECUTING PIPELINE..."}</span>
                  </>
                ) : inferenceMode === "edge" ? (
                  <>
                    <Zap className="h-3.5 w-3.5" />
                    <span>Execute Local Edge Analysis (Client ONNX)</span>
                  </>
                ) : (
                  <>
                    <Cloud className="h-3.5 w-3.5" />
                    <span>Execute Cloud Analysis (FastAPI Service)</span>
                  </>
                )}
              </button>

              {errorMsg && (
                <div
                  id="analysis-error-banner"
                  className="rounded-md border border-rose-800 bg-rose-950/30 p-3 text-xs font-mono text-rose-300 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                >
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-rose-400 flex-shrink-0" />
                    <span>{errorMsg}</span>
                  </div>
                  {inferenceMode === "edge" ? (
                    <button
                      type="button"
                      onClick={() => {
                        setInferenceMode("cloud");
                        setErrorMsg(null);
                      }}
                      className="px-2.5 py-1 rounded text-[11px] bg-indigo-900/50 hover:bg-indigo-800/60 text-indigo-200 border border-indigo-700 self-start sm:self-auto"
                    >
                      Switch to Cloud
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => {
                        setInferenceMode("edge");
                        setErrorMsg(null);
                      }}
                      className="px-2.5 py-1 rounded text-[11px] bg-emerald-900/50 hover:bg-emerald-800/60 text-emerald-200 border border-emerald-700 self-start sm:self-auto"
                    >
                      Switch to Edge
                    </button>
                  )}
                </div>
              )}
            </section>
          </div>

          {/* Right Column: Dense Telemetry & Verdict Console (5 cols) */}
          <div className="lg:col-span-5 space-y-5">
            <section className="rounded-lg border border-zinc-800 bg-[#111114] p-4 space-y-4">
              <div className="flex items-center justify-between pb-2.5 border-b border-zinc-800">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-zinc-400" />
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-200">
                    Telemetry &amp; Verdict
                  </h2>
                </div>
                <div className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="px-2 py-0.5 rounded border border-zinc-700 bg-zinc-900 text-zinc-300">
                    {analysisReport?.runtime_backend || (inferenceMode === "edge" ? "Client ONNX" : "Cloud ORT")}
                  </span>
                  <span className="text-zinc-500">
                    {analysisReport ? `${analysisReport?.processing_time_sec ?? 0}s` : "IDLE"}
                  </span>
                </div>
              </div>

              {isAnalyzing && (
                <div id="analyzing-state" className="py-10 px-4 text-center space-y-3 border border-zinc-800/80 rounded bg-zinc-950/60">
                  <RefreshCw className="h-6 w-6 text-zinc-300 animate-spin mx-auto" />
                  <div className="space-y-1">
                    <div className="text-xs font-mono font-semibold uppercase tracking-wider text-zinc-200">
                      Running Inference Pipeline
                    </div>
                    <p className="text-xs font-mono text-indigo-400">
                      {activeScanStatus || SCAN_STEPS[scanStepIndex]}
                    </p>
                  </div>
                </div>
              )}

              {!isAnalyzing && !analysisReport && (
                <div id="idle-report-state" className="py-12 px-4 text-center space-y-2 border border-zinc-800/60 rounded bg-zinc-950/40">
                  <Terminal className="h-5 w-5 text-zinc-500 mx-auto" />
                  <h3 className="text-xs font-mono uppercase tracking-wider text-zinc-300">
                    Awaiting Stream Ingestion
                  </h3>
                  <p className="text-xs text-zinc-500 max-w-xs mx-auto">
                    Select a validation sample or drop a local MP4 file to inspect ST-GCN graph telemetry.
                  </p>
                </div>
              )}

              {!isAnalyzing && analysisReport && (
                <div id="analysis-report-view" className="space-y-4">
                  {/* Verdict Card */}
                  {analysisReport.overall_verdict === "Violence" ? (
                    <div
                      id="verdict-banner"
                      className="rounded-md border border-rose-600/80 bg-rose-950/30 p-4 space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="inline-flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase tracking-wider text-rose-300">
                          <AlertTriangle className="h-3.5 w-3.5 text-rose-400" />
                          Hostile Kinematics Confirmed
                        </span>
                        <span className="text-[11px] font-mono text-rose-300 tabular-nums">
                          {(((analysisReport?.confidence ?? 0) * 100)).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-xl font-mono font-bold tracking-tight text-rose-400">
                        VIOLENCE DETECTED
                      </div>
                      <div className="h-1.5 w-full rounded-sm bg-zinc-950 overflow-hidden border border-rose-900/50">
                        <div
                          className="h-full bg-rose-500"
                          style={{ width: `${Math.min(100, Math.max(0, (analysisReport?.confidence ?? 0) * 100))}%` }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div
                      id="verdict-banner"
                      className="rounded-md border border-emerald-700/70 bg-emerald-950/25 p-4 space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="inline-flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase tracking-wider text-emerald-300">
                          <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                          Normal Kinematics Verified
                        </span>
                        <span className="text-[11px] font-mono text-emerald-300 tabular-nums">
                          {(((analysisReport?.confidence ?? 0) * 100)).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-xl font-mono font-bold tracking-tight text-emerald-400">
                        NORMAL / SECURE
                      </div>
                      <div className="h-1.5 w-full rounded-sm bg-zinc-950 overflow-hidden border border-emerald-900/50">
                        <div
                          className="h-full bg-emerald-500"
                          style={{ width: `${Math.min(100, Math.max(0, (analysisReport?.confidence ?? 0) * 100))}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Dense Telemetry Grid */}
                  <div className="grid grid-cols-2 gap-2.5 text-xs">
                    <div className="rounded border border-zinc-800 bg-zinc-950/70 p-2.5">
                      <span className="text-zinc-500 text-[10px] font-mono uppercase flex items-center gap-1">
                        <Clock className="h-3 w-3" /> Runtime
                      </span>
                      <p className="font-mono font-semibold text-zinc-100 mt-0.5 tabular-nums">
                        {analysisReport?.processing_time_sec ?? 0}s
                      </p>
                    </div>
                    <div className="rounded border border-zinc-800 bg-zinc-950/70 p-2.5">
                      <span className="text-zinc-500 text-[10px] font-mono uppercase flex items-center gap-1">
                        <Layers className="h-3 w-3" /> Frames
                      </span>
                      <p className="font-mono font-semibold text-zinc-100 mt-0.5 tabular-nums">
                        {analysisReport?.frame_count ?? "—"}
                      </p>
                    </div>
                    <div className="rounded border border-zinc-800 bg-zinc-950/70 p-2.5">
                      <span className="text-zinc-500 text-[10px] font-mono uppercase flex items-center gap-1">
                        <Film className="h-3 w-3" /> Duration / Rate
                      </span>
                      <p className="font-mono font-semibold text-zinc-100 mt-0.5 tabular-nums">
                        {analysisReport?.video_duration_sec ?? "—"}s @ {analysisReport?.video_fps ?? 30} fps
                      </p>
                    </div>
                    <div className="rounded border border-zinc-800 bg-zinc-950/70 p-2.5">
                      <span className="text-zinc-500 text-[10px] font-mono uppercase flex items-center gap-1">
                        <Activity className="h-3 w-3" /> Peak P(Violence)
                      </span>
                      <p className="font-mono font-semibold text-zinc-100 mt-0.5 tabular-nums">
                        {analysisReport?.peak_probability != null
                          ? `${(((analysisReport?.peak_probability ?? 0) * 100)).toFixed(1)}%`
                          : "—"}
                      </p>
                    </div>
                  </div>

                  {/* Incident Timestamps */}
                  <div className="space-y-2 pt-2 border-t border-zinc-800">
                    <div className="flex items-center justify-between font-mono text-[11px]">
                      <span className="text-zinc-300 uppercase">Incident Triggers</span>
                      <span className="text-zinc-500">
                        {(analysisReport?.incident_timestamps ?? []).length} events
                      </span>
                    </div>

                    {(analysisReport?.incident_timestamps ?? []).length === 0 ? (
                      <div className="rounded border border-zinc-800 bg-zinc-950/50 p-3 text-xs font-mono text-zinc-500">
                        No temporal windows breached strike gate threshold.
                      </div>
                    ) : (
                      <div id="incident-timestamps-list" className="flex flex-wrap gap-1.5 max-h-28 overflow-y-auto">
                        {(analysisReport?.incident_timestamps ?? []).map((ts, idx) => (
                          <button
                            key={`ts-${idx}-${ts}`}
                            onClick={() => seekToTimestamp(ts ?? 0)}
                            className="px-2.5 py-1 rounded text-xs font-mono bg-rose-950/60 hover:bg-rose-900/70 text-rose-300 border border-rose-700/70 transition flex items-center gap-1 tabular-nums"
                            title={`Seek to ${(ts ?? 0).toFixed(2)}s`}
                          >
                            <Flame className="h-3 w-3 text-rose-400" />
                            <span>{(ts ?? 0).toFixed(2)}s</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Telemetry Table */}
                  {(analysisReport?.timeline ?? []).length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-zinc-800">
                      <span className="text-[11px] font-mono uppercase text-zinc-400 block">
                        Sampled Sequence Telemetry
                      </span>
                      <div className="max-h-56 overflow-y-auto border border-zinc-800 rounded bg-zinc-950 divide-y divide-zinc-800/70 font-mono text-[11px] tabular-nums">
                        {(analysisReport?.timeline ?? []).map((point, idx) => (
                          <div
                            key={`tl-${idx}`}
                            onClick={() => seekToTimestamp(point?.timestamp ?? 0)}
                            className={`flex items-center justify-between px-3 py-1.5 cursor-pointer transition ${
                              point?.is_fighting
                                ? "bg-rose-950/30 text-rose-300 hover:bg-rose-950/50"
                                : "text-zinc-400 hover:bg-zinc-900"
                            }`}
                          >
                            <span className="w-14 font-semibold">{(point?.timestamp ?? 0).toFixed(2)}s</span>
                            <span className="w-16">P={(((point?.probability ?? 0) * 100)).toFixed(1)}%</span>
                            <span className="w-16 text-zinc-500">N={point?.person_count ?? 0}</span>
                            <span
                              className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                point?.is_fighting
                                  ? "bg-rose-600 text-white"
                                  : "bg-zinc-800 text-zinc-400"
                              }`}
                            >
                              {point?.is_fighting ? "ALERT" : "SAFE"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </section>

            {/* Compact CLI Native Edge Runner Reference */}
            <section
              id="edge-hardware-banner"
              className="rounded-lg border border-zinc-800 bg-[#111114] p-4 space-y-2.5"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs font-mono text-zinc-300">
                  <Terminal className="h-3.5 w-3.5 text-emerald-400" />
                  <span>NATIVE PYTHON RTSP / WEBCAM CLI (~50 FPS)</span>
                </div>
                <button
                  id="copy-edge-command-btn"
                  onClick={copySnippet}
                  className="flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-mono bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-700 transition"
                >
                  {copiedSnippet ? (
                    <>
                      <Check className="h-3 w-3 text-emerald-400" />
                      <span className="text-emerald-400">COPIED</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span>COPY</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="font-mono text-[11px] text-zinc-400 bg-zinc-950 border border-zinc-800 rounded p-2.5 overflow-x-auto leading-relaxed">
                <code>{edgeSnippet}</code>
              </pre>
            </section>
          </div>
        </div>

        {/* ── In-App Technical Documentation & Benchmarks ─────────────────── */}
        <Documentation />
      </main>
    </div>
  );
}
