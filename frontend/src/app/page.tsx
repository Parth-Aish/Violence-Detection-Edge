"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
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
  Sparkles,
  RefreshCw,
  FileVideo,
  Radar,
  Flame,
  CheckCircle2,
  ChevronRight,
  Shield,
  Zap,
} from "lucide-react";

// ─── API Config ─────────────────────────────────────────────────────────────
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : typeof window !== "undefined"
      ? window.location.origin
      : "http://localhost:8000");

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
  timeline?: TimelinePoint[];
}

interface SampleItem {
  id: string;
  name: string;
  label: string;
  category: "Safe" | "Violence";
  description: string;
  videoUrl: string;
  badgeColor: string;
}

const PRESET_SAMPLES: SampleItem[] = [
  {
    id: "NV_1.mp4",
    name: "Normal Activity (NV_1)",
    label: "Safe Scenario",
    category: "Safe",
    description: "Multi-person physical gestures and normal walking without hostile convergence.",
    videoUrl: "/samples/safe_sample.mp4",
    badgeColor: "emerald",
  },
  {
    id: "V_1.mp4",
    name: "Physical Altercation (V_1)",
    label: "Violence Scenario",
    category: "Violence",
    description: "Rapid strike acceleration, close inter-actor proximity, and violent kinematics.",
    videoUrl: "/samples/violence_sample.mp4",
    badgeColor: "rose",
  },
];

const SCAN_STEPS = [
  "Extracting Skeletal Kinematics (YOLOv8-Pose)...",
  "Tracking Multi-Actor Occlusions (Kalman Momentum)...",
  "Computing Spatial-Temporal Graphs (ST-GCN Topology)...",
  "Evaluating Cross-Person Relational Strike Gate...",
  "Synthesizing Biomechanical Telemetry...",
];

export default function AppleGradeVisionDashboard() {
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(PRESET_SAMPLES[0]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string>(PRESET_SAMPLES[0].videoUrl);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisReport, setAnalysisReport] = useState<AnalysisReport | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [copiedSnippet, setCopiedSnippet] = useState<boolean>(false);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [scanStepIndex, setScanStepIndex] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Check backend health
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_URL}/api/health`);
        setApiConnected(res.ok);
      } catch {
        setApiConnected(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Active Waiting sequence cycling
  useEffect(() => {
    if (!isAnalyzing) {
      setScanStepIndex(0);
      return;
    }
    const timer = setInterval(() => {
      setScanStepIndex((prev) => (prev + 1) % SCAN_STEPS.length);
    }, 1200);
    return () => clearInterval(timer);
  }, [isAnalyzing]);

  // Handle sample selection
  const handleSelectSample = (sample: SampleItem) => {
    setSelectedSample(sample);
    setUploadedFile(null);
    setPreviewVideoUrl(sample.videoUrl);
    setAnalysisReport(null);
    setErrorMsg(null);
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
    }
  };

  // Handle uploaded video file
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.includes("mp4") && !file.name.toLowerCase().endsWith(".mp4")) {
      setErrorMsg("Please upload an MP4 video file.");
      return;
    }

    setUploadedFile(file);
    setSelectedSample(null);
    setAnalysisReport(null);
    setErrorMsg(null);

    const objectUrl = URL.createObjectURL(file);
    setPreviewVideoUrl(objectUrl);
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
    }
  };

  // Drag and drop handlers
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
    if (!file.name.toLowerCase().endsWith(".mp4")) {
      setErrorMsg("Please upload an MP4 video file.");
      return;
    }
    setUploadedFile(file);
    setSelectedSample(null);
    setAnalysisReport(null);
    setErrorMsg(null);

    const objectUrl = URL.createObjectURL(file);
    setPreviewVideoUrl(objectUrl);
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
    }
  };

  // Run batch video analysis
  const handleRunAnalysis = async () => {
    setIsAnalyzing(true);
    setErrorMsg(null);
    setAnalysisReport(null);

    try {
      const formData = new FormData();
      if (uploadedFile) {
        formData.append("file", uploadedFile);
      } else if (selectedSample) {
        formData.append("sample_name", selectedSample.id);
      } else {
        throw new Error("No video selected for analysis.");
      }

      const res = await fetch(`${API_URL}/api/analyze_video`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned error status ${res.status}`);
      }

      const report: AnalysisReport = await res.json();
      setAnalysisReport(report);

      // Jump video to first incident
      if (report.incident_timestamps && report.incident_timestamps.length > 0 && videoRef.current) {
        videoRef.current.currentTime = Math.max(0, report.incident_timestamps[0] - 0.5);
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to process video analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Jump video to timestamp
  const seekToTimestamp = (sec: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, sec);
      videoRef.current.play().catch(() => {});
    }
  };

  const edgeSnippet = `git clone https://github.com/Parth-Aish/Violence-Detection-Edge.git
cd Violence-Detection-Edge
pip install -r requirements.txt
python run_live.py --source 0`;

  const copySnippet = () => {
    navigator.clipboard.writeText(edgeSnippet);
    setCopiedSnippet(true);
    setTimeout(() => setCopiedSnippet(false), 2500);
  };

  return (
    <div className="min-h-screen bg-[#070709] text-zinc-100 flex flex-col font-sans selection:bg-rose-500/30 selection:text-white relative overflow-x-hidden">
      {/* ── Ambient Apple-Grade Background Glows ─────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div className="absolute top-[-15%] left-[20%] w-[650px] h-[650px] bg-gradient-to-br from-indigo-600/15 via-purple-600/10 to-transparent rounded-full blur-[140px] transform-gpu" />
        <div className="absolute bottom-[-10%] right-[10%] w-[550px] h-[550px] bg-gradient-to-tl from-cyan-600/10 via-emerald-600/5 to-transparent rounded-full blur-[130px] transform-gpu" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.08),rgba(255,255,255,0))]" />
      </div>

      {/* ── Header: Apple Frosted Glass Bar ───────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-white/[0.08] bg-[#070709]/70 backdrop-blur-2xl px-6 py-3.5 transition-all">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3.5">
            <div className="h-10 w-10 rounded-2xl bg-gradient-to-tr from-white/[0.12] to-white/[0.04] border border-white/20 flex items-center justify-center shadow-lg shadow-black/40 backdrop-blur-xl">
              <Activity className="h-5 w-5 text-zinc-100" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-sm font-semibold tracking-tight text-white">
                  Vision Kinematics
                </h1>
                <span className="text-[10px] font-medium tracking-wide uppercase px-2 py-0.5 rounded-full bg-white/[0.06] text-zinc-300 border border-white/10">
                  ST-GCN • Edge & Cloud
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 font-normal">
                Spatial-Temporal Graph Convolutions & Relational Strike Gating
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-xs backdrop-blur-xl shadow-inner">
              <span
                className={`h-2 w-2 rounded-full transition-all duration-300 ${
                  apiConnected === true
                    ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]"
                    : apiConnected === false
                    ? "bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.8)]"
                    : "bg-amber-400 animate-pulse"
                }`}
              />
              <span className="text-zinc-300 text-[11px] font-medium">
                {apiConnected === true
                  ? "Cloud API Connected"
                  : apiConnected === false
                  ? "API Offline (Run Locally)"
                  : "Connecting..."}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Dashboard Body ───────────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 space-y-7 z-10">
        {/* ── Heavyweight Edge Notice Glass Pill ───────────────────────────── */}
        <section
          id="edge-hardware-banner"
          className="relative overflow-hidden rounded-3xl border border-white/[0.09] bg-white/[0.03] backdrop-blur-2xl p-6 shadow-[0_8px_32px_0_rgba(0,0,0,0.36)]"
        >
          <div className="relative flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div className="space-y-2 max-w-2xl">
              <div className="inline-flex items-center gap-2 text-[11px] font-semibold tracking-wider uppercase text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Heavyweight Local Edge Mode Available
              </div>
              <h2 className="text-lg md:text-xl font-bold tracking-tight text-white">
                50+ FPS Real-Time Live Surveillance Inference
              </h2>
              <p className="text-xs md:text-sm text-zinc-300/80 leading-relaxed">
                Connect directly to your local webcam or RTSP security camera. Runs full multi-core CPU/GPU
                acceleration with zero network latency, persistent OpenCV skeletal HUDs, and MobileNetV3 Scene Gating.
              </p>
            </div>

            <div className="flex-shrink-0 w-full lg:w-auto">
              <div className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-3.5 shadow-inner">
                <div className="flex items-center justify-between gap-6 pb-2 mb-2 border-b border-white/[0.08] text-[11px]">
                  <div className="flex items-center gap-1.5 text-zinc-400">
                    <Terminal className="h-3.5 w-3.5 text-emerald-400" />
                    <span className="font-mono">Quick Edge Run</span>
                  </div>
                  <button
                    id="copy-edge-command-btn"
                    onClick={copySnippet}
                    className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[11px] font-medium bg-white/[0.08] hover:bg-white/[0.15] text-white transition active:scale-95"
                  >
                    {copiedSnippet ? (
                      <>
                        <Check className="h-3 w-3 text-emerald-400" />
                        <span className="text-emerald-300">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3 text-zinc-300" />
                        <span>Copy Command</span>
                      </>
                    )}
                  </button>
                </div>
                <pre className="font-mono text-xs text-zinc-300 select-all overflow-x-auto whitespace-pre leading-relaxed pr-4">
                  <code>{edgeSnippet}</code>
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* ── Two-Column Interactive Analysis Console ─────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-7 items-start">
          {/* Left Column: Input Dropzone & Video Preview (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            {/* Hick's Law Action Hub: Massive Drag & Drop Zone */}
            <div className="rounded-3xl border border-white/[0.08] bg-white/[0.025] backdrop-blur-2xl p-6 shadow-[0_8px_32px_0_rgba(0,0,0,0.36)] space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold tracking-tight text-white flex items-center gap-2">
                    <Film className="h-4 w-4 text-zinc-300" />
                    Select or Drop Video
                  </h3>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    Choose from standard benchmark presets or test with your custom MP4
                  </p>
                </div>
                <span className="text-[11px] font-mono text-zinc-500 uppercase tracking-wider">
                  Input Stream
                </span>
              </div>

              {/* Massive Drag-and-Drop Dropzone */}
              <motion.div
                id="video-dropzone"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                whileHover={{ scale: 1.005 }}
                whileTap={{ scale: 0.995 }}
                className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center gap-3 ${
                  isDragOver
                    ? "border-indigo-400 bg-indigo-500/10 shadow-[0_0_30px_rgba(99,102,241,0.25)]"
                    : uploadedFile
                    ? "border-emerald-500/40 bg-emerald-500/[0.04]"
                    : "border-white/[0.12] bg-white/[0.015] hover:border-white/25 hover:bg-white/[0.03]"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime"
                  onChange={handleFileUpload}
                  className="hidden"
                />

                <div className="h-14 w-14 rounded-2xl bg-white/[0.05] border border-white/10 flex items-center justify-center text-zinc-200 shadow-inner">
                  {uploadedFile ? (
                    <CheckCircle2 className="h-7 w-7 text-emerald-400" />
                  ) : (
                    <UploadCloud className="h-7 w-7 text-zinc-300" />
                  )}
                </div>

                <div className="space-y-1">
                  {uploadedFile ? (
                    <div>
                      <p className="text-sm font-semibold text-white tracking-tight">
                        {uploadedFile.name}
                      </p>
                      <p className="text-xs text-emerald-400 font-mono mt-0.5">
                        {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for ST-GCN analysis
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-sm font-medium text-zinc-200">
                        Drop video file here, or <span className="text-indigo-400 underline underline-offset-4">browse files</span>
                      </p>
                      <p className="text-xs text-zinc-400 mt-1">
                        MP4, AVI, or MOV format • Up to 150 MB
                      </p>
                    </div>
                  )}
                </div>
              </motion.div>

              {/* Hick's Law: 2 Dedicated Quick Select Buttons */}
              <div className="space-y-2">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider block">
                  Or Test Preset Benchmarks
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                  {PRESET_SAMPLES.map((sample) => {
                    const isSelected = selectedSample?.id === sample.id && !uploadedFile;
                    const isViolence = sample.category === "Violence";

                    return (
                      <motion.button
                        key={sample.id}
                        id={`sample-btn-${sample.id.replace(/\W/g, "-")}`}
                        onClick={() => handleSelectSample(sample)}
                        disabled={isAnalyzing}
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.985 }}
                        className={`relative text-left p-4 rounded-2xl border transition-all duration-300 flex flex-col justify-between ${
                          isSelected
                            ? isViolence
                              ? "border-rose-500/60 bg-rose-950/25 shadow-[0_0_24px_-4px_rgba(244,63,94,0.35)] ring-1 ring-rose-500/40"
                              : "border-emerald-500/60 bg-emerald-950/25 shadow-[0_0_24px_-4px_rgba(16,185,129,0.35)] ring-1 ring-emerald-500/40"
                            : "border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/20"
                        } disabled:opacity-50`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-xs text-white tracking-tight flex items-center gap-1.5">
                            {isViolence ? (
                              <Flame className="h-3.5 w-3.5 text-rose-400" />
                            ) : (
                              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                            )}
                            {sample.name}
                          </span>
                          <span
                            className={`text-[10px] font-semibold px-2.5 py-0.5 rounded-full border ${
                              isViolence
                                ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
                                : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                            }`}
                          >
                            {sample.category}
                          </span>
                        </div>
                        <p className="text-[11px] text-zinc-400 leading-snug line-clamp-2 font-normal">
                          {sample.description}
                        </p>
                      </motion.button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Video Player Preview Glass Card */}
            <div className="rounded-3xl border border-white/[0.08] bg-white/[0.025] backdrop-blur-2xl p-6 shadow-[0_8px_32px_0_rgba(0,0,0,0.36)] space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Play className="h-4 w-4 text-zinc-300" />
                  <span className="text-xs font-semibold text-zinc-200">Video Canvas</span>
                </div>
                <span className="text-[11px] font-mono text-zinc-400">
                  {uploadedFile ? uploadedFile.name : selectedSample?.name}
                </span>
              </div>

              {/* Player Container */}
              <div className="relative aspect-video w-full rounded-2xl overflow-hidden bg-black/90 border border-white/10 shadow-2xl">
                <video
                  ref={videoRef}
                  id="preview-video-player"
                  src={previewVideoUrl}
                  controls
                  playsInline
                  className="w-full h-full object-contain"
                />

                {/* Subtle active scanning overlay line if analyzing */}
                {isAnalyzing && (
                  <motion.div
                    className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-indigo-400 to-transparent shadow-[0_0_15px_#818cf8]"
                    initial={{ top: "0%" }}
                    animate={{ top: "100%" }}
                    transition={{
                      repeat: Infinity,
                      duration: 1.8,
                      ease: "linear",
                    }}
                  />
                )}
              </div>

              {/* Primary Analyze Action Button */}
              <motion.button
                id="start-analysis-btn"
                onClick={handleRunAnalysis}
                disabled={isAnalyzing}
                whileHover={!isAnalyzing ? { scale: 1.01 } : {}}
                whileTap={!isAnalyzing ? { scale: 0.99 } : {}}
                className={`w-full py-4 px-6 rounded-2xl font-bold text-sm tracking-wide transition shadow-xl flex items-center justify-center gap-3 ${
                  isAnalyzing
                    ? "bg-zinc-800/80 text-zinc-400 cursor-not-allowed border border-white/10"
                    : "bg-white text-zinc-950 hover:bg-zinc-100 shadow-[0_4px_24px_rgba(255,255,255,0.15)] border border-white active:scale-[0.99]"
                }`}
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="h-4 w-4 text-indigo-400 animate-spin" />
                    <span>Processing Neural Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 text-zinc-900 fill-zinc-900" />
                    <span>Run Kinematic ST-GCN Analysis</span>
                  </>
                )}
              </motion.button>

              {errorMsg && (
                <div
                  id="analysis-error-banner"
                  className="rounded-2xl border border-rose-500/30 bg-rose-950/20 p-4 text-xs text-rose-300 flex items-center gap-2.5 backdrop-blur-md"
                >
                  <AlertTriangle className="h-4 w-4 text-rose-400 flex-shrink-0" />
                  <span>{errorMsg}</span>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Detailed Kinematic Analysis Report (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            <div className="rounded-3xl border border-white/[0.08] bg-white/[0.025] backdrop-blur-2xl p-6 shadow-[0_8px_32px_0_rgba(0,0,0,0.36)] space-y-5">
              <div className="flex items-center justify-between pb-3 border-b border-white/[0.08]">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-zinc-300" />
                  <h3 className="text-sm font-semibold tracking-tight text-white">
                    Telemetry & Verdict
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-zinc-400">
                  {analysisReport ? `${analysisReport.processing_time_sec}s runtime` : "Standby"}
                </span>
              </div>

              {/* State 1: Active Waiting / Progress State (UX Psychology) */}
              <AnimatePresence mode="wait">
                {isAnalyzing && (
                  <motion.div
                    key="analyzing-state"
                    id="analyzing-state"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="py-12 px-4 text-center space-y-5"
                  >
                    {/* Pulsing Orbital Radar Animation */}
                    <div className="relative mx-auto h-20 w-20 flex items-center justify-center">
                      <div className="absolute inset-0 rounded-full border border-indigo-500/20 animate-ping" />
                      <div className="absolute inset-1 rounded-full border border-indigo-400/30 animate-pulse" />
                      <motion.div
                        className="h-20 w-20 rounded-full border-2 border-indigo-400 border-t-transparent"
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
                      />
                      <Radar className="h-8 w-8 text-indigo-400 absolute" />
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-bold text-white tracking-tight">
                        Active Waiting: Neural Evaluation
                      </h4>
                      <AnimatePresence mode="wait">
                        <motion.p
                          key={scanStepIndex}
                          initial={{ opacity: 0, y: 4 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -4 }}
                          className="text-xs font-mono text-indigo-300 h-5"
                        >
                          {SCAN_STEPS[scanStepIndex]}
                        </motion.p>
                      </AnimatePresence>
                      <p className="text-[11px] text-zinc-400 max-w-xs mx-auto leading-relaxed pt-1">
                        Normalizing COCO-17 keypoint coordinates, computing multi-frame graph convolutions, and validating strike proximity boundaries.
                      </p>
                    </div>
                  </motion.div>
                )}

                {/* State 2: Idle (No Report Yet) */}
                {!isAnalyzing && !analysisReport && (
                  <motion.div
                    key="idle-state"
                    id="idle-report-state"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="py-14 px-4 text-center space-y-3.5"
                  >
                    <div className="h-14 w-14 rounded-2xl bg-white/[0.04] border border-white/10 flex items-center justify-center mx-auto text-zinc-400 shadow-inner">
                      <Sparkles className="h-6 w-6 text-zinc-400" />
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-zinc-200 tracking-tight">
                        Awaiting Ingestion
                      </h4>
                      <p className="text-xs text-zinc-400 max-w-xs mx-auto leading-relaxed mt-1">
                        Select an altercation or safe clip above, or upload custom surveillance footage to inspect kinematic graph predictions.
                      </p>
                    </div>
                  </motion.div>
                )}

                {/* State 3: Analysis Report Result with Semantic Urgency */}
                {!isAnalyzing && analysisReport && (
                  <motion.div
                    key="report-state"
                    id="analysis-report-view"
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ type: "spring", stiffness: 300, damping: 25 }}
                    className="space-y-6"
                  >
                    {/* The Semantic Urgency Verdict Card */}
                    {analysisReport.overall_verdict === "Violence" ? (
                      /* High-Contrast Apple System Red */
                      <div
                        id="verdict-banner"
                        className="rounded-2xl border border-rose-500/50 bg-gradient-to-b from-rose-950/40 via-rose-950/20 to-black/60 p-5 text-center shadow-[0_0_35px_rgba(244,63,94,0.3)] backdrop-blur-xl space-y-3"
                      >
                        <div className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full bg-rose-500/20 border border-rose-500/30 text-[11px] font-bold uppercase tracking-wider text-rose-300">
                          <AlertTriangle className="h-3.5 w-3.5 text-rose-400" />
                          Hostile Kinematics Confirmed
                        </div>

                        <div className="text-2xl font-black tracking-tight text-rose-400">
                          VIOLENCE DETECTED
                        </div>

                        {/* Confidence Meter */}
                        <div className="space-y-1.5 pt-2 border-t border-rose-500/20">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-zinc-400">Confidence Rating:</span>
                            <span className="font-bold font-mono text-rose-300">
                              {(analysisReport.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-2 w-full rounded-full bg-black/60 overflow-hidden border border-white/10">
                            <motion.div
                              className="h-full rounded-full bg-gradient-to-r from-rose-500 via-red-500 to-amber-500 shadow-[0_0_12px_rgba(244,63,94,0.8)]"
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.min(100, Math.max(0, analysisReport.confidence * 100))}%` }}
                              transition={{ duration: 0.8, ease: "easeOut" }}
                            />
                          </div>
                        </div>
                      </div>
                    ) : (
                      /* Muted, Serene Emerald Safe State */
                      <div
                        id="verdict-banner"
                        className="rounded-2xl border border-emerald-500/30 bg-gradient-to-b from-emerald-950/30 via-emerald-950/10 to-black/60 p-5 text-center shadow-[0_0_25px_rgba(16,185,129,0.2)] backdrop-blur-xl space-y-3"
                      >
                        <div className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/30 text-[11px] font-bold uppercase tracking-wider text-emerald-300">
                          <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                          Normal Kinematics Verified
                        </div>

                        <div className="text-2xl font-black tracking-tight text-emerald-400">
                          NORMAL / SECURE
                        </div>

                        {/* Confidence Meter */}
                        <div className="space-y-1.5 pt-2 border-t border-emerald-500/20">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-zinc-400">Confidence Rating:</span>
                            <span className="font-bold font-mono text-emerald-300">
                              {((1 - analysisReport.confidence) * 100).toFixed(1)}% Safe
                            </span>
                          </div>
                          <div className="h-2 w-full rounded-full bg-black/60 overflow-hidden border border-white/10">
                            <motion.div
                              className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 shadow-[0_0_12px_rgba(16,185,129,0.7)]"
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.min(100, Math.max(0, (1 - analysisReport.confidence) * 100))}%` }}
                              transition={{ duration: 0.8, ease: "easeOut" }}
                            />
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Telemetry Metrics Grid */}
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-3.5 space-y-1">
                        <span className="text-zinc-400 text-[11px] flex items-center gap-1">
                          <Clock className="h-3 w-3 text-zinc-400" />
                          Processing Time
                        </span>
                        <p className="font-mono font-bold text-white text-sm">
                          {analysisReport.processing_time_sec}s
                        </p>
                      </div>
                      <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-3.5 space-y-1">
                        <span className="text-zinc-400 text-[11px] flex items-center gap-1">
                          <Layers className="h-3 w-3 text-zinc-400" />
                          Frames Evaluated
                        </span>
                        <p className="font-mono font-bold text-white text-sm">
                          {analysisReport.frame_count ?? "—"} frames
                        </p>
                      </div>
                      <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-3.5 space-y-1">
                        <span className="text-zinc-400 text-[11px] flex items-center gap-1">
                          <Film className="h-3 w-3 text-zinc-400" />
                          Duration / FPS
                        </span>
                        <p className="font-mono font-bold text-white text-sm">
                          {analysisReport.video_duration_sec ?? "—"}s ({analysisReport.video_fps ?? 30} fps)
                        </p>
                      </div>
                      <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-3.5 space-y-1">
                        <span className="text-zinc-400 text-[11px] flex items-center gap-1">
                          <Activity className="h-3 w-3 text-zinc-400" />
                          Peak Probability
                        </span>
                        <p className="font-mono font-bold text-white text-sm">
                          {analysisReport.peak_probability
                            ? `${(analysisReport.peak_probability * 100).toFixed(1)}%`
                            : "—"}
                        </p>
                      </div>
                    </div>

                    {/* Incident Timestamps Section */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-300 flex items-center gap-1.5">
                          <Clock className="h-3.5 w-3.5 text-zinc-400" />
                          Incident Triggers
                        </h4>
                        <span className="text-[11px] font-mono text-zinc-400">
                          {analysisReport.incident_timestamps.length} Detected
                        </span>
                      </div>

                      {analysisReport.incident_timestamps.length === 0 ? (
                        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-4 text-center text-xs text-zinc-400">
                          No violent kinetic spikes breached the Relational Strike Gate.
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <p className="text-[11px] text-zinc-400">
                            Click any incident pill to jump directly to the altercation moment:
                          </p>
                          <div
                            id="incident-timestamps-list"
                            className="flex flex-wrap gap-2 max-h-36 overflow-y-auto pr-1"
                          >
                            {analysisReport.incident_timestamps.map((ts, idx) => (
                              <motion.button
                                key={`ts-${idx}-${ts}`}
                                onClick={() => seekToTimestamp(ts)}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                className="px-3 py-1.5 rounded-full text-xs font-mono font-bold bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 hover:border-rose-300 transition flex items-center gap-1.5 shadow-sm"
                                title={`Jump to ${ts.toFixed(2)}s`}
                              >
                                <Flame className="h-3 w-3 text-rose-400" />
                                <span>{ts.toFixed(2)}s</span>
                              </motion.button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Timeline Breakdown */}
                    {analysisReport.timeline && analysisReport.timeline.length > 0 && (
                      <div className="space-y-2.5 pt-2 border-t border-white/[0.08]">
                        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
                          Sampled Sequence Telemetry
                        </span>
                        <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1 font-mono text-[11px]">
                          {analysisReport.timeline.map((point, idx) => (
                            <div
                              key={`tl-${idx}`}
                              onClick={() => seekToTimestamp(point.timestamp)}
                              className={`flex items-center justify-between p-2.5 rounded-xl border cursor-pointer transition ${
                                point.is_fighting
                                  ? "bg-rose-950/20 border-rose-500/30 text-rose-300 hover:bg-rose-950/30"
                                  : "bg-white/[0.02] border-white/[0.06] text-zinc-400 hover:bg-white/[0.05]"
                              }`}
                            >
                              <span className="font-bold">{(point.timestamp ?? 0).toFixed(2)}s</span>
                              <span>P: {((point.probability ?? 0) * 100).toFixed(0)}%</span>
                              {point.min_strike_dist !== undefined && point.min_strike_dist < 9000 && (
                                <span>Dist: {point.min_strike_dist.toFixed(0)}px</span>
                              )}
                              <span
                                className={`px-2 py-0.5 rounded-full text-[10px] font-sans font-bold ${
                                  point.is_fighting
                                    ? "bg-rose-500 text-white shadow-[0_0_8px_rgba(244,63,94,0.6)]"
                                    : "bg-white/10 text-zinc-300"
                                }`}
                              >
                                {point.is_fighting ? "ALERT" : "SECURE"}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
