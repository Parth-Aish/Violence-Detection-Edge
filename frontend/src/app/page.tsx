"use client";

import { useCallback, useEffect, useRef, useState } from "react";

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
  person_count: number;
  bio_score: number;
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
  timeline?: TimelinePoint[];
}

interface SampleItem {
  id: string;
  name: string;
  category: "Safe" | "Violence";
  description: string;
  videoUrl: string;
}

const PRESET_SAMPLES: SampleItem[] = [
  {
    id: "NV_1.mp4",
    name: "Safe Activity (NV_1)",
    category: "Safe",
    description: "Multi-person physical motion without hostile kinetic acceleration.",
    videoUrl: "/samples/safe_sample.mp4",
  },
  {
    id: "V_1.mp4",
    name: "Physical Altercation (V_1)",
    category: "Violence",
    description: "Rapid strike velocity bursts, close proximity, and interaction turbulence.",
    videoUrl: "/samples/violence_sample.mp4",
  },
];

export default function VideoAnalysisDashboard() {
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(PRESET_SAMPLES[0]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string>(PRESET_SAMPLES[0].videoUrl);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisReport, setAnalysisReport] = useState<AnalysisReport | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [copiedSnippet, setCopiedSnippet] = useState<boolean>(false);

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

  // Update preview URL when sample changes
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

  // Drag and drop handler
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
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

      // If incidents found, jump video to first incident
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
python run_live.py`;

  const copySnippet = () => {
    navigator.clipboard.writeText(edgeSnippet);
    setCopiedSnippet(true);
    setTimeout(() => setCopiedSnippet(false), 2500);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-[#fafafa] flex flex-col font-sans selection:bg-indigo-500/30 selection:text-white">
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-white/10 bg-[#09090b]/80 backdrop-blur-xl px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <span className="text-white font-black text-lg">V</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-white">
                  Vision Kinematics
                </h1>
                <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-white/[0.08] text-zinc-300 border border-white/10">
                  Batch Analytics
                </span>
              </div>
              <p className="text-xs text-zinc-400">
                Spatio-Temporal Pose Kinematics & Kinetic Incident Detection
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-black/40 px-3 py-1 text-xs">
              <span
                className={`h-2 w-2 rounded-full ${
                  apiConnected === true
                    ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"
                    : apiConnected === false
                    ? "bg-rose-500"
                    : "bg-amber-400 animate-pulse"
                }`}
              />
              <span className="text-zinc-300">
                {apiConnected === true
                  ? "Cloud API Active"
                  : apiConnected === false
                  ? "API Offline (Local Edge)"
                  : "Connecting..."}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Dashboard Body ───────────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* ── Prominent Local Edge Instructions Banner ─────────────────────── */}
        <section
          id="edge-hardware-banner"
          className="relative overflow-hidden rounded-2xl border border-emerald-500/30 bg-gradient-to-r from-emerald-950/40 via-zinc-900/90 to-cyan-950/30 p-5 shadow-xl shadow-emerald-950/20 backdrop-blur-md"
        >
          <div className="absolute -top-12 -right-12 h-44 w-44 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none" />
          <div className="relative flex flex-col lg:flex-row lg:items-center justify-between gap-5">
            <div className="space-y-1.5 max-w-2xl">
              <div className="inline-flex items-center gap-2 text-xs font-bold tracking-wider uppercase text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
                Edge Hardware Real-Time Mode
              </div>
              <h2 className="text-lg md:text-xl font-bold tracking-tight text-white">
                For 30+ FPS Real-Time Live Camera Detection, run this system locally on Edge hardware.
              </h2>
              <p className="text-xs md:text-sm text-zinc-300 leading-relaxed">
                Unlock native webcam and RTSP surveillance camera inference with full multi-threaded CPU/GPU acceleration,
                zero network latency, and persistent local HUD bounding box overlays.
              </p>
            </div>

            <div className="flex-shrink-0 w-full lg:w-auto">
              <div className="rounded-xl border border-white/10 bg-black/70 p-3 shadow-inner">
                <div className="flex items-center justify-between gap-4 pb-2 mb-2 border-b border-white/10 text-[11px] text-zinc-400">
                  <span className="font-mono text-emerald-300">Terminal (PowerShell / Bash)</span>
                  <button
                    id="copy-edge-command-btn"
                    onClick={copySnippet}
                    className="flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-semibold bg-white/10 hover:bg-white/20 text-white transition"
                  >
                    {copiedSnippet ? (
                      <>
                        <span className="text-emerald-400">✓</span> Copied
                      </>
                    ) : (
                      <>
                        <span>📋</span> Copy Snippet
                      </>
                    )}
                  </button>
                </div>
                <pre className="font-mono text-xs text-zinc-200 select-all overflow-x-auto whitespace-pre leading-relaxed">
                  <code>{edgeSnippet}</code>
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* ── Two-Column Interactive Analysis Console ─────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Input Selection & Video Preview (7 cols) */}
          <div className="lg:col-span-7 space-y-5">
            {/* Quick-Select Sample Clips */}
            <div className="rounded-2xl border border-white/10 bg-zinc-900/60 p-5 backdrop-blur-md space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold tracking-wide text-zinc-200 flex items-center gap-2">
                  <span>📹</span> 1. Select Video Source
                </h3>
                <span className="text-xs text-zinc-400">Sample Clips or Custom MP4</span>
              </div>

              {/* Sample Clips Buttons */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {PRESET_SAMPLES.map((sample) => {
                  const isSelected = selectedSample?.id === sample.id && !uploadedFile;
                  return (
                    <button
                      key={sample.id}
                      id={`sample-btn-${sample.id.replace(/\W/g, "-")}`}
                      onClick={() => handleSelectSample(sample)}
                      disabled={isAnalyzing}
                      className={`relative text-left p-3.5 rounded-xl border transition duration-200 flex flex-col justify-between ${
                        isSelected
                          ? sample.category === "Safe"
                            ? "border-emerald-500/50 bg-emerald-950/20 shadow-[0_0_15px_-3px_rgba(16,185,129,0.3)] ring-1 ring-emerald-500/30"
                            : "border-rose-500/50 bg-rose-950/20 shadow-[0_0_15px_-3px_rgba(244,63,94,0.3)] ring-1 ring-rose-500/30"
                          : "border-white/10 bg-zinc-800/40 hover:bg-zinc-800/70 hover:border-white/20"
                      } disabled:opacity-50`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-xs text-white">
                          {sample.name}
                        </span>
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                            sample.category === "Safe"
                              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                              : "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                          }`}
                        >
                          {sample.category}
                        </span>
                      </div>
                      <p className="text-[11px] text-zinc-400 leading-snug line-clamp-2">
                        {sample.description}
                      </p>
                    </button>
                  );
                })}
              </div>

              {/* Upload Dropzone */}
              <div
                id="video-dropzone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition ${
                  uploadedFile
                    ? "border-indigo-500/60 bg-indigo-950/20"
                    : "border-white/15 bg-black/20 hover:border-white/30 hover:bg-black/30"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <div className="flex flex-col items-center justify-center gap-1.5">
                  <span className="text-2xl">{uploadedFile ? "📁" : "⬆️"}</span>
                  {uploadedFile ? (
                    <div>
                      <p className="text-xs font-semibold text-indigo-300">
                        {uploadedFile.name}
                      </p>
                      <p className="text-[10px] text-zinc-400">
                        {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for analysis
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-xs font-medium text-zinc-300">
                        Drag & Drop or Click to Upload Custom Video
                      </p>
                      <p className="text-[10px] text-zinc-400">
                        Supports MP4, AVI, MOV up to 150 MB
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Video Player Preview */}
            <div className="rounded-2xl border border-white/10 bg-zinc-900/60 p-5 backdrop-blur-md space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-300 flex items-center gap-1.5">
                  <span>📺</span> Video Preview
                </span>
                <span className="text-[11px] text-zinc-400">
                  {uploadedFile ? uploadedFile.name : selectedSample?.name}
                </span>
              </div>

              <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-black border border-white/10 shadow-lg">
                <video
                  ref={videoRef}
                  id="preview-video-player"
                  src={previewVideoUrl}
                  controls
                  playsInline
                  className="w-full h-full object-contain"
                />
              </div>

              {/* Start Analysis Button */}
              <button
                id="start-analysis-btn"
                onClick={handleRunAnalysis}
                disabled={isAnalyzing}
                className={`w-full py-3.5 px-4 rounded-xl font-bold text-sm tracking-wide transition shadow-lg flex items-center justify-center gap-2.5 ${
                  isAnalyzing
                    ? "bg-zinc-800 text-zinc-400 cursor-not-allowed border border-white/10"
                    : "bg-gradient-to-r from-indigo-600 via-indigo-500 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-indigo-500/25 border border-indigo-400/30 active:scale-[0.99]"
                }`}
              >
                {isAnalyzing ? (
                  <>
                    <svg
                      className="animate-spin h-4 w-4 text-indigo-400"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8v8H4z"
                      />
                    </svg>
                    <span>Analyzing video kinematics...</span>
                  </>
                ) : (
                  <>
                    <span>⚡</span>
                    <span>Analyze Video Kinematics</span>
                  </>
                )}
              </button>

              {errorMsg && (
                <div
                  id="analysis-error-banner"
                  className="rounded-xl border border-rose-500/30 bg-rose-950/20 p-3 text-xs text-rose-300 flex items-center gap-2"
                >
                  <span>⚠️</span>
                  <span>{errorMsg}</span>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Detailed Kinematic Analysis Report (5 cols) */}
          <div className="lg:col-span-5 space-y-5">
            <div className="rounded-2xl border border-white/10 bg-zinc-900/60 p-5 backdrop-blur-md space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-white/10">
                <h3 className="text-sm font-semibold tracking-wide text-zinc-200 flex items-center gap-2">
                  <span>📊</span> 2. Analysis Report
                </h3>
                <span className="text-[11px] font-mono text-zinc-400">
                  {analysisReport ? `Time: ${analysisReport.processing_time_sec}s` : "Pending Run"}
                </span>
              </div>

              {/* State 1: Loading Progress */}
              {isAnalyzing && (
                <div id="analyzing-state" className="py-12 px-4 text-center space-y-4">
                  <div className="relative mx-auto h-16 w-16">
                    <div className="absolute inset-0 rounded-full border-4 border-indigo-500/20 animate-ping" />
                    <div className="h-16 w-16 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white tracking-tight">
                      Analyzing Video Kinematics
                    </h4>
                    <p className="text-xs text-zinc-400 mt-1 max-w-xs mx-auto">
                      Processing frame-by-frame sequential pose estimation, 1st-order velocities,
                      2nd-order accelerations, and interaction turbulence...
                    </p>
                  </div>
                </div>
              )}

              {/* State 2: Idle (No Report Yet) */}
              {!isAnalyzing && !analysisReport && (
                <div id="idle-report-state" className="py-12 px-4 text-center space-y-3">
                  <div className="h-12 w-12 rounded-2xl bg-zinc-800/80 border border-white/10 flex items-center justify-center text-xl mx-auto text-zinc-400">
                    🔬
                  </div>
                  <h4 className="text-sm font-semibold text-zinc-300">
                    Ready for Kinematic Analysis
                  </h4>
                  <p className="text-xs text-zinc-400 max-w-xs mx-auto leading-relaxed">
                    Select a sample clip or upload an MP4 video, then click &ldquo;Analyze Video Kinematics&rdquo; to generate a full spatio-temporal report.
                  </p>
                </div>
              )}

              {/* State 3: Analysis Report Result */}
              {!isAnalyzing && analysisReport && (
                <div id="analysis-report-view" className="space-y-5">
                  {/* Verdict Card */}
                  <div
                    id="verdict-banner"
                    className={`rounded-xl border p-4.5 text-center shadow-lg transition ${
                      analysisReport.overall_verdict === "Violence"
                        ? "border-rose-500/50 bg-gradient-to-b from-rose-950/40 to-black/60 shadow-rose-950/30"
                        : "border-emerald-500/50 bg-gradient-to-b from-emerald-950/40 to-black/60 shadow-emerald-950/30"
                    }`}
                  >
                    <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-400">
                      Overall Verdict
                    </span>
                    <div
                      className={`text-2xl font-black tracking-tight mt-1 mb-2 ${
                        analysisReport.overall_verdict === "Violence"
                          ? "text-rose-400"
                          : "text-emerald-400"
                      }`}
                    >
                      {analysisReport.overall_verdict === "Violence"
                        ? "VIOLENCE DETECTED"
                        : "NORMAL / SAFE"}
                    </div>

                    {/* Confidence Meter */}
                    <div className="space-y-1.5 pt-2 border-t border-white/10">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-zinc-400">Confidence Score:</span>
                        <span className="font-bold font-mono text-white">
                          {(analysisReport.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-black/60 overflow-hidden border border-white/10">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            analysisReport.overall_verdict === "Violence"
                              ? "bg-gradient-to-r from-rose-500 to-amber-500"
                              : "bg-gradient-to-r from-emerald-500 to-teal-400"
                          }`}
                          style={{
                            width: `${Math.min(100, Math.max(0, analysisReport.confidence * 100))}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Telemetry Metrics Grid */}
                  <div className="grid grid-cols-2 gap-2.5 text-xs">
                    <div className="rounded-xl border border-white/10 bg-black/30 p-3 space-y-1">
                      <span className="text-zinc-400 text-[11px]">Processing Time</span>
                      <p className="font-mono font-bold text-white text-sm">
                        {analysisReport.processing_time_sec}s
                      </p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-black/30 p-3 space-y-1">
                      <span className="text-zinc-400 text-[11px]">Frames Analyzed</span>
                      <p className="font-mono font-bold text-white text-sm">
                        {analysisReport.frame_count ?? "—"} frames
                      </p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-black/30 p-3 space-y-1">
                      <span className="text-zinc-400 text-[11px]">Video Duration</span>
                      <p className="font-mono font-bold text-white text-sm">
                        {analysisReport.video_duration_sec ?? "—"}s
                      </p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-black/30 p-3 space-y-1">
                      <span className="text-zinc-400 text-[11px]">Peak Probability</span>
                      <p className="font-mono font-bold text-white text-sm">
                        {analysisReport.peak_probability
                          ? `${(analysisReport.peak_probability * 100).toFixed(1)}%`
                          : "—"}
                      </p>
                    </div>
                  </div>

                  {/* Incident Timestamps Section */}
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-300 flex items-center gap-1.5">
                        <span>⏱️</span> Incident Timestamps
                      </h4>
                      <span className="text-[11px] font-mono text-zinc-400">
                        {analysisReport.incident_timestamps.length} Trigger(s)
                      </span>
                    </div>

                    {analysisReport.incident_timestamps.length === 0 ? (
                      <div className="rounded-xl border border-white/10 bg-black/20 p-3 text-center text-xs text-zinc-400">
                        No violent kinematic triggers detected. Entire clip is verified secure.
                      </div>
                    ) : (
                      <div className="space-y-1.5">
                        <p className="text-[11px] text-zinc-400">
                          Click any timestamp chip to seek video directly to that incident:
                        </p>
                        <div
                          id="incident-timestamps-list"
                          className="flex flex-wrap gap-2 max-h-36 overflow-y-auto pr-1"
                        >
                          {analysisReport.incident_timestamps.map((ts, idx) => (
                            <button
                              key={`ts-${idx}-${ts}`}
                              onClick={() => seekToTimestamp(ts)}
                              className="px-2.5 py-1 rounded-lg text-xs font-mono font-bold bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 hover:border-rose-400 transition flex items-center gap-1 active:scale-95"
                              title={`Jump to ${ts.toFixed(2)}s`}
                            >
                              <span>🚨</span>
                              <span>{ts.toFixed(2)}s</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Keyframe Timeline Breakdown */}
                  {analysisReport.timeline && analysisReport.timeline.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-white/10">
                      <span className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
                        Kinematic Timeline Events
                      </span>
                      <div className="max-h-40 overflow-y-auto space-y-1.5 pr-1 font-mono text-[11px]">
                        {analysisReport.timeline.map((point, idx) => (
                          <div
                            key={`tl-${idx}`}
                            onClick={() => seekToTimestamp(point.timestamp)}
                            className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition ${
                              point.is_fighting
                                ? "bg-rose-950/20 border-rose-500/30 text-rose-300 hover:bg-rose-950/30"
                                : "bg-black/30 border-white/5 text-zinc-400 hover:bg-black/50"
                            }`}
                          >
                            <span className="font-bold">{(point.timestamp ?? 0).toFixed(2)}s</span>
                            <span>Prob: {((point.probability ?? 0) * 100).toFixed(0)}%</span>
                            <span>Bio: {(point.bio_score ?? point.probability ?? 0).toFixed(2)}</span>
                            <span
                              className={`px-1.5 py-0.5 rounded text-[10px] font-sans font-bold ${
                                point.is_fighting
                                  ? "bg-rose-500 text-white"
                                  : "bg-white/10 text-zinc-300"
                              }`}
                            >
                              {point.is_fighting ? "ALERT" : "SAFE"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
