"use client";

import { useEffect, useRef, useState, useCallback } from "react";

/* ════════════════════════════════════════════════════════════
   Types
   ════════════════════════════════════════════════════════════ */

interface Incident {
  id: string;
  timestamp: number;
  timestamp_iso: string;
  person_id: number;
  severity: number;
  severity_label: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  max_velocity: number;
  mean_velocity: number;
  confidence: number;
  bbox_x1: number;
  bbox_y1: number;
  bbox_x2: number;
  bbox_y2: number;
}

interface Stats {
  LOW: number;
  MEDIUM: number;
  HIGH: number;
  CRITICAL: number;
  total: number;
}

type ConnectionStatus = "connecting" | "connected" | "disconnected";

/* ════════════════════════════════════════════════════════════
   Constants
   ════════════════════════════════════════════════════════════ */

function getWsUrl(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window !== "undefined") {
    // If browsing through Next.js dev server (port 3000), connect to backend port 7860
    if (window.location.port === "3000") {
      return `ws://${window.location.hostname}:7860/ws/stream`;
    }
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws/stream`;
  }
  return "ws://localhost:7860/ws/stream";
}

function getApiUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined") {
    // If browsing through Next.js dev server (port 3000), connect to backend port 7860
    if (window.location.port === "3000") {
      return `${window.location.protocol}//${window.location.hostname}:7860`;
    }
    return window.location.origin;
  }
  return "http://localhost:7860";
}

const MAX_INCIDENTS = 100;
const RECONNECT_DELAY = 3000;

/* ════════════════════════════════════════════════════════════
   Helpers
   ════════════════════════════════════════════════════════════ */

function severityBadgeClass(label: string): string {
  switch (label) {
    case "CRITICAL": return "badge badge-critical";
    case "HIGH":     return "badge badge-high";
    case "MEDIUM":   return "badge badge-medium";
    case "LOW":      return "badge badge-low";
    default:         return "badge";
  }
}

function severityDotClass(label: string): string {
  switch (label) {
    case "CRITICAL": return "bg-[var(--severity-critical)]";
    case "HIGH":     return "bg-[var(--severity-high)]";
    case "MEDIUM":   return "bg-[var(--severity-medium)]";
    case "LOW":      return "bg-[var(--severity-low)]";
    default:         return "bg-zinc-500";
  }
}

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000 - ts);
  if (diff < 5) return "just now";
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

/* ════════════════════════════════════════════════════════════
   Main Page Component
   ════════════════════════════════════════════════════════════ */

export default function DashboardPage() {
  /* ── State ── */
  const [frameData, setFrameData] = useState<string | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [stats, setStats] = useState<Stats>({ LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0, total: 0 });
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const [isViolence, setIsViolence] = useState(false);
  const [sourceMode, setSourceMode] = useState<"webcam" | "upload">("webcam");
  const [sourceBusy, setSourceBusy] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [uploadedFilename, setUploadedFilename] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const frameCountRef = useRef(0);
  const fpsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectRef = useRef<NodeJS.Timeout | null>(null);
  const incidentScrollRef = useRef<HTMLDivElement>(null);

  /* ── Fetch initial incidents via REST ── */
  const fetchIncidents = useCallback(async () => {
    try {
      const res = await fetch(`${getApiUrl()}/api/incidents?limit=50&min_severity=1`);
      if (res.ok) {
        const data = await res.json();
        setIncidents(data.incidents || []);
      }
    } catch {
      /* Server not reachable yet — fine */
    }
  }, []);

  /* ── Fetch stats ── */
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${getApiUrl()}/api/stats?hours=24`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch {
      /* noop */
    }
  }, []);

  /* ── WebSocket connection ── */
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      console.log("[WS] Connected");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "frame") {
          setFrameData(msg.data);
          setDetectionCount(msg.detections || 0);
          setIsViolence(msg.is_violence === true);
          frameCountRef.current++;
        }

        if (msg.type === "incident") {
          const incident = msg.data as Incident;
          setIncidents((prev) => {
            const next = [incident, ...prev];
            return next.slice(0, MAX_INCIDENTS);
          });
          /* Auto-refresh stats on new incident */
          fetchStats();
        }
      } catch (e) {
        console.error("[WS] Parse error:", e);
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      console.log("[WS] Disconnected — reconnecting...");
      reconnectRef.current = setTimeout(connectWs, RECONNECT_DELAY);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [fetchStats]);

  /* ── Lifecycle ── */
  useEffect(() => {
    fetchIncidents();
    fetchStats();
    connectWs();

    /* FPS counter */
    fpsIntervalRef.current = setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
    }, 1000);

    /* Refresh stats every 30s */
    const statsInterval = setInterval(fetchStats, 30000);

    return () => {
      wsRef.current?.close();
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (fpsIntervalRef.current) clearInterval(fpsIntervalRef.current);
      clearInterval(statsInterval);
    };
  }, [connectWs, fetchIncidents, fetchStats]);

  /* ── Derive status ── */
  const statusDotClass =
    status === "connected" ? "live" :
    status === "connecting" ? "live" : "offline";

  const statusLabel =
    status === "connected" ? "LIVE" :
    status === "connecting" ? "CONNECTING" : "OFFLINE";

  const hasAlerts = incidents.some(
    (i) => i.severity_label === "CRITICAL" || i.severity_label === "HIGH"
  );

  const useWebcam = async () => {
    setSourceBusy(true);
    setSourceError(null);
    try {
      const response = await fetch(`${getApiUrl()}/source/webcam`, { method: "POST" });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Could not switch to webcam.");
      }
      setSourceMode("webcam");
      setUploadedFilename(null);
      setIsViolence(false);
      setFrameData(null);
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : "Could not switch to webcam.");
    } finally {
      setSourceBusy(false);
    }
  };

  const uploadVideo = async (file?: File) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".mp4")) {
      setSourceError("Choose an MP4 video file.");
      return;
    }

    setSourceBusy(true);
    setSourceError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${getApiUrl()}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Could not upload video.");
      }
      setSourceMode("upload");
      setUploadedFilename(file.name);
      setIsViolence(false);
      setFrameData(null);
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : "Could not upload video.");
    } finally {
      setSourceBusy(false);
      if (uploadInputRef.current) uploadInputRef.current.value = "";
    }
  };

  /* ════════════════════════════════════════════════════════
     Render
     ════════════════════════════════════════════════════════ */
  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden">

      {/* ────── Top Bar ────── */}
      <header className="glass-panel-strong flex items-center justify-between px-6 py-3 mx-4 mt-4 mb-3 animate-fade-in">
        <div className="flex items-center gap-3">
          {/* Logo / Shield icon */}
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2l7 4v6c0 5.25-3.5 9.74-7 11-3.5-1.26-7-5.75-7-11V6l7-4z"/>
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight">Violence Detection System</h1>
            <p className="text-[10px] text-zinc-500 tracking-wide uppercase">Edge-Optimized Surveillance AI</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          {/* FPS */}
          <div className="text-right">
            <div className="text-xs font-mono text-zinc-400">{fps} FPS</div>
            <div className="text-[10px] text-zinc-600">inference</div>
          </div>

          {/* Detections */}
          <div className="text-right">
            <div className="text-xs font-mono text-zinc-400">{detectionCount}</div>
            <div className="text-[10px] text-zinc-600">detections</div>
          </div>

          {/* Status */}
          <div className="flex items-center gap-2">
            <div className={`status-dot ${statusDotClass}`} />
            <span className="text-[11px] font-medium tracking-wider text-zinc-400">
              {statusLabel}
            </span>
          </div>
        </div>
      </header>

      {/* ────── Main Content ────── */}
      <main className="flex-1 flex gap-3 px-4 pb-4 min-h-0">

        {/* ════ Left: Video Feed ════ */}
        <section className="flex-1 flex flex-col gap-3 min-w-0">
          {/* Video */}
          <div className="video-container flex-1 relative animate-fade-in">
            {frameData ? (
              <img
                src={`data:image/jpeg;base64,${frameData}`}
                alt="Live surveillance feed"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-zinc-800/50 flex items-center justify-center">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-zinc-600">
                      <rect x="2" y="4" width="20" height="16" rx="3"/>
                      <circle cx="12" cy="12" r="3"/>
                      <path d="M17 4v-1a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1"/>
                    </svg>
                  </div>
                  <p className="text-sm text-zinc-500 font-medium">
                    {status === "connecting" ? "Connecting to stream..." : "No video feed"}
                  </p>
                  <p className="text-xs text-zinc-600 mt-1">
                    {status === "disconnected"
                      ? "Start the backend server to begin streaming"
                      : "Waiting for frames..."}
                  </p>
                </div>
              </div>
            )}

            {/* Video overlay — top bar */}
            {frameData && (
              <div className="video-overlay">
                <div className="flex items-center gap-2">
                  <div className={`status-dot ${hasAlerts ? "alert" : "live"}`} />
                  <span className="text-[11px] font-medium text-white/80 uppercase tracking-wider">
                    {hasAlerts ? "Alert Active" : "Monitoring"}
                  </span>
                </div>
                <span className="text-[11px] font-mono text-white/50">
                  {new Date().toLocaleTimeString("en-US", { hour12: false })}
                </span>
              </div>
            )}

            {frameData && (
              <div
                role={isViolence ? "alert" : "status"}
                aria-live={isViolence ? "assertive" : "off"}
                className={`absolute left-4 top-16 z-10 rounded-full border px-4 py-2 text-[11px] font-bold tracking-[0.12em] shadow-lg backdrop-blur-md ${
                  isViolence
                    ? "animate-pulse border-red-300 bg-red-600 text-white shadow-red-950/60"
                    : "border-emerald-300/40 bg-emerald-950/80 text-emerald-300 shadow-emerald-950/40"
                }`}
              >
                {isViolence ? "🚨 VIOLENCE DETECTED" : "SAFE: NO VIOLENCE"}
              </div>
            )}
          </div>

          {/* ════ Input Controls ════ */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 shadow-xl shadow-black/20 backdrop-blur-2xl">
            <div className="flex min-w-0 flex-col">
              <span className="text-xs font-semibold tracking-tight text-zinc-100">Video source</span>
              <span className="max-w-full truncate text-[11px] text-zinc-500">
                {sourceMode === "upload" && uploadedFilename ? uploadedFilename : "Choose a live camera or an MP4 file"}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={useWebcam}
                disabled={sourceBusy}
                aria-pressed={sourceMode === "webcam"}
                className={`rounded-xl border px-4 py-2 text-xs font-semibold transition disabled:cursor-wait disabled:opacity-50 ${
                  sourceMode === "webcam"
                    ? "border-white/20 bg-white text-zinc-950 shadow-md shadow-white/10"
                    : "border-white/10 bg-white/[0.06] text-zinc-300 hover:bg-white/10"
                }`}
              >
                {sourceBusy ? "Please wait..." : sourceMode === "webcam" ? "Using Webcam" : "Use Webcam"}
              </button>
              <input
                ref={uploadInputRef}
                type="file"
                accept="video/mp4,.mp4"
                className="hidden"
                onChange={(event) => uploadVideo(event.target.files?.[0])}
              />
              <button
                type="button"
                onClick={() => uploadInputRef.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  uploadVideo(event.dataTransfer.files[0]);
                }}
                disabled={sourceBusy}
                className="rounded-xl border border-indigo-300/20 bg-indigo-500/15 px-4 py-2 text-xs font-semibold text-indigo-200 transition hover:border-indigo-200/40 hover:bg-indigo-500/25 disabled:cursor-wait disabled:opacity-50"
              >
                {sourceBusy ? "Working..." : "Upload Video File"}
              </button>
            </div>
            {sourceError && (
              <p role="alert" className="w-full text-[11px] text-red-300">{sourceError}</p>
            )}
          </div>

          {/* ════ Stats Row ════ */}
          <div className="grid grid-cols-4 gap-2 animate-fade-in">
            <div className="stat-card">
              <div className="stat-value text-[var(--severity-critical)]">{stats.CRITICAL}</div>
              <div className="stat-label">Critical</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-[var(--severity-high)]">{stats.HIGH}</div>
              <div className="stat-label">High</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-[var(--severity-medium)]">{stats.MEDIUM}</div>
              <div className="stat-label">Medium</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-[var(--severity-low)]">{stats.LOW}</div>
              <div className="stat-label">Low</div>
            </div>
          </div>
        </section>

        {/* ════ Right: Incident Log ════ */}
        <aside className="w-[360px] flex-shrink-0 glass-panel-strong flex flex-col animate-fade-in">
          {/* Header */}
          <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-zinc-400">
                <path d="M12 9v4l3 3"/>
                <circle cx="12" cy="12" r="10"/>
              </svg>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
                Incident Log
              </h2>
            </div>
            <span className="text-[10px] font-mono text-zinc-600">
              {incidents.length} events
            </span>
          </div>

          {/* Incident List */}
          <div
            ref={incidentScrollRef}
            className="incident-scroll flex-1 p-3 space-y-2 min-h-0"
          >
            {incidents.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center py-16">
                <div className="w-12 h-12 rounded-xl bg-zinc-800/50 flex items-center justify-center mb-3">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-zinc-600">
                    <path d="M9 12l2 2 4-4"/>
                    <path d="M12 2l7 4v6c0 5.25-3.5 9.74-7 11-3.5-1.26-7-5.75-7-11V6l7-4z"/>
                  </svg>
                </div>
                <p className="text-xs text-zinc-500 font-medium">No incidents detected</p>
                <p className="text-[10px] text-zinc-600 mt-1">System is monitoring</p>
              </div>
            ) : (
              incidents.map((incident, idx) => (
                <div
                  key={incident.id}
                  className={`incident-card severity-${incident.severity_label.toLowerCase()} ${
                    idx < 3 ? "animate-slide-in" : ""
                  }`}
                  style={{ animationDelay: `${idx * 50}ms` }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className={severityBadgeClass(incident.severity_label)}>
                      <span className={`w-1.5 h-1.5 rounded-full ${severityDotClass(incident.severity_label)}`} />
                      {incident.severity_label}
                    </span>
                    <span className="text-[10px] text-zinc-500 font-mono">
                      {timeAgo(incident.timestamp)}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Person</span>
                      <span className="font-mono text-zinc-300">#{incident.person_id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Velocity</span>
                      <span className="font-mono text-zinc-300">
                        {incident.max_velocity.toFixed(1)}
                        <span className="text-zinc-600"> px/f</span>
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Conf</span>
                      <span className="font-mono text-zinc-300">
                        {(incident.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Time</span>
                      <span className="font-mono text-zinc-300">
                        {formatTime(incident.timestamp)}
                      </span>
                    </div>
                  </div>

                  {/* Bounding box micro-detail */}
                  <div className="mt-2 text-[9px] font-mono text-zinc-600 truncate">
                    bbox: [{incident.bbox_x1.toFixed(0)}, {incident.bbox_y1.toFixed(0)}, {incident.bbox_x2.toFixed(0)}, {incident.bbox_y2.toFixed(0)}]
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-2.5 border-t border-white/5 flex items-center justify-between">
            <span className="text-[10px] text-zinc-600">
              24h total: <span className="text-zinc-400 font-mono">{stats.total}</span>
            </span>
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
              <span className="text-[10px] text-zinc-500">Auto-updating</span>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
