"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";

interface Incident {
  id: string;
  timestamp: number;
  probability: number;
  severity_label: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  fps: number;
  inference_ms: number;
}

interface Stats {
  LOW: number;
  MEDIUM: number;
  HIGH: number;
  CRITICAL: number;
  total: number;
}

type SourceMode = "webcam" | "upload" | "rtsp";
type ConnectionStatus = "connecting" | "connected" | "disconnected";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? `${window.location.protocol}//${window.location.hostname}:7860`
    : typeof window !== "undefined"
      ? window.location.origin
      : "http://localhost:7860");

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.hostname}:7860/ws/stream`
    : typeof window !== "undefined"
      ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/stream`
      : "ws://localhost:7860/ws/stream");

const EMPTY_STATS: Stats = { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0, total: 0 };

export default function DashboardPage() {
  const [frame, setFrame] = useState<string | null>(null);
  const [isViolence, setIsViolence] = useState(false);
  const [probability, setProbability] = useState(0);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [sourceMode, setSourceMode] = useState<SourceMode>("webcam");
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [stats, setStats] = useState<Stats>(EMPTY_STATS);
  const [rtspUrl, setRtspUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let stopped = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let currentSocket: WebSocket | undefined;

    async function refresh() {
      try {
        const [incidentResponse, statsResponse] = await Promise.all([
          fetch(`${API_URL}/api/incidents?limit=50`),
          fetch(`${API_URL}/api/stats?hours=24`),
        ]);
        if (incidentResponse.ok) {
          const payload = await incidentResponse.json();
          setIncidents(payload.incidents ?? []);
        }
        if (statsResponse.ok) setStats(await statsResponse.json());
      } catch {
        setError("Backend is not reachable. Start the API server and try again.");
      }
    }

    function connect() {
      if (stopped) return;
      setStatus("connecting");
      currentSocket = new WebSocket(WS_URL);
      currentSocket.onopen = () => setStatus("connected");
      currentSocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === "frame") {
            setFrame(message.data);
            setIsViolence(message.is_violence === true);
            setProbability(Number(message.probability) || 0);
            setFps(Number(message.fps) || 0);
            setLatency(Number(message.inference_ms) || 0);
          } else if (message.type === "incident") {
            const incident = message.data as Incident;
            setIncidents((previous) => [incident, ...previous].slice(0, 100));
            void refresh();
          }
        } catch {
          setError("Received an unreadable message from the video stream.");
        }
      };
      currentSocket.onclose = () => {
        setStatus("disconnected");
        if (!stopped) reconnectTimer = setTimeout(connect, 3000);
      };
      currentSocket.onerror = () => currentSocket?.close();
    }

    void refresh();
    const connectTimer = setTimeout(connect, 0);

    return () => {
      stopped = true;
      if (connectTimer) clearTimeout(connectTimer);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      currentSocket?.close();
    };
  }, []);

  async function switchSource(endpoint: string, options?: RequestInit) {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}${endpoint}`, options);
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail ?? "Could not switch video source.");
      }
      setFrame(null);
      setIsViolence(false);
      setProbability(0);
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not switch video source.");
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function selectWebcam() {
    if (await switchSource("/source/webcam", { method: "POST" })) {
      setSourceMode("webcam");
      setFilename(null);
    }
  }

  async function selectRtsp() {
    const input = rtspUrl.trim();
    if (!/^rtsps?:\/\/\S+$/i.test(input)) {
      setError("Enter a valid rtsp:// or rtsps:// stream URL.");
      return;
    }
    if (
      await switchSource("/source/rtsp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: input }),
      })
    ) {
      setSourceMode("rtsp");
      setFilename(null);
    }
  }

  async function uploadVideo(file?: File) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".mp4")) {
      setError("Choose an MP4 video file.");
      return;
    }
    const body = new FormData();
    body.append("file", file);
    if (await switchSource("/upload", { method: "POST", body })) {
      setSourceMode("upload");
      setFilename(file.name);
    }
    if (fileInput.current) fileInput.current.value = "";
  }

  const statusText =
    status === "connected" ? "CONNECTED" : status === "connecting" ? "CONNECTING" : "OFFLINE";

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#08090b] text-white">
      <header className="mx-4 mt-4 flex items-center justify-between rounded-2xl border border-white/[0.08] bg-white/[0.04] px-5 py-4 shadow-xl shadow-black/20 backdrop-blur-2xl md:mx-8">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-indigo-300">
            Edge Vision
          </p>
          <h1 className="mt-1 text-lg font-semibold tracking-tight">Violence Monitor</h1>
        </div>
        <div className="flex items-center gap-5 text-right">
          <div>
            <p className="font-mono text-sm text-zinc-200">{fps.toFixed(1)} FPS</p>
            <p className="text-[10px] uppercase tracking-wider text-zinc-500">Throughput</p>
          </div>
          <div>
            <p className="font-mono text-sm text-zinc-200">{latency.toFixed(1)} ms</p>
            <p className="text-[10px] uppercase tracking-wider text-zinc-500">Latency</p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                status === "connected" ? "bg-emerald-400 shadow-[0_0_12px_#34d399]" :
                  status === "connecting" ? "animate-pulse bg-amber-300" : "bg-zinc-600"
              }`}
            />
            <span className="text-[10px] font-semibold tracking-widest text-zinc-400">{statusText}</span>
          </div>
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto px-4 pb-4 pt-4 md:px-8 lg:flex-row">
        <section className="flex min-h-[420px] min-w-0 flex-1 flex-col gap-4">
          <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-3xl border border-white/10 bg-black shadow-[0_24px_100px_-40px_rgba(99,102,241,0.35)]">
            {frame ? (
              <Image
                src={`data:image/jpeg;base64,${frame}`}
                alt="Live violence-monitoring video"
                width={640}
                height={480}
                unoptimized
                className="h-full w-full object-contain"
              />
            ) : (
              <div className="px-6 text-center">
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.04] text-2xl text-zinc-500">
                  ◉
                </div>
                <p className="text-sm font-medium text-zinc-300">
                  {status === "connected" ? "Waiting for camera frames" : "Connecting to the inference service"}
                </p>
                <p className="mt-2 text-xs text-zinc-600">
                  Select a video source below to begin monitoring.
                </p>
              </div>
            )}

            <div className="absolute left-4 top-4 rounded-full border border-white/10 bg-black/50 px-3 py-1.5 text-[10px] font-medium uppercase tracking-widest text-white/70 backdrop-blur-lg">
              {sourceMode === "webcam" ? "Local webcam" : sourceMode === "rtsp" ? "RTSP stream" : filename ?? "Uploaded video"}
            </div>
            {frame && (
              <div
                role={isViolence ? "alert" : "status"}
                aria-live={isViolence ? "assertive" : "polite"}
                className={`absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full border px-5 py-3 text-xs font-bold tracking-widest shadow-2xl backdrop-blur-xl ${
                  isViolence
                    ? "animate-pulse border-red-300 bg-red-600 text-white shadow-red-950/70"
                    : "border-emerald-300/30 bg-emerald-950/90 text-emerald-300 shadow-emerald-950/50"
                }`}
              >
                {isViolence ? "🚨 VIOLENCE DETECTED" : "SAFE: NO VIOLENCE"}
                <span className="ml-2 font-mono opacity-75">{(probability * 100).toFixed(0)}%</span>
              </div>
            )}
          </div>

          <section className="rounded-3xl border border-white/[0.08] bg-white/[0.035] p-4 shadow-xl shadow-black/20 backdrop-blur-2xl md:p-5">
            <div className="mb-4">
              <h2 className="text-sm font-semibold">Video source</h2>
              <p className="mt-1 text-xs text-zinc-500">Choose webcam, an RTSP stream, or an MP4 clip.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                disabled={busy}
                aria-pressed={sourceMode === "webcam"}
                onClick={selectWebcam}
                className={`rounded-xl border px-4 py-2.5 text-xs font-semibold transition disabled:opacity-50 ${
                  sourceMode === "webcam" ? "border-white/20 bg-white text-zinc-950" :
                    "border-white/10 bg-white/[0.04] text-zinc-300 hover:bg-white/10"
                }`}
              >
                Webcam
              </button>
              <button
                type="button"
                disabled={busy}
                aria-pressed={sourceMode === "upload"}
                onClick={() => fileInput.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  void uploadVideo(event.dataTransfer.files[0]);
                }}
                className={`rounded-xl border px-4 py-2.5 text-xs font-semibold transition disabled:opacity-50 ${
                  sourceMode === "upload" ? "border-indigo-300/40 bg-indigo-400/20 text-indigo-100" :
                    "border-white/10 bg-white/[0.04] text-zinc-300 hover:bg-white/10"
                }`}
              >
                {busy && sourceMode === "upload" ? "Uploading…" : "Upload Video"}
              </button>
              <input
                ref={fileInput}
                type="file"
                accept=".mp4,video/mp4"
                className="hidden"
                onChange={(event) => void uploadVideo(event.target.files?.[0])}
              />
              <input
                type="url"
                value={rtspUrl}
                onChange={(event) => setRtspUrl(event.target.value)}
                placeholder="rtsp://camera-address/stream"
                aria-label="RTSP stream URL"
                className="min-w-56 flex-1 rounded-xl border border-white/10 bg-black/30 px-3 py-2.5 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-indigo-400/50"
              />
              <button
                type="button"
                disabled={busy}
                aria-pressed={sourceMode === "rtsp"}
                onClick={selectRtsp}
                className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2.5 text-xs font-semibold text-zinc-300 transition hover:bg-white/10 disabled:opacity-50"
              >
                {busy && sourceMode === "rtsp" ? "Connecting…" : "Use RTSP"}
              </button>
            </div>
            {error && <p role="alert" className="mt-3 text-xs text-red-300">{error}</p>}
          </section>
        </section>

        <aside className="flex w-full flex-col gap-4 lg:w-80 lg:flex-shrink-0">
          <div className="grid grid-cols-2 gap-3">
            {(["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((level) => (
              <div key={level} className="rounded-2xl border border-white/[0.08] bg-white/[0.035] p-4">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">{level}</p>
                <p className="mt-2 font-mono text-2xl text-zinc-100">{stats[level]}</p>
              </div>
            ))}
          </div>
          <div className="flex min-h-64 flex-1 flex-col overflow-hidden rounded-3xl border border-white/[0.08] bg-white/[0.035] backdrop-blur-2xl">
            <div className="flex items-center justify-between border-b border-white/[0.07] px-4 py-4">
              <h2 className="text-sm font-semibold">Incident log</h2>
              <span className="text-[10px] text-zinc-500">{stats.total} events</span>
            </div>
            <div className="flex-1 space-y-2 overflow-auto p-3">
              {incidents.length === 0 ? (
                <p className="py-10 text-center text-xs text-zinc-600">No violence events logged.</p>
              ) : incidents.map((incident) => (
                <article key={incident.id} className="rounded-2xl border border-white/[0.06] bg-black/20 p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold tracking-widest text-red-300">{incident.severity_label}</span>
                    <time className="text-[10px] text-zinc-500">
                      {new Date(incident.timestamp * 1000).toLocaleTimeString()}
                    </time>
                  </div>
                  <div className="mt-2 flex justify-between font-mono text-[10px] text-zinc-500">
                    <span>Score {(incident.probability * 100).toFixed(0)}%</span>
                    <span>{incident.fps.toFixed(1)} FPS · {incident.inference_ms.toFixed(1)} ms</span>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
