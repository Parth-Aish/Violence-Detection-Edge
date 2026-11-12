"use client";

import { useState } from "react";
import {
  Activity,
  BookOpen,
  Check,
  Copy,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Shield,
  Terminal,
} from "lucide-react";

const LOCAL_CLI_COMMANDS = `git clone https://github.com/Parth-Aish/Violence-Detection-Edge.git
cd Violence-Detection-Edge
pip install -r requirements.txt
python backend/run_live.py --source 0`;

const ABLATION_ROWS = [
  {
    config: "Baseline ST-GCN (Raw Pose Graph)",
    precision: "69.86%",
    recall: "97.33%",
    f1: "0.813",
    fp: "63 / 143 (44.1%)",
    highlight: false,
  },
  {
    config: "+ Kalman Tracking (only)",
    precision: "69.19%",
    recall: "97.33%",
    f1: "0.809",
    fp: "65 / 143 (45.5%)",
    highlight: false,
  },
  {
    config: "+ Naive Centroid Gate (d ≤ 0.22W)",
    precision: "69.86%",
    recall: "97.33%",
    f1: "0.813",
    fp: "63 / 143 (44.1%)",
    highlight: false,
  },
  {
    config: "+ Relational Strike Gate (only)",
    precision: "70.19%",
    recall: "97.33%",
    f1: "0.816",
    fp: "62 / 143 (43.4%)",
    highlight: false,
  },
  {
    config: "+ Tracking + Naive Centroid Gate",
    precision: "71.22%",
    recall: "97.33%",
    f1: "0.823",
    fp: "59 / 143 (41.3%)",
    highlight: false,
  },
  {
    config: "+ Tracking + Strike Gate (Cloud Config)",
    precision: "70.53%",
    recall: "97.33%",
    f1: "0.818",
    fp: "61 / 143 (42.7%)",
    highlight: false,
  },
  {
    config: "Full System (+ MobileNetV3 Scene Gating)",
    precision: "76.04%",
    recall: "97.33%",
    f1: "0.854",
    fp: "46 / 143 (32.2%)",
    highlight: true,
  },
];

export default function Documentation() {
  const [copiedCli, setCopiedCli] = useState(false);

  const handleCopyCli = () => {
    navigator.clipboard.writeText(LOCAL_CLI_COMMANDS);
    setCopiedCli(true);
    setTimeout(() => setCopiedCli(false), 2000);
  };

  return (
    <section
      id="about-documentation-section"
      className="border-t border-zinc-800 pt-8 mt-4 space-y-6"
    >
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-zinc-400">
            <BookOpen className="h-3.5 w-3.5 text-zinc-300" />
            <span>Technical Documentation &amp; Benchmarks</span>
          </div>
          <h2 className="text-base font-semibold tracking-tight text-zinc-100">
            Architecture Reference, Evaluation Protocol &amp; Local Edge Setup
          </h2>
        </div>
        <span className="font-mono text-[11px] px-2.5 py-1 rounded border border-zinc-800 bg-[#111114] text-zinc-400 self-start sm:self-auto">
          SPEC // ST-GCN + KALMAN + ONNX
        </span>
      </div>

      {/* Key Highlight Metrics Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 font-mono">
        <div className="rounded-lg border border-zinc-800 bg-[#111114] p-4">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500">
            Held-Out F1 Score
          </div>
          <div className="text-2xl font-bold text-emerald-400 mt-1 tabular-nums">
            0.854
          </div>
          <div className="text-[11px] text-zinc-400 mt-1">
            76.04% Prec • 97.33% Rec
          </div>
        </div>

        <div className="rounded-lg border border-zinc-800 bg-[#111114] p-4">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500">
            Source-Partitioned Corpus
          </div>
          <div className="text-2xl font-bold text-zinc-100 mt-1 tabular-nums">
            1,951 Videos
          </div>
          <div className="text-[11px] text-zinc-400 mt-1">
            70/15/15 Chronological Split
          </div>
        </div>

        <div className="rounded-lg border border-zinc-800 bg-[#111114] p-4">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500">
            CPU Throughput (w/ Decode)
          </div>
          <div className="text-2xl font-bold text-indigo-400 mt-1 tabular-nums">
            48.5 FPS
          </div>
          <div className="text-[11px] text-zinc-400 mt-1">
            20.63 ± 13.49 ms (i5-1135G7)
          </div>
        </div>

        <div className="rounded-lg border border-zinc-800 bg-[#111114] p-4">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500">
            Held-Out Accuracy
          </div>
          <div className="text-2xl font-bold text-zinc-100 mt-1 tabular-nums">
            82.94%
          </div>
          <div className="text-[11px] text-zinc-400 mt-1">
            293 Unseen Test Clips
          </div>
        </div>
      </div>

      {/* Architecture + Evaluation Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left 7 Cols: System Architecture Breakdown */}
        <div className="lg:col-span-7 rounded-lg border border-zinc-800 bg-[#111114] p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2.5">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-zinc-400" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-200">
                Pipeline Architecture (YOLOv8-Pose + ST-GCN + Kalman)
              </h3>
            </div>
            <span className="font-mono text-[10px] text-zinc-500">
              99.8% DATA REDUCTION
            </span>
          </div>

          <div className="space-y-3 text-xs text-zinc-300 leading-relaxed">
            <div className="rounded border border-zinc-800/80 bg-zinc-950/60 p-3 space-y-1">
              <div className="font-mono text-[11px] font-semibold text-zinc-100 flex items-center gap-2">
                <Cpu className="h-3.5 w-3.5 text-indigo-400" />
                <span>1. YOLOv8n-Pose Extraction &amp; Mid-Hip Normalization</span>
              </div>
              <p className="text-zinc-400">
                Each frame is letterboxed to{" "}
                <code className="font-mono text-zinc-200">320×320</code> and
                decoded into 17 COCO anatomical keypoints. Joints are centered at
                the mid-hip origin and scaled by torso length while keeping
                occluded joints strictly masked at{" "}
                <code className="font-mono text-zinc-200">[0.0, 0.0]</code>.
              </p>
            </div>

            <div className="rounded border border-zinc-800/80 bg-zinc-950/60 p-3 space-y-1">
              <div className="font-mono text-[11px] font-semibold text-zinc-100 flex items-center gap-2">
                <GitBranch className="h-3.5 w-3.5 text-amber-400" />
                <span>2. Multi-Agent Kalman Tracking &amp; Hungarian Assignment</span>
              </div>
              <p className="text-zinc-400">
                An 8-dimensional constant-velocity Kalman filter{" "}
                <code className="font-mono text-zinc-200">
                  [cx, cy, w, h, vx, vy, vw, vh]
                </code>{" "}
                paired with Hungarian bipartite matching ({" "}
                <code className="font-mono text-zinc-200">max_age=10</code>);
                qualitative validation indicates it preserves track identities
                across transient occlusions and reduces ID switches vs. greedy
                IoU matching.
              </p>
            </div>

            <div className="rounded border border-zinc-800/80 bg-zinc-950/60 p-3 space-y-1">
              <div className="font-mono text-[11px] font-semibold text-zinc-100 flex items-center gap-2">
                <Activity className="h-3.5 w-3.5 text-emerald-400" />
                <span>3. ST-GCN Topology &amp; Relational Strike Gating</span>
              </div>
              <p className="text-zinc-400">
                Thirty uniformly sampled poses form a{" "}
                <code className="font-mono text-zinc-200">(1, 2, 30, 17)</code>{" "}
                graph tensor evaluated by 3 residual Spatial-Temporal Graph Conv
                blocks (<code className="font-mono text-zinc-200">257 KB</code>{" "}
                ONNX). The Relational Strike Gate checks effector-to-target
                proximity (<code className="font-mono text-zinc-200">&lt; 0.22·W</code>)
                and box convergence (<code className="font-mono text-zinc-200">IoU &gt; 0</code>)
                to attenuate isolated non-contact false positives.
              </p>
            </div>
          </div>
        </div>

        {/* Right 5 Cols: Ablation Benchmarks + Local CLI Setup */}
        <div className="lg:col-span-5 space-y-5">
          {/* Holdout Ablation Table */}
          <div className="rounded-lg border border-zinc-800 bg-[#111114] p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-zinc-400" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-200">
                  Held-Out Ablation (293 Test Clips)
                </h3>
              </div>
              <span className="font-mono text-[10px] text-zinc-500">
                SOURCE-ID SPLIT
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse font-mono text-[11px] tabular-nums">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500">
                    <th className="py-1.5 pr-2 font-medium">Config</th>
                    <th className="py-1.5 px-2 font-medium text-right">Prec</th>
                    <th className="py-1.5 px-2 font-medium text-right">Rec</th>
                    <th className="py-1.5 pl-2 font-medium text-right">F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/70">
                  {ABLATION_ROWS.map((row) => (
                    <tr
                      key={row.config}
                      className={
                        row.highlight
                          ? "bg-emerald-950/20 text-emerald-300"
                          : "text-zinc-400"
                      }
                    >
                      <td className="py-2 pr-2 font-sans text-[11px]">
                        {row.config}
                      </td>
                      <td className="py-2 px-2 text-right">{row.precision}</td>
                      <td className="py-2 px-2 text-right">{row.recall}</td>
                      <td className="py-2 pl-2 text-right font-bold">
                        {row.f1}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Local Edge Setup Block */}
          <div className="rounded-lg border border-zinc-800 bg-[#111114] p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <div className="flex items-center gap-2">
                <Terminal className="h-4 w-4 text-emerald-400" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-200">
                  Local Edge CLI &amp; Live Camera Setup
                </h3>
              </div>
              <button
                type="button"
                id="docs-copy-cli-btn"
                onClick={handleCopyCli}
                className="flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-mono bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-700 transition"
              >
                {copiedCli ? (
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

            <pre className="font-mono text-[11px] text-zinc-300 bg-zinc-950 border border-zinc-800 rounded p-3 overflow-x-auto leading-relaxed">
              <code>{LOCAL_CLI_COMMANDS}</code>
            </pre>

            <div className="flex items-center gap-2 text-[11px] font-mono text-zinc-500">
              <Shield className="h-3.5 w-3.5 text-zinc-400 flex-shrink-0" />
              <span>
                Edge Mode guarantees zero-upload WASM/WebGPU privacy; Cloud Mode acts as a standard FastAPI REST API.
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

