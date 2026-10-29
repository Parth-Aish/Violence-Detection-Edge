"""Dataset-Wide Evaluation & Kinematic Performance Reporting.

Evaluates the ST-GCN + Relational Gating architecture against all raw video samples
(data/raw/Violence and data/raw/NonViolence) to generate a comprehensive confusion matrix,
F1/accuracy metrics, and misclassification error analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# Ensure backend directory is in sys.path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from core.inference import PoseSTGCNDetector


def natural_sort_key(path: Path) -> tuple[int | str, ...]:
    """Sort filenames with natural numeric ordering (e.g. V_2 before V_10)."""
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


@dataclass
class VideoEvaluationResult:
    filename: str
    relative_path: str
    ground_truth: str  # "Violence" or "Safe"
    predicted_verdict: str  # "Violence" or "Safe"
    confidence: float
    peak_probability: float
    is_correct: bool
    incident_count: int
    frame_count: int
    video_duration_sec: float
    processing_time_sec: float
    fps: float


def collect_dataset_files(
    root_dir: Path,
    limit_per_class: Optional[int] = None,
) -> list[tuple[Path, str]]:
    """Dynamically glob and sort all video files from raw dataset folders."""
    v_dir = root_dir / "data" / "raw" / "Violence"
    nv_dir = root_dir / "data" / "raw" / "NonViolence"

    if not v_dir.is_dir() or not nv_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directories not found. Checked: {v_dir} and {nv_dir}"
        )

    v_files = sorted(v_dir.glob("*.mp4"), key=natural_sort_key)
    nv_files = sorted(nv_dir.glob("*.mp4"), key=natural_sort_key)

    if limit_per_class and limit_per_class > 0:
        v_files = v_files[:limit_per_class]
        nv_files = nv_files[:limit_per_class]

    dataset = [(f, "Violence") for f in v_files] + [(f, "Safe") for f in nv_files]
    return dataset


def run_evaluation(
    dataset: list[tuple[Path, str]],
    stride: int = 2,
    threads: Optional[int] = None,
    output_json: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute sequential batch evaluation across all provided dataset videos."""
    available_cores = os.cpu_count() or 4
    optimal_threads = threads or min(4, available_cores)

    print("=" * 80)
    print("  ENTERPRISE ST-GCN + RELATIONAL GATE: DATASET-WIDE EVALUATION")
    print("=" * 80)
    print(f"Total Videos Queued: {len(dataset)}")
    v_count = sum(1 for _, gt in dataset if gt == "Violence")
    nv_count = sum(1 for _, gt in dataset if gt == "Safe")
    print(f"  • Violence Altercation Clips: {v_count}")
    print(f"  • Non-Violence / Safe Clips:   {nv_count}")
    print(f"Detector Configuration: ORT Threads={optimal_threads}, Pose Stride={stride}")
    print("-" * 80)

    detector = PoseSTGCNDetector(
        intra_op_threads=optimal_threads,
        pose_stride=stride,
    )

    results: list[VideoEvaluationResult] = []
    misclassified: list[VideoEvaluationResult] = []

    total_frames = 0
    total_processing_sec = 0.0
    t_global_start = time.perf_counter()

    for idx, (video_path, ground_truth) in enumerate(dataset, start=1):
        rel_path = str(video_path.relative_to(video_path.parents[2]))
        sys.stdout.write(f"[{idx:4d}/{len(dataset):4d}] Analyzing {video_path.name:<16} (GT: {ground_truth:<8}) ... ")
        sys.stdout.flush()

        try:
            report = detector.analyze_video(video_path)
            pred_verdict = report["overall_verdict"]
            peak_prob = report.get("peak_probability", 0.0)
            confidence = report.get("confidence", 0.0)
            proc_time = max(report.get("processing_time_sec", 0.001), 0.001)
            f_count = report.get("frame_count", 0)
            v_duration = report.get("video_duration_sec", 0.0)
            incidents = len(report.get("incident_timestamps", []))
            calc_fps = f_count / proc_time if proc_time > 0 else 0.0

            is_correct = pred_verdict == ground_truth
            status_tag = "[PASS] CORRECT" if is_correct else "[FAIL] MISCLASSIFIED"

            print(
                f"{status_tag:<19} Pred={pred_verdict:<8} (PeakProb={peak_prob * 100:5.1f}%, "
                f"FPS={calc_fps:4.1f}, Time={proc_time:4.2f}s)"
            )

            res = VideoEvaluationResult(
                filename=video_path.name,
                relative_path=rel_path,
                ground_truth=ground_truth,
                predicted_verdict=pred_verdict,
                confidence=confidence,
                peak_probability=peak_prob,
                is_correct=is_correct,
                incident_count=incidents,
                frame_count=f_count,
                video_duration_sec=v_duration,
                processing_time_sec=proc_time,
                fps=round(calc_fps, 1),
            )
            results.append(res)
            if not is_correct:
                misclassified.append(res)

            total_frames += f_count
            total_processing_sec += proc_time

            # Periodic checkpoint saving every 50 videos
            if output_json and idx % 50 == 0:
                cur_tp = sum(1 for r in results if r.ground_truth == "Violence" and r.predicted_verdict == "Violence")
                cur_fp = sum(1 for r in results if r.ground_truth == "Safe" and r.predicted_verdict == "Violence")
                cur_tn = sum(1 for r in results if r.ground_truth == "Safe" and r.predicted_verdict == "Safe")
                cur_fn = sum(1 for r in results if r.ground_truth == "Violence" and r.predicted_verdict == "Safe")
                cur_acc = (cur_tp + cur_tn) / len(results) if results else 0.0
                cur_summary = {
                    "checkpoint_progress": f"{idx}/{len(dataset)}",
                    "accuracy": round(cur_acc, 4),
                    "tp": cur_tp, "fp": cur_fp, "tn": cur_tn, "fn": cur_fn,
                    "results_count": len(results),
                    "misclassified_count": len(misclassified),
                }
                try:
                    with open(output_json, "w", encoding="utf-8") as f:
                        json.dump(cur_summary, f, indent=2)
                except Exception:
                    pass

        except Exception as exc:
            print(f"[ERROR] Failed processing {video_path.name}: {exc}")

    t_global_total = time.perf_counter() - t_global_start

    # Metrics calculation
    tp = sum(1 for r in results if r.ground_truth == "Violence" and r.predicted_verdict == "Violence")
    fp = sum(1 for r in results if r.ground_truth == "Safe" and r.predicted_verdict == "Violence")
    tn = sum(1 for r in results if r.ground_truth == "Safe" and r.predicted_verdict == "Safe")
    fn = sum(1 for r in results if r.ground_truth == "Violence" and r.predicted_verdict == "Safe")

    total_evaluated = len(results)
    accuracy = (tp + tn) / total_evaluated if total_evaluated > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    overall_throughput_fps = total_frames / total_processing_sec if total_processing_sec > 0 else 0.0

    print("\n" + "=" * 80)
    print("  COMPREHENSIVE PERFORMANCE EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Videos Analyzed:      {total_evaluated}")
    print(f"  • Violence Altercations:   {v_count}")
    print(f"  • Normal / Non-Violence:   {nv_count}")
    print(f"Total Frames Processed:     {total_frames:,} frames")
    print(f"Total Processing Time:      {t_global_total:.2f} s ({t_global_total / 60:.2f} min)")
    print(f"Average System Throughput:  {overall_throughput_fps:.1f} FPS (Target > 30.0 FPS)")
    print("-" * 80)
    print("CONFUSION MATRIX:")
    print("                       Predicted Safe    Predicted Violence      Total")
    print(f"  Actual Safe (NV)          {tn:5d}               {fp:5d}            {tn + fp:5d}")
    print(f"  Actual Violence (V)       {fn:5d}               {tp:5d}            {fn + tp:5d}")
    print(f"  Total                     {tn + fn:5d}               {fp + tp:5d}            {total_evaluated:5d}")
    print("-" * 80)
    print(f"  Accuracy:                 {accuracy * 100:6.2f}%   ({tp + tn}/{total_evaluated})")
    print(f"  Precision:                {precision * 100:6.2f}%   ({tp}/{tp + fp})")
    print(f"  Recall (Sensitivity):     {recall * 100:6.2f}%   ({tp}/{tp + fn})")
    print(f"  Specificity:              {specificity * 100:6.2f}%   ({tn}/{tn + fp})")
    print(f"  F1 Score:                 {f1:8.4f}")
    print("=" * 80)

    if misclassified:
        print(f"\nMISCLASSIFIED VIDEOS ({len(misclassified)} total):")
        print(f"{'Filename':<16} {'Ground Truth':<14} {'Predicted':<12} {'Peak Prob':<12} {'Triggers':<10} {'Path'}")
        print("-" * 80)
        for m in misclassified:
            print(
                f"{m.filename:<16} {m.ground_truth:<14} {m.predicted_verdict:<12} "
                f"{m.peak_probability * 100:5.1f}%       {m.incident_count:<10} {m.relative_path}"
            )
        print("-" * 80)
    else:
        print("\nPERFECT CLASSIFICATION: Zero false positives and zero false negatives recorded!")

    summary = {
        "total_evaluated": total_evaluated,
        "violence_count": v_count,
        "non_violence_count": nv_count,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "average_fps": round(overall_throughput_fps, 1),
            "total_frames": total_frames,
            "total_processing_time_sec": round(t_global_total, 2),
        },
        "misclassified_count": len(misclassified),
        "misclassified": [asdict(m) for m in misclassified],
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved structured evaluation report to: {output_json}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset-Wide Evaluation & Kinematic Performance Reporting"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of clips to evaluate per class (e.g. --limit 50). Default is all.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Pose estimation frame stride (default=2 for 50+ FPS throughput).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ONNX Runtime intra-op threads (default=4).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("backend/data/evaluation_report.json"),
        help="Path to save the structured JSON evaluation summary.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset = collect_dataset_files(project_root, limit_per_class=args.limit)

    run_evaluation(
        dataset=dataset,
        stride=args.stride,
        threads=args.threads,
        output_json=args.output_json.resolve() if args.output_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
