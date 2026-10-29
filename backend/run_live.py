"""Standalone Local Real-Time Edge Inference Script.

Runs high-speed (50+ FPS) violence detection on webcam or RTSP camera feeds,
rendering COCO skeleton edges (imported from backend.core.graph) and the
runtime Relational Strike Distance between the two closest actors.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure backend directory is in sys.path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from core.graph import COCO_EDGES
from core.inference import PoseSTGCNDetector


def draw_hud(
    frame: np.ndarray,
    is_fighting: bool,
    probability: float,
    fps: float,
    inference_ms: float,
    person_count: int,
    min_strike_dist: float,
    gate_triggered: bool,
) -> None:
    """Draw enterprise surveillance edge HUD overlay."""
    height, width = frame.shape[:2]

    # Top status banner
    banner_height = 56
    status_color = (34, 34, 220) if is_fighting else (34, 180, 80)  # Red : Green (BGR)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), status_color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Status text
    status_label = "ALERT: VIOLENT ALTERCATION DETECTED" if is_fighting else "STATUS: NORMAL / SECURE"
    cv2.putText(
        frame,
        status_label,
        (16, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Right-aligned confidence badge
    prob_text = f"CONF: {probability * 100:.1f}%"
    (tw, _), _ = cv2.getTextSize(prob_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    cv2.putText(
        frame,
        prob_text,
        (width - tw - 16, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Bottom telemetry bar
    bottom_y = height - 12
    cv2.rectangle(frame, (0, height - 34), (width, height), (20, 20, 20), -1)

    gate_status = "STRIKE-GATE OVERRIDE" if gate_triggered else "ST-GCN ACTIVE"
    dist_str = f"{min_strike_dist:.1f}px" if min_strike_dist < 9000 else "N/A"
    telemetry = (
        f"FPS: {fps:4.1f} | Latency: {inference_ms:4.1f}ms | Persons: {person_count} "
        f"| StrikeDist: {dist_str} | Mode: {gate_status} | 'q' to Quit"
    )
    cv2.putText(
        frame,
        telemetry,
        (12, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int]) -> None:
    """Render COCO-17 skeleton limbs and joint circles on the frame using COCO_EDGES."""
    pts = keypoints[:, :2]

    # Draw limb connection lines using authoritative COCO_EDGES
    for p1, p2 in COCO_EDGES:
        if p1 < len(pts) and p2 < len(pts):
            pt1 = (int(pts[p1, 0]), int(pts[p1, 1]))
            pt2 = (int(pts[p2, 0]), int(pts[p2, 1]))
            if pt1 != (0, 0) and pt2 != (0, 0):
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joint nodes
    for i in range(len(pts)):
        pt = (int(pts[i, 0]), int(pts[i, 1]))
        if pt != (0, 0):
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, color, -1, cv2.LINE_AA)


def draw_strike_distance_line(
    frame: np.ndarray,
    strike_line: tuple[float, float, float, float],
    min_dist: float,
    is_fighting: bool,
) -> None:
    """Visually draw a colored indicator line representing the Relational Strike Distance."""
    x1, y1, x2, y2 = [int(v) for v in strike_line]
    line_color = (0, 0, 255) if is_fighting else (0, 215, 255)  # Red if fight, Gold/Yellow if safe

    # Draw dotted or thick interaction line
    cv2.line(frame, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)
    cv2.circle(frame, (x1, y1), 6, line_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x2, y2), 6, line_color, -1, cv2.LINE_AA)

    # Midpoint label
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2
    label = f"{min_dist:.0f}px"
    cv2.putText(
        frame,
        label,
        (mx + 6, my - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        line_color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Edge Violence Detection System (ST-GCN)")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (0, 1), RTSP URL ('rtsp://...'), or video file path.",
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
        default=None,
        help="ONNX Runtime intra-op threads (defaults to os.cpu_count()).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture frame width (default=640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture frame height (default=480).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless terminal mode without opening GUI window.",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Optional path to record annotated video output (e.g. output.mp4).",
    )
    args = parser.parse_args()

    # Determine camera source
    source_val = int(args.source) if args.source.isdigit() else args.source
    print(f"[EDGE-AI] Opening video source: {args.source}...")
    cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source '{args.source}'.")
        print("Tip: If testing without a live webcam, pass a sample video:")
        print("     python run_live.py --source data/raw/Violence/V_1.mp4")
        sys.exit(1)

    if isinstance(source_val, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30.0)

    # Unlock full CPU threads for 50+ FPS throughput
    available_cores = os.cpu_count() or 4
    threads = args.threads or available_cores
    print(f"[EDGE-AI] Initializing PoseSTGCNDetector (threads={threads}, stride={args.stride})...")
    detector = PoseSTGCNDetector(intra_op_threads=threads, pose_stride=max(1, args.stride))

    window_name = "Edge Vision - Real-Time ST-GCN & Relational Strike Detection"
    gui_available = not args.headless

    if gui_available:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 720)
            print("[EDGE-AI] Stream active. Displaying local GUI window. Press 'q' or ESC in window to exit.")
        except cv2.error as exc:
            gui_available = False
            print(f"[WARNING] OpenCV GUI window unavailable ({exc}).")
            print("[EDGE-AI] Falling back to Terminal Telemetry Mode (press Ctrl+C to exit).")
    else:
        print("[EDGE-AI] Running in Headless Terminal Mode (press Ctrl+C to exit).")

    video_writer: cv2.VideoWriter | None = None
    if args.save_output:
        out_path = Path(args.save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (args.width, args.height))
        print(f"[EDGE-AI] Recording annotated feed to: {out_path}")

    fps_smooth = 30.0
    last_tick = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not isinstance(source_val, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    detector.reset()
                    continue
                else:
                    print("\n[EDGE-AI] Camera feed ended.")
                    break

            now_tick = time.perf_counter()
            dt = max(now_tick - last_tick, 0.001)
            fps_instant = 1.0 / dt
            fps_smooth = 0.85 * fps_smooth + 0.15 * fps_instant
            last_tick = now_tick

            # Process frame through ST-GCN pipeline
            t0 = time.perf_counter()
            pred = detector.process_frame(frame)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # Render bounding boxes and skeletons for detected persons
            active_color = (34, 34, 220) if pred.is_fighting else (34, 180, 80)
            for idx, (box, kps) in enumerate(zip(pred.person_boxes, pred.person_keypoints)):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), active_color, 2)
                draw_skeleton(frame, kps, active_color)

                label = f"Person {idx + 1}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    active_color,
                    1,
                    cv2.LINE_AA,
                )

            # Draw visual Relational Strike Distance line between closest actors
            if pred.strike_line is not None and pred.person_count >= 2:
                draw_strike_distance_line(
                    frame, pred.strike_line, pred.min_strike_dist, pred.is_fighting
                )

            # Draw HUD
            draw_hud(
                frame,
                is_fighting=pred.is_fighting,
                probability=pred.probability,
                fps=fps_smooth,
                inference_ms=inference_ms,
                person_count=pred.person_count,
                min_strike_dist=pred.min_strike_dist,
                gate_triggered=pred.relational_gate_triggered,
            )

            if video_writer is not None:
                video_writer.write(frame)

            if gui_available:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    break
            else:
                status_str = "ALERT: VIOLENCE" if pred.is_fighting else "NORMAL"
                status_color_code = "\033[91m" if pred.is_fighting else "\033[92m"
                reset_code = "\033[0m"
                sys.stdout.write(
                    f"\r[LIVE EDGE] FPS: {fps_smooth:4.1f} | Latency: {inference_ms:4.1f}ms | "
                    f"Persons: {pred.person_count} | Status: {status_color_code}{status_str}{reset_code} "
                    f"({pred.probability * 100:4.1f}%)"
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[EDGE-AI] Interrupted by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if gui_available:
            cv2.destroyAllWindows()
        print("\n[EDGE-AI] Video stream closed. Edge pipeline terminated.")


if __name__ == "__main__":
    main()
