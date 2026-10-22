"""
ingest.py — Producer-Consumer RTSP & Local Video Frame Ingestion Engine
========================================================================

Architecture:
    Producer Process → FFmpeg subprocess (RTSP or local .mp4) → raw bytes → decode → Queue(1)
    Consumer (inference) pulls from Queue(maxsize=1) to guarantee zero-lag frames.

Design Decisions:
    1. Dual Source Support: Handles both live RTSP streams and local video files (.mp4).
    2. Real-Time Simulation for Local Files: When ingesting a local video file,
       dynamically sleeps for the remaining frame interval to simulate 1x real-time
       playback at the configured FPS. This guarantees that inter-frame
       temporal velocities (dx/dt) calculated by the inference heuristics behave
       identically between live camera streams and pre-recorded test footage.
    3. Queue(maxsize=1): Only the LATEST frame survives. This prevents memory leaks
       when the consumer (inference) is slower than the producer (camera/video).
    4. Deterministic Lifecycle: Subprocess-managed FFmpeg pipeline with SIGINT isolation,
       exponential backoff reconnection for RTSP, and optional seamless looping for local files.

Author: Violence Detection System
"""

import os
import subprocess
import multiprocessing
import time
import signal
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("ingest")

# Common video file extensions recognized as local files
VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv", ".ts"
)


@dataclass(frozen=True)
class StreamConfig:
    """Immutable stream configuration. Frozen to prevent mutation across processes."""
    source: str                         # RTSP URL or local file path
    width: int = 640                    # Output frame width
    height: int = 480                   # Output frame height
    fps: int = 15                       # Target FPS (decimation / pacing)
    pixel_format: str = "bgr24"         # Output pixel format (BGR for OpenCV compat)
    reconnect_delay_init: float = 1.0   # Initial reconnect delay (seconds)
    reconnect_delay_max: float = 30.0   # Maximum reconnect delay cap
    ffmpeg_path: str = "ffmpeg"         # Path to FFmpeg binary
    transport: str = "tcp"              # RTSP transport protocol (tcp/udp)
    loop_file: bool = True              # Automatically loop local video files
    simulate_realtime: bool = True      # Enforce 1x real-time playback pacing for local files


class FrameProducer:
    """
    Ingests frames from an RTSP stream or local video file using FFmpeg,
    pushing only the latest frame into a bounded multiprocessing.Queue(maxsize=1).

    This class runs in a separate multiprocessing.Process.
    It handles FFmpeg lifecycle, 1x real-time playback pacing for local files,
    reconnection with backoff for live streams, and graceful shutdown.
    """

    def __init__(self, config: StreamConfig, frame_queue: multiprocessing.Queue,
                 shutdown_event: multiprocessing.Event):
        self.config = config
        self.frame_queue = frame_queue
        self.shutdown_event = shutdown_event
        self._frame_size = config.width * config.height * 3  # BGR24 = 3 bytes/pixel
        self._is_local = self._detect_local_file(config.source)
        self._is_webcam = config.source.isdigit()

    @staticmethod
    def _detect_local_file(source: str) -> bool:
        """Determines whether the provided source is a local video file."""
        src_lower = source.lower()
        if src_lower.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            return False
        if os.path.isfile(source) or any(src_lower.endswith(ext) for ext in VIDEO_EXTENSIONS):
            return True
        if not source.isdigit() and "://" not in source:
            return True
        return False

    def is_local_file(self) -> bool:
        """Public helper to inspect source type."""
        return self._is_local

    def _get_resolved_source(self) -> str:
        """Returns the normalized or absolute path if local, else original source."""
        if self._is_local:
            return os.path.abspath(self.config.source)
        return self.config.source

    def _build_ffmpeg_cmd(self) -> list[str]:
        """
        Constructs the FFmpeg command pipeline.
        
        Tailors flags based on input type:
        - RTSP: Low-latency network hints (-rtsp_transport tcp, -fflags nobuffer, -flags low_delay, -stimeout)
        - Local File: Optional infinite loop (-stream_loop -1), omitting network socket timeouts.
        """
        cmd = [self.config.ffmpeg_path]
        resolved_source = self._get_resolved_source()

        if self._is_local:
            # Local video file options
            if self.config.loop_file:
                cmd.extend(["-stream_loop", "-1"])
        else:
            # RTSP network stream options
            if resolved_source.lower().startswith(("rtsp://", "rtsps://")):
                cmd.extend([
                    "-rtsp_transport", self.config.transport,
                    "-fflags", "nobuffer",
                    "-flags", "low_delay",
                    "-stimeout", "5000000",       # 5s socket timeout (microseconds)
                ])

        cmd.extend([
            "-i", resolved_source,
            "-an",                             # Drop audio
            "-vf", f"scale={self.config.width}:{self.config.height}",
            "-r", str(self.config.fps),
            "-pix_fmt", self.config.pixel_format,
            "-f", "rawvideo",
            "-loglevel", "error",
            "pipe:1",                          # Output to stdout
        ])
        return cmd

    def _spawn_ffmpeg(self) -> Optional[subprocess.Popen]:
        """Spawns FFmpeg subprocess. Returns None on failure."""
        cmd = self._build_ffmpeg_cmd()
        source_desc = "Local Video File" if self._is_local else "RTSP Stream"
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 2,  # Buffer exactly 2 frames
            )
            logger.info(
                f"FFmpeg spawned (PID={proc.pid}) for {source_desc}: "
                f"{self._get_resolved_source()} @ {self.config.fps} FPS"
            )
            return proc
        except FileNotFoundError:
            logger.error(f"FFmpeg binary not found at: {self.config.ffmpeg_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to spawn FFmpeg: {e}")
            return None

    def _kill_ffmpeg(self):
        """Deterministic FFmpeg teardown. Kill PID, drain pipes, reap process."""
        if self._process is None:
            return
        try:
            self._process.kill()
            if self._process.stdout:
                self._process.stdout.close()
            if self._process.stderr:
                self._process.stderr.close()
            self._process.wait(timeout=5)
            logger.info(f"FFmpeg (PID={self._process.pid}) terminated.")
        except Exception as e:
            logger.warning(f"FFmpeg cleanup error: {e}")
        finally:
            self._process = None

    def _read_frame(self) -> Optional[np.ndarray]:
        """
        Reads exactly one raw frame from FFmpeg stdout.
        Returns None if the pipe is broken (FFmpeg died or file ended).
        """
        if self._process is None or self._process.stdout is None:
            return None
        raw = self._process.stdout.read(self._frame_size)
        if len(raw) != self._frame_size:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.config.height, self.config.width, 3)
        )
        return frame

    def _run_webcam_capture(self):
        """Native OpenCV capture loop for local webcams (0, 1, etc.)."""
        import cv2
        device_id = int(self.config.source)
        logger.info(f"FrameProducer starting native OpenCV capture for webcam index {device_id}")
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        frame_interval = 1.0 / max(1, self.config.fps)
        next_frame_time = time.perf_counter()

        while not self.shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read from webcam {device_id}. Retrying...")
                if self.shutdown_event.wait(timeout=1.0):
                    break
                continue

            if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
                frame = cv2.resize(frame, (self.config.width, self.config.height))

            try:
                self.frame_queue.put_nowait(frame)
            except Exception:
                # Queue is full — discard stale frame and insert newest
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self.frame_queue.put_nowait(frame)
                except Exception:
                    pass

            next_frame_time += frame_interval
            sleep_dur = next_frame_time - time.perf_counter()
            if sleep_dur > 0:
                if self.shutdown_event.wait(timeout=sleep_dur):
                    break
            elif sleep_dur < -0.2:
                next_frame_time = time.perf_counter()

        cap.release()
        logger.info(f"Webcam {device_id} released cleanly.")

    def run(self):
        """
        Main producer loop. Designed to be called as the target of
        multiprocessing.Process.

        Pattern:
            1. If webcam: run native OpenCV capture loop.
            2. If RTSP or local video: spawn FFmpeg subprocess.
            3. Push to Queue(maxsize=1) with drop-old policy.
            4. If local file: pace frames with a dynamic sleep timer for 1x real-time playback.
            5. On failure or EOF: cleanup and restart/reconnect if required.
        """
        # Ignore SIGINT in child process — parent handles orchestration
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self._is_webcam:
            self._run_webcam_capture()
            return

        backoff = self.config.reconnect_delay_init
        consecutive_failures = 0

        source_type = "Local File" if self._is_local else "RTSP Stream"
        logger.info(
            f"FrameProducer starting: {source_type} [{self._get_resolved_source()}]"
        )
        if self._is_local and self.config.simulate_realtime:
            logger.info(
                f"1x Real-time simulation active: sleep pacing enabled "
                f"at {self.config.fps} FPS ({1000.0 / self.config.fps:.1f}ms/frame)"
            )

        while not self.shutdown_event.is_set():
            # === SPAWN ===
            self._process = self._spawn_ffmpeg()
            if self._process is None:
                consecutive_failures += 1
                sleep_time = min(backoff * (2 ** consecutive_failures),
                                 self.config.reconnect_delay_max)
                logger.warning(f"Reconnecting in {sleep_time:.1f}s "
                               f"(attempt #{consecutive_failures})")
                self.shutdown_event.wait(timeout=sleep_time)
                continue

            # === READ LOOP ===
            consecutive_failures = 0
            backoff = self.config.reconnect_delay_init
            frames_read = 0

            # Real-time pacing timer
            frame_interval = 1.0 / max(1, self.config.fps)
            next_frame_time = time.perf_counter()

            while not self.shutdown_event.is_set():
                frame = self._read_frame()
                if frame is None:
                    if self._is_local:
                        logger.info("Local video playback reached EOF.")
                    else:
                        logger.warning("FFmpeg pipe broken — RTSP stream lost or error.")
                    break

                # === QUEUE STRATEGY: DROP-OLD ===
                # If queue is full, discard the stale frame and push the latest one.
                # Guarantees the inference engine always processes the freshest frame.
                try:
                    self.frame_queue.put_nowait(frame)
                except Exception:
                    # Queue is full — discard stale frame and insert newest
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.frame_queue.put_nowait(frame)
                    except Exception:
                        pass

                frames_read += 1
                if frames_read % (self.config.fps * 60) == 0:
                    logger.info(f"Ingested {frames_read} frames from {self.config.source}")

                # === 1x REAL-TIME SIMULATION SLEEP TIMER ===
                # When reading a local file, FFmpeg decodes faster than real-time.
                # Sleep only for the remainder of this frame's target interval so
                # read/queue time is accounted for and the schedule does not drift.
                if self._is_local and self.config.simulate_realtime:
                    next_frame_time += frame_interval
                    sleep_dur = next_frame_time - time.perf_counter()
                    if sleep_dur > 0:
                        time.sleep(sleep_dur)
                    elif sleep_dur < -0.2:
                        # Prevent accumulation of negative deficit if pipeline stalls
                        next_frame_time = time.perf_counter()

            # === CLEANUP ===
            self._kill_ffmpeg()

            if not self.shutdown_event.is_set():
                # For local non-looping files, exit when finished
                if self._is_local and not self.config.loop_file:
                    logger.info("Local video playback complete. Stopping producer.")
                    break

                sleep_time = 0.5 if self._is_local else min(
                    backoff * (2 ** consecutive_failures),
                    self.config.reconnect_delay_max
                )
                if not self._is_local:
                    logger.warning(f"Stream lost. Reconnecting in {sleep_time:.1f}s...")
                self.shutdown_event.wait(timeout=sleep_time)
                consecutive_failures += 1

        logger.info("FrameProducer shutting down gracefully.")


def create_ingestion_pipeline(
    source: str,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    loop_file: bool = True,
    simulate_realtime: bool = True,
) -> tuple[multiprocessing.Process, multiprocessing.Queue, multiprocessing.Event]:
    """
    Factory function to create a complete RTSP or local video ingestion pipeline.

    Args:
        source: RTSP stream URL (e.g. 'rtsp://...') or local video file path (e.g. 'sample.mp4')
        width: Downscaled frame width for inference
        height: Downscaled frame height for inference
        fps: Target frame rate (e.g. 15 or 30 FPS)
        loop_file: Whether local files should automatically loop
        simulate_realtime: Whether to enforce 1x real-time sleep pacing for local files

    Returns:
        (process, frame_queue, shutdown_event)
        
    Usage:
        proc, queue, stop = create_ingestion_pipeline("sample.mp4", fps=30)
        proc.start()
        frame = queue.get(timeout=5)  # Latest frame as np.ndarray (H, W, 3)
        stop.set()                     # Graceful shutdown
        proc.join(timeout=10)
    """
    config = StreamConfig(
        source=source,
        width=width,
        height=height,
        fps=fps,
        loop_file=loop_file,
        simulate_realtime=simulate_realtime,
    )
    frame_queue = multiprocessing.Queue(maxsize=1)
    shutdown_event = multiprocessing.Event()
    producer = FrameProducer(config, frame_queue, shutdown_event)

    process = multiprocessing.Process(
        target=producer.run,
        name=f"ingest-{Path(source).name[:24]}",
        daemon=True,
    )

    return process, frame_queue, shutdown_event
