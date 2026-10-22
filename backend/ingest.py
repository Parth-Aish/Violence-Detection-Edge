"""
ingest.py — Producer-Consumer RTSP Frame Ingestion Engine
==========================================================

Architecture:
    Producer Process → FFmpeg subprocess (RTSP/file) → raw bytes → decode → Queue
    Consumer (inference) pulls from Queue(maxsize=1) to guarantee zero-lag frames.

Design Decisions:
    1. Queue(maxsize=1): Only the LATEST frame survives. This prevents memory leaks
       when the consumer (inference) is slower than the producer (camera).
    2. FFmpeg over OpenCV: OpenCV's VideoCapture leaks file descriptors on RTSP
       reconnect. FFmpeg subprocess is deterministic — kill PID, respawn.
    3. Graceful degradation: If FFmpeg dies (network blip), the producer auto-restarts
       with exponential backoff (capped at 30s).

Author: Violence Detection System
"""

import subprocess
import multiprocessing
import time
import signal
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ingest")


@dataclass(frozen=True)
class StreamConfig:
    """Immutable stream configuration. Frozen to prevent mutation across processes."""
    source: str                         # RTSP URL or local file path
    width: int = 640                    # Output frame width
    height: int = 480                   # Output frame height
    fps: int = 15                       # Target FPS (decimation)
    pixel_format: str = "bgr24"         # Output pixel format (BGR for OpenCV compat)
    reconnect_delay_init: float = 1.0   # Initial reconnect delay (seconds)
    reconnect_delay_max: float = 30.0   # Maximum reconnect delay cap
    ffmpeg_path: str = "ffmpeg"         # Path to FFmpeg binary
    transport: str = "tcp"              # RTSP transport protocol (tcp/udp)


class FrameProducer:
    """
    Ingests frames from an RTSP stream or video file using FFmpeg,
    pushing only the latest frame into a bounded multiprocessing.Queue.

    This class is designed to run in a separate multiprocessing.Process.
    It handles FFmpeg lifecycle, reconnection with backoff, and graceful shutdown.
    """

    def __init__(self, config: StreamConfig, frame_queue: multiprocessing.Queue,
                 shutdown_event: multiprocessing.Event):
        self.config = config
        self.frame_queue = frame_queue
        self.shutdown_event = shutdown_event
        self._process: Optional[subprocess.Popen] = None
        self._frame_size = config.width * config.height * 3  # BGR24 = 3 bytes/pixel

    def _build_ffmpeg_cmd(self) -> list[str]:
        """
        Constructs the FFmpeg command pipeline.
        
        Key flags:
            -rtsp_transport tcp    → Prevents UDP packet loss on congested networks
            -fflags nobuffer       → Disables input buffering for minimum latency
            -flags low_delay       → Decoder-level low-latency hint
            -an                    → Drops audio (saves bandwidth and CPU)
            -r {fps}               → Output framerate decimation
            -f rawvideo            → Outputs raw uncompressed frames to stdout
        """
        cmd = [self.config.ffmpeg_path]

        # Input options — applied BEFORE -i
        if self.config.source.startswith("rtsp://"):
            cmd.extend([
                "-rtsp_transport", self.config.transport,
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-stimeout", "5000000",       # 5s socket timeout (microseconds)
            ])
        
        cmd.extend([
            "-i", self.config.source,
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
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 2,  # Buffer exactly 2 frames
            )
            logger.info(f"FFmpeg spawned (PID={proc.pid}) for source: {self.config.source}")
            return proc
        except FileNotFoundError:
            logger.error(f"FFmpeg binary not found at: {self.config.ffmpeg_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to spawn FFmpeg: {e}")
            return None

    def _kill_ffmpeg(self):
        """Deterministic FFmpeg teardown. Kill PID, drain pipes, reap zombie."""
        if self._process is None:
            return
        try:
            self._process.kill()
            self._process.stdout.close()
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
        Returns None if the pipe is broken (FFmpeg died or stream ended).
        """
        raw = self._process.stdout.read(self._frame_size)
        if len(raw) != self._frame_size:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.config.height, self.config.width, 3)
        )
        return frame

    def run(self):
        """
        Main producer loop. Designed to be called as the target of
        multiprocessing.Process.

        Pattern:
            1. Spawn FFmpeg
            2. Read frame → put_nowait into Queue (drop old if full)
            3. On failure → kill FFmpeg → backoff → retry
            4. On shutdown_event → clean exit
        """
        backoff = self.config.reconnect_delay_init
        consecutive_failures = 0

        # Ignore SIGINT in child — parent handles orchestration
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        logger.info(f"FrameProducer starting for: {self.config.source}")

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

            while not self.shutdown_event.is_set():
                frame = self._read_frame()
                if frame is None:
                    logger.warning("FFmpeg pipe broken — stream ended or error.")
                    break

                # === QUEUE STRATEGY: DROP-OLD ===
                # If queue is full, discard the stale frame and push the new one.
                # This guarantees the consumer always gets the latest frame.
                try:
                    # Try to drain the stale frame (non-blocking)
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                    self.frame_queue.put_nowait(frame)
                except Exception:
                    # Queue is in a bad state — skip this frame
                    pass

                frames_read += 1
                if frames_read % (self.config.fps * 60) == 0:
                    logger.info(f"Ingested {frames_read} frames from {self.config.source}")

            # === CLEANUP ===
            self._kill_ffmpeg()

            if not self.shutdown_event.is_set():
                sleep_time = min(backoff * (2 ** consecutive_failures),
                                 self.config.reconnect_delay_max)
                logger.warning(f"Stream lost. Reconnecting in {sleep_time:.1f}s...")
                self.shutdown_event.wait(timeout=sleep_time)
                consecutive_failures += 1

        logger.info("FrameProducer shutting down gracefully.")


def create_ingestion_pipeline(
    source: str,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
) -> tuple[multiprocessing.Process, multiprocessing.Queue, multiprocessing.Event]:
    """
    Factory function to create a complete ingestion pipeline.

    Returns:
        (process, frame_queue, shutdown_event)
        
    Usage:
        proc, queue, stop = create_ingestion_pipeline("rtsp://camera:554/stream")
        proc.start()
        frame = queue.get(timeout=5)  # Latest frame as np.ndarray (H, W, 3)
        stop.set()                     # Graceful shutdown
        proc.join(timeout=10)
    """
    config = StreamConfig(source=source, width=width, height=height, fps=fps)
    frame_queue = multiprocessing.Queue(maxsize=1)
    shutdown_event = multiprocessing.Event()
    producer = FrameProducer(config, frame_queue, shutdown_event)

    process = multiprocessing.Process(
        target=producer.run,
        name=f"ingest-{source[:32]}",
        daemon=True,
    )

    return process, frame_queue, shutdown_event
