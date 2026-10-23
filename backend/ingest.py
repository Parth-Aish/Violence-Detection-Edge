"""Async video-source capture with bounded, low-latency frame buffering."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

cv2.setNumThreads(1)

logger = logging.getLogger("ingest")

MAX_INGEST_FPS = 15 if os.getenv("RENDER") is not None else 30
QUEUE_SIZE = 2
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}


def open_video_source(source: str) -> cv2.VideoCapture:
    """Open a webcam, RTSP source, or local file and fail clearly on errors."""
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        path = Path(source)
        if "://" not in source and path.suffix.lower() in VIDEO_EXTENSIONS:
            if not path.is_file():
                raise FileNotFoundError(f"Video file does not exist: {path}")
            capture = cv2.VideoCapture(str(path))
        elif source.lower().startswith(("rtsp://", "rtsps://")):
            capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    3000,
                ],
            )
        else:
            raise ValueError(
                "Video source must be a webcam index, local video file, or RTSP URL."
            )
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video source: {source}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _read_resized(
    capture: cv2.VideoCapture, width: int, height: int
) -> tuple[bool, Optional[np.ndarray]]:
    success, frame = capture.read()
    if not success or frame is None:
        return False, None
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return True, frame


def _keep_latest_frame(
    frame_queue: asyncio.Queue[np.ndarray], frame: np.ndarray
) -> None:
    try:
        frame_queue.put_nowait(frame)
        return
    except asyncio.QueueFull:
        pass
    try:
        frame_queue.get_nowait()
    except asyncio.QueueEmpty:
        pass
    try:
        frame_queue.put_nowait(frame)
    except asyncio.QueueFull:
        logger.debug("Frame queue filled while replacing a stale frame.")


async def produce_frames(
    capture: cv2.VideoCapture,
    frame_queue: asyncio.Queue[np.ndarray],
    stop_event: asyncio.Event,
    *,
    source: str,
    width: int,
    height: int,
    fps: int = MAX_INGEST_FPS,
    loop_file: bool = False,
) -> None:
    """Pace capture and skip stale local-video frames to stay on wall clock."""
    target_fps = min(max(fps, 1), MAX_INGEST_FPS)
    frame_interval = 1.0 / target_fps
    is_local_file = (
        "://" not in source and Path(source).suffix.lower() in VIDEO_EXTENSIONS
    )
    native_fps_value = await asyncio.to_thread(capture.get, cv2.CAP_PROP_FPS)
    native_fps = float(native_fps_value or 30.0)
    if not np.isfinite(native_fps) or native_fps <= 0:
        native_fps = 30.0
    playback_start = time.perf_counter()
    next_capture_at = playback_start
    current_frame_idx = 0
    source_label = (
        "local video"
        if is_local_file
        else "RTSP stream"
        if source.lower().startswith(("rtsp://", "rtsps://"))
        else "webcam"
    )
    logger.info("Started %s capture at up to %d FPS", source_label, target_fps)

    try:
        while not stop_event.is_set():
            if is_local_file:
                elapsed = time.perf_counter() - playback_start
                target_idx = int(elapsed * native_fps)
                skipped = 0
                while current_frame_idx < target_idx and skipped < 15:
                    grabbed = await asyncio.to_thread(capture.grab)
                    if not grabbed:
                        break
                    current_frame_idx += 1
                    skipped += 1
                if current_frame_idx < target_idx and skipped == 15:
                    await asyncio.sleep(0)
                    continue
                success, frame = await asyncio.to_thread(
                    _read_resized, capture, width, height
                )
                if success:
                    current_frame_idx += 1
            else:
                success, frame = await asyncio.to_thread(
                    _read_resized, capture, width, height
                )
            if not success or frame is None:
                if is_local_file and loop_file:
                    if not await asyncio.to_thread(
                        capture.set, cv2.CAP_PROP_POS_FRAMES, 0
                    ):
                        raise RuntimeError(f"Could not rewind video source: {source}")
                    playback_start = time.perf_counter()
                    current_frame_idx = 0
                    next_capture_at = playback_start
                    continue
                if not stop_event.is_set():
                    logger.info("Video source reached end of stream: %s", source)
                return

            _keep_latest_frame(frame_queue, frame)
            next_capture_at += frame_interval
            remaining = next_capture_at - time.perf_counter()
            if remaining > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Video capture failed.")
        raise
    finally:
        await asyncio.to_thread(capture.release)


def create_frame_queue() -> asyncio.Queue[np.ndarray]:
    return asyncio.Queue(maxsize=QUEUE_SIZE)
