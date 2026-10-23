"""Async FastAPI service for ONNX violence inference and video ingestion."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import (
    CONSECUTIVE_TRIGGERS_REQUIRED,
    PoseLSTMDetector,
    ViolencePrediction,
)
from ingest import (
    MAX_INGEST_FPS,
    create_frame_queue,
    open_video_source,
    produce_frames,
)
from logger import IncidentLogger


cv2.setNumThreads(1)

STREAM_SOURCE = os.getenv("STREAM_SOURCE", "idle")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
FRAME_FPS = max(
    1, min(int(os.getenv("FRAME_FPS", str(MAX_INGEST_FPS))), MAX_INGEST_FPS)
)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "60"))
MAX_STREAM_WIDTH = 480
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("main")

_frame_queue: Optional[asyncio.Queue[np.ndarray]] = None
_producer_stop: Optional[asyncio.Event] = None
_producer_task: Optional[asyncio.Task] = None
_classifier: Optional[PoseLSTMDetector] = None
_incident_logger: Optional[IncidentLogger] = None
_inference_task: Optional[asyncio.Task] = None
_active_upload_path: Optional[Path] = None
_active_source_type = "idle"
_source_generation = 0

_ws_clients: set[WebSocket] = set()
_ws_lock = asyncio.Lock()
_source_lock = asyncio.Lock()


class RTSPSourceRequest(BaseModel):
    source: str = Field(min_length=8, max_length=2048)


def _source_type(source: str, uploaded: bool = False) -> str:
    if uploaded:
        return "upload"
    if source.isdigit():
        return "webcam"
    if source.lower().startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if source.lower() in {"", "idle"}:
        return "idle"
    return "file"


async def _stop_producer() -> None:
    global _producer_task, _producer_stop
    if _producer_stop is not None:
        _producer_stop.set()
    if _producer_task is not None:
        _producer_task.cancel()
        try:
            await _producer_task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Video producer stopped after an error.")
        _producer_task = None
    _producer_stop = None


async def _swap_source(
    source: str,
    *,
    loop_file: bool = False,
    upload_path: Optional[Path] = None,
) -> None:
    global _frame_queue, _producer_stop, _producer_task
    global _active_upload_path, _active_source_type, _source_generation

    async with _source_lock:
        capture = None
        if _source_type(source) != "idle":
            capture = await asyncio.to_thread(open_video_source, source)

        old_upload_path = _active_upload_path
        await _stop_producer()
        _frame_queue = create_frame_queue()
        _active_upload_path = upload_path
        _active_source_type = _source_type(source, uploaded=upload_path is not None)
        _source_generation += 1

        if _classifier is not None:
            _classifier.reset()
        if capture is not None:
            _producer_stop = asyncio.Event()
            _producer_task = asyncio.create_task(
                produce_frames(
                    capture,
                    _frame_queue,
                    _producer_stop,
                    source=source,
                    width=FRAME_WIDTH,
                    height=FRAME_HEIGHT,
                    fps=FRAME_FPS,
                    loop_file=loop_file,
                ),
                name="video-frame-producer",
            )
            _producer_task.add_done_callback(_report_producer_error)

        if old_upload_path is not None and old_upload_path != upload_path:
            old_upload_path.unlink(missing_ok=True)
        log.info("Video source changed to %s", _active_source_type)


def _report_producer_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        log.error(
            "Video producer task failed.",
            exc_info=(type(error), error, error.__traceback__),
        )


async def _broadcast(payload: dict) -> None:
    if not _ws_clients:
        return
    encoded = json.dumps(payload)
    async with _ws_lock:
        for websocket in tuple(_ws_clients):
            try:
                await websocket.send_text(encoded)
            except (WebSocketDisconnect, RuntimeError, OSError):
                _ws_clients.discard(websocket)


def _annotate_frame(frame: np.ndarray, prediction: ViolencePrediction) -> np.ndarray:
    annotated = frame.copy()
    if prediction.bbox is not None:
        x1, y1, x2, y2 = (int(value) for value in prediction.bbox)
        color = (0, 0, 255) if prediction.is_fighting else (40, 180, 90)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{prediction.severity.name} {prediction.probability:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _jpeg_base64(frame: np.ndarray) -> str:
    height, width = frame.shape[:2]
    if width > MAX_STREAM_WIDTH:
        resized_height = max(1, round(height * MAX_STREAM_WIDTH / width))
        frame = cv2.resize(
            frame,
            (MAX_STREAM_WIDTH, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    success, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not success:
        raise RuntimeError("OpenCV failed to encode a WebSocket frame.")
    return base64.b64encode(buffer).decode("ascii")


async def _inference_loop() -> None:
    global _frame_queue
    frame_count = 0
    rate_started = time.monotonic()
    measured_fps = 0.0

    while True:
        frame_queue = _frame_queue
        generation = _source_generation
        if frame_queue is None:
            await asyncio.sleep(0.1)
            continue
        try:
            frame = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if generation != _source_generation or _classifier is None:
            continue

        try:
            prediction = await asyncio.to_thread(_classifier.process_frame, frame)
            if generation != _source_generation:
                continue

            frame_count += 1
            elapsed = time.monotonic() - rate_started
            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                rate_started = time.monotonic()

            if _incident_logger is not None:
                record = await asyncio.to_thread(
                    _incident_logger.log_prediction, prediction
                )
                if record:
                    await _broadcast({"type": "incident", "data": record})

            await _broadcast(
                {
                    "type": "frame",
                    "data": _jpeg_base64(_annotate_frame(frame, prediction)),
                    "is_violence": prediction.is_fighting,
                    "probability": prediction.probability,
                    "severity": prediction.severity.name,
                    "fps": measured_fps or prediction.fps,
                    "inference_ms": prediction.inference_ms,
                    "timestamp": prediction.timestamp,
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Frame inference or broadcast failed.")
            await asyncio.sleep(0.05)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _classifier, _incident_logger, _inference_task, _frame_queue
    global _active_source_type

    log.info(
        "Starting ONNX pose/LSTM detection at %dx%d and at most %d FPS",
        FRAME_WIDTH,
        FRAME_HEIGHT,
        FRAME_FPS,
    )
    _classifier = PoseLSTMDetector()
    _incident_logger = IncidentLogger()
    _frame_queue = create_frame_queue()
    _active_source_type = "idle"
    if _source_type(STREAM_SOURCE) != "idle":
        await _swap_source(STREAM_SOURCE, loop_file=True)
    _inference_task = asyncio.create_task(_inference_loop(), name="frame-inference")

    try:
        yield
    finally:
        if _inference_task is not None:
            _inference_task.cancel()
            try:
                await _inference_task
            except asyncio.CancelledError:
                pass
            _inference_task = None
        await _stop_producer()
        if _active_upload_path is not None:
            _active_upload_path.unlink(missing_ok=True)
        async with _ws_lock:
            for websocket in tuple(_ws_clients):
                try:
                    await websocket.close()
                except (WebSocketDisconnect, RuntimeError, OSError):
                    pass
            _ws_clients.clear()
        _classifier = None
        _incident_logger = None


app = FastAPI(
    title="Spatio-Temporal Violence Detection",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> dict:
    return {
        "status": "running" if _classifier is not None else "starting",
        "source": _active_source_type,
        "fps_target": FRAME_FPS,
        "max_ingest_fps": MAX_INGEST_FPS,
        "violence_threshold": _classifier.threshold if _classifier else None,
        "consecutive_triggers_required": CONSECUTIVE_TRIGGERS_REQUIRED,
        "pose_stride": _classifier.pose_stride if _classifier else None,
        "ort_threads": _classifier.intra_op_threads if _classifier else None,
        "ingestion_alive": bool(_producer_task and not _producer_task.done()),
        "model": str(_classifier.model_path) if _classifier else None,
        "pose_model": str(_classifier.pose_model_path) if _classifier else None,
        "websocket_clients": len(_ws_clients),
    }


@app.post("/source/webcam")
async def use_webcam() -> dict:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Inference model is not loaded.")
    try:
        await _swap_source("0")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "source": "webcam"}


@app.post("/source/rtsp")
async def use_rtsp(request: RTSPSourceRequest) -> dict:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Inference model is not loaded.")
    parsed = urlsplit(request.source)
    if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Provide a valid rtsp:// or rtsps:// URL.")
    try:
        await _swap_source(request.source, loop_file=True)
    except (OSError, ValueError) as exc:
        log.warning("Could not open RTSP input (%s).", type(exc).__name__)
        raise HTTPException(status_code=400, detail="Could not open RTSP source.") from exc
    return {"status": "ok", "source": "rtsp"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Inference model is not loaded.")
    if Path(file.filename or "").suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Only .mp4 videos are supported.")

    directory = Path(tempfile.gettempdir()) / "violence-detection-uploads"
    temporary_path: Optional[Path] = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".mp4", prefix="upload-", dir=directory, delete=False
        ) as destination:
            temporary_path = Path(destination.name)
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes.",
                    )
                destination.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty.")
        await _swap_source(
            str(temporary_path), loop_file=False, upload_path=temporary_path
        )
        temporary_path = None
        return {"status": "ok", "source": "upload"}
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        log.exception("Could not start uploaded MP4.")
        raise HTTPException(status_code=400, detail="Could not open uploaded video.") from exc
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.get("/api/incidents")
async def get_incidents(
    limit: int = Query(default=50, ge=1, le=500),
    since: Optional[float] = Query(default=None),
) -> dict:
    if _incident_logger is None:
        raise HTTPException(status_code=503, detail="Logger is not initialized.")
    records = await asyncio.to_thread(
        _incident_logger.get_recent_incidents, limit, since
    )
    return {"incidents": records, "count": len(records)}


@app.get("/api/stats")
async def get_stats(hours: float = Query(default=24.0, ge=0.1, le=720)) -> dict:
    if _incident_logger is None:
        raise HTTPException(status_code=503, detail="Logger is not initialized.")
    return await asyncio.to_thread(_incident_logger.get_incident_count, hours)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.add(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                if json.loads(message).get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "detail": "Invalid JSON"})
                )
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            _ws_clients.discard(websocket)


frontend_dir = next(
    (
        path
        for path in (
            Path(__file__).parent.parent / "frontend" / "out",
            Path("/app/frontend/out"),
        )
        if path.is_dir()
    ),
    None,
)
if frontend_dir is not None:
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:

    @app.get("/")
    async def root() -> dict:
        return {"service": "Spatio-Temporal Violence Detection", "status": "running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level=LOG_LEVEL.lower(),
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )
