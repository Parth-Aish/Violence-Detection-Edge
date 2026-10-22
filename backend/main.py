"""
main.py — FastAPI Server: Process Orchestration + WebSocket Streaming
======================================================================

Architecture:
    ┌─────────────┐    Queue(1)    ┌──────────────┐    WS /ws/stream
    │ FrameProducer├──────────────►│ InferenceLoop ├──────────────────► Clients
    │ (subprocess) │               │ (async task)  │    (JPEG + JSON)
    └─────────────┘               └───────┬───────┘
                                          │
                                  ┌───────▼───────┐
                                  │ IncidentLogger │
                                  │ (SQLite+JSONL) │
                                  └───────────────┘

Endpoints:
    GET  /                         → Health check
    GET  /api/incidents            → Recent incidents (JSON)
    GET  /api/stats                → Incident count stats
    WS   /ws/stream                → Live annotated video + detection events

Process Management:
    - FastAPI lifespan manages the ingestion process lifecycle
    - Inference runs in an asyncio background task (not a process)
    - WebSocket broadcasts to all connected clients

Author: Violence Detection System
"""

import asyncio
import base64
import json
import logging
import multiprocessing
import os
import tempfile
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ingest import create_ingestion_pipeline, StreamConfig
from inference import TemporalVelocityClassifier, DetectionResult, Severity
from logger import IncidentLogger

# ──────────────────────────────────────────────
# Configuration via environment variables
# ──────────────────────────────────────────────
STREAM_SOURCE = os.getenv("STREAM_SOURCE", "0")  # RTSP URL, file path, or "0" for webcam
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
FRAME_FPS = int(os.getenv("FRAME_FPS", "15"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n-pose.pt")
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main")

# ──────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────
_frame_queue: Optional[multiprocessing.Queue] = None
_shutdown_event: Optional[multiprocessing.Event] = None
_ingest_process: Optional[multiprocessing.Process] = None
_classifier: Optional[TemporalVelocityClassifier] = None
_incident_logger: Optional[IncidentLogger] = None
_inference_task: Optional[asyncio.Task] = None
_active_upload_path: Optional[Path] = None
_active_source = STREAM_SOURCE
_source_generation = 0
_source_lock = asyncio.Lock()
_classifier_lock = asyncio.Lock()

# WebSocket client registry
_ws_clients: set[WebSocket] = set()
_ws_lock = asyncio.Lock()

# Latest frame for WebSocket broadcast
_latest_frame: Optional[np.ndarray] = None
_latest_detections: list[DetectionResult] = []
_frame_lock = asyncio.Lock()


# ──────────────────────────────────────────────
# WebSocket broadcast helpers
# ──────────────────────────────────────────────
async def _broadcast(message: dict):
    """Send JSON message to all connected WebSocket clients."""
    global _ws_clients
    if not _ws_clients:
        return
    
    payload = json.dumps(message)
    disconnected = set()
    
    async with _ws_lock:
        for ws in set(_ws_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        
        for ws in disconnected:
            _ws_clients.discard(ws)


def _annotate_frame(frame: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
    """
    Draw bounding boxes and severity labels on the frame.
    Color-coded by severity: GREEN(LOW), YELLOW(MED), ORANGE(HIGH), RED(CRIT).
    """
    annotated = frame.copy()
    
    severity_colors = {
        Severity.LOW: (0, 200, 100),       # Green
        Severity.MEDIUM: (0, 220, 255),     # Yellow
        Severity.HIGH: (0, 140, 255),       # Orange
        Severity.CRITICAL: (0, 0, 255),     # Red
    }
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = severity_colors.get(det.severity, (255, 255, 255))
        
        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label background
        label = f"{det.severity.name} | {det.max_velocity:.0f}px/f"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return annotated


def _frame_to_jpeg_base64(frame: np.ndarray) -> str:
    """Encode frame to JPEG and return as base64 string."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buffer).decode("utf-8")


# ──────────────────────────────────────────────
# Inference loop (runs as asyncio background task)
# ──────────────────────────────────────────────
async def _inference_loop():
    """
    Main inference loop:
        1. Pull latest frame from Queue (non-blocking)
        2. Run YOLOv8-Pose + temporal classifier
        3. Log incidents
        4. Broadcast annotated frame + detections via WebSocket
    """
    global _latest_frame, _latest_detections
    
    log.info("Inference loop started.")
    
    loop = asyncio.get_event_loop()
    frame_count = 0
    fps_start = time.time()
    
    while True:
        try:
            frame_queue = _frame_queue
            source_generation = _source_generation
            if frame_queue is None:
                await asyncio.sleep(0.05)
                continue

            # Pull frame from queue (run in executor to avoid blocking event loop)
            try:
                frame = await loop.run_in_executor(
                    None, lambda: frame_queue.get(timeout=1.0)
                )
            except Exception:
                await asyncio.sleep(0.01)
                continue

            if source_generation != _source_generation:
                continue
            
            frame_count += 1
            
            # Run inference in executor (CPU-bound)
            async with _classifier_lock:
                if source_generation != _source_generation:
                    continue
                detections = await loop.run_in_executor(
                    None, _classifier.process_frame, frame
                )

            if source_generation != _source_generation:
                continue

            # Log incidents
            if detections:
                logged = await loop.run_in_executor(
                    None, _incident_logger.log_detections, detections
                )
                
                # Broadcast detection events
                for record in logged:
                    await _broadcast({
                        "type": "incident",
                        "data": record,
                    })
            
            # Annotate and encode frame
            annotated = _annotate_frame(frame, detections)
            jpeg_b64 = await loop.run_in_executor(
                None, _frame_to_jpeg_base64, annotated
            )
            
            # Update latest state
            async with _frame_lock:
                _latest_frame = annotated
                _latest_detections = detections
            
            # Broadcast frame to WebSocket clients
            await _broadcast({
                "type": "frame",
                "data": jpeg_b64,
                "detections": len(detections),
                "is_violence": bool(detections),
                "timestamp": time.time(),
            })
            
            # FPS logging every 5 seconds
            elapsed = time.time() - fps_start
            if elapsed >= 5.0:
                fps = frame_count / elapsed
                log.info(f"Inference FPS: {fps:.1f} (clients: {len(_ws_clients)})")
                frame_count = 0
                fps_start = time.time()
            
            # Yield to event loop
            await asyncio.sleep(0.001)
        except Exception as e:
            log.error(f"Error in inference loop: {e}", exc_info=True)
            await asyncio.sleep(0.05)


# ──────────────────────────────────────────────
# FastAPI lifespan
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of:
        1. Frame ingestion process (multiprocessing)
        2. Inference classifier
        3. Incident logger
        4. Background inference task
    """
    global _frame_queue, _shutdown_event, _ingest_process, _active_source
    global _classifier, _incident_logger, _inference_task
    
    log.info("=" * 60)
    log.info("Violence Detection System — Starting")
    log.info(f"Stream source: {STREAM_SOURCE}")
    log.info(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_FPS}fps")
    log.info(f"Model: {MODEL_PATH}")
    log.info("=" * 60)
    
    # Initialize classifier
    _classifier = TemporalVelocityClassifier(
        window_size=15,
        fps=FRAME_FPS,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        model_path=MODEL_PATH,
    )
    
    # Initialize logger
    _incident_logger = IncidentLogger(cooldown_seconds=2.0)
    
    # Start ingestion process
    source = STREAM_SOURCE
    if source.isdigit():
        demo_path = Path(__file__).parent / "demo_sample.mp4"
        if not sys.platform.startswith("win") and not os.path.exists(f"/dev/video{source}") and demo_path.exists():
            log.info(f"No webcam /dev/video{source} detected on Linux/Docker. Falling back to demo video: {demo_path}")
            source = str(demo_path)
        elif os.getenv("USE_DEMO_STREAM", "0") == "1" and demo_path.exists():
            log.info(f"Using demo stream: {demo_path}")
            source = str(demo_path)
        else:
            log.info(f"Using webcam index: {source}")

    _active_source = source
    
    _ingest_process, _frame_queue, _shutdown_event = create_ingestion_pipeline(
        source=source,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=FRAME_FPS,
    )
    _ingest_process.start()
    log.info(f"Ingestion process started (PID={_ingest_process.pid})")
    
    # Start inference background task
    _inference_task = asyncio.create_task(_inference_loop())
    
    yield  # === App runs here ===
    
    # Shutdown
    log.info("Shutting down...")
    
    # Cancel inference task
    if _inference_task:
        _inference_task.cancel()
        try:
            await _inference_task
        except asyncio.CancelledError:
            pass
    
    # Stop ingestion process
    _stop_ingestion(_ingest_process, _shutdown_event)
    if _active_upload_path:
        _active_upload_path.unlink(missing_ok=True)
    
    # Close WebSocket clients
    async with _ws_lock:
        for ws in _ws_clients:
            try:
                await ws.close()
            except Exception:
                pass
        _ws_clients.clear()
    
    log.info("Violence Detection System — Stopped")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Violence Detection System",
    description="Edge-optimized spatio-temporal violence detection for legacy surveillance.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _stop_ingestion(
    process: Optional[multiprocessing.Process],
    shutdown_event: Optional[multiprocessing.Event],
) -> None:
    """Stop an ingestion worker and reap its process."""
    if shutdown_event:
        shutdown_event.set()
    if process and process.is_alive():
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                raise RuntimeError("Ingestion process did not stop after termination.")


async def _swap_ingestion_source(source: str, *, loop_file: bool, upload_path: Optional[Path] = None) -> None:
    """Start a replacement producer, then atomically make it the active source."""
    global _frame_queue, _shutdown_event, _ingest_process
    global _active_upload_path, _active_source, _source_generation

    async with _source_lock:
        new_process, new_queue, new_shutdown_event = create_ingestion_pipeline(
            source=source,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fps=FRAME_FPS,
            loop_file=loop_file,
            simulate_realtime=True,
        )
        try:
            new_process.start()
        except Exception:
            new_shutdown_event.set()
            raise

        old_process = _ingest_process
        old_shutdown_event = _shutdown_event
        old_upload_path = _active_upload_path

        try:
            _stop_ingestion(old_process, old_shutdown_event)
        except Exception:
            _stop_ingestion(new_process, new_shutdown_event)
            raise

        _ingest_process = new_process
        _frame_queue = new_queue
        _shutdown_event = new_shutdown_event
        _active_upload_path = upload_path
        _active_source = source
        _source_generation += 1

        async with _classifier_lock:
            if _classifier is not None:
                _classifier.reset()
        async with _frame_lock:
            global _latest_frame, _latest_detections
            _latest_frame = None
            _latest_detections = []

        if old_upload_path and old_upload_path != upload_path:
            old_upload_path.unlink(missing_ok=True)

        log.info("Ingestion source switched to %s", source)


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# REST endpoints
# ──────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Health check and system status."""
    return {
        "status": "running",
        "stream_source": _active_source,
        "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "fps": FRAME_FPS,
        "model": MODEL_PATH,
        "ws_clients": len(_ws_clients),
        "ingestion_alive": _ingest_process.is_alive() if _ingest_process else False,
    }


@app.post("/source/webcam")
async def use_webcam():
    """Switch the active source to local webcam index zero."""
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Ingestion is not initialized.")
    try:
        await _swap_ingestion_source("0", loop_file=False)
    except Exception as exc:
        log.exception("Could not switch ingestion to webcam.")
        raise HTTPException(status_code=500, detail="Could not switch to webcam.") from exc
    return {"status": "ok", "source": "webcam"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save an uploaded MP4 and switch ingestion to its real-time playback."""
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Ingestion is not initialized.")
    if Path(file.filename or "").suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Only .mp4 video files are supported.")

    upload_dir = Path(tempfile.gettempdir()) / "violence-detection-uploads"
    upload_path: Optional[Path] = None
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".mp4", prefix="upload-", dir=upload_dir, delete=False
        ) as temporary_file:
            upload_path = Path(temporary_file.name)
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {MAX_UPLOAD_BYTES}-byte upload limit.",
                    )
                temporary_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty.")

        await _swap_ingestion_source(
            str(upload_path), loop_file=False, upload_path=upload_path
        )
        upload_path = None
        return {"status": "ok", "source": "upload"}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not save or start uploaded video.")
        raise HTTPException(status_code=500, detail="Could not start uploaded video.") from exc
    finally:
        await file.close()
        if upload_path:
            upload_path.unlink(missing_ok=True)


@app.get("/api/incidents")
async def get_incidents(
    limit: int = Query(default=50, ge=1, le=500),
    min_severity: int = Query(default=1, ge=0, le=4),
    since: Optional[float] = Query(default=None, description="Unix timestamp"),
):
    """Retrieve recent violence detection incidents."""
    if _incident_logger is None:
        return JSONResponse(status_code=503, content={"error": "Logger not initialized"})
    
    incidents = _incident_logger.get_recent_incidents(
        limit=limit,
        min_severity=min_severity,
        since_timestamp=since,
    )
    return {"incidents": incidents, "count": len(incidents)}


@app.get("/api/stats")
async def get_stats(hours: float = Query(default=24.0, ge=0.1, le=720)):
    """Get incident statistics for the last N hours."""
    if _incident_logger is None:
        return JSONResponse(status_code=503, content={"error": "Logger not initialized"})
    
    stats = _incident_logger.get_incident_count(since_hours=hours)
    return stats


# ──────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Live video + incident stream via WebSocket.
    
    Messages sent to client:
        {"type": "frame", "data": "<base64 JPEG>", "detections": N,
         "is_violence": bool, "timestamp": T}
        {"type": "incident", "data": {incident record}}
    """
    await ws.accept()
    log.info(f"WebSocket client connected. Total: {len(_ws_clients) + 1}")
    
    async with _ws_lock:
        _ws_clients.add(ws)
    
    try:
        # Keep connection alive by reading (client can send ping/config)
        while True:
            try:
                data = await ws.receive_text()
                # Could handle client commands here (e.g., change thresholds)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        async with _ws_lock:
            _ws_clients.discard(ws)
        log.info(f"WebSocket client disconnected. Remaining: {len(_ws_clients)}")


# ──────────────────────────────────────────────
# Static Frontend Mount (Next.js export)
# ──────────────────────────────────────────────
_frontend_candidates = [
    Path(__file__).parent.parent / "frontend" / "out",
    Path(__file__).parent / "static",
    Path("/app/frontend/out"),
    Path("/app/static"),
]
_frontend_dir = next((p for p in _frontend_candidates if p.exists() and p.is_dir()), None)

if _frontend_dir:
    log.info(f"Mounting static Next.js frontend export from: {_frontend_dir}")
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    async def root_fallback():
        return {
            "status": "running",
            "service": "Violence Detection System",
            "stream_source": STREAM_SOURCE,
            "clients": len(_ws_clients),
        }


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    
    # Ensure multiprocessing works on Windows
    multiprocessing.freeze_support()
    
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=LOG_LEVEL.lower(),
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )
