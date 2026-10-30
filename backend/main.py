"""FastAPI backend service for batch video analysis and model inference."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure backend directory is in sys.path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from core.inference import (
    ALERT_HOLD_FRAMES,
    CONSECUTIVE_TRIGGERS_REQUIRED,
    DEFAULT_POSE_STRIDE,
    POSE_STRIDE,
    PoseLSTMDetector,
    PoseSTGCNDetector,
)
from logger import IncidentLogger

cv2.setNumThreads(1)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(150 * 1024 * 1024)))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("main")

_classifier: Optional[PoseLSTMDetector] = None
_incident_logger: Optional[IncidentLogger] = None
_analysis_lock = asyncio.Lock()


def resolve_sample_video(sample_name: str) -> Path:
    """Locate a sample video in raw data directories or frontend public samples."""
    clean_name = sample_name.strip().strip("'\"")
    direct = Path(clean_name)
    if direct.is_file():
        return direct.resolve()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    public_samples_dir = project_root / "frontend" / "public" / "samples"

    candidates = [clean_name]
    if not clean_name.lower().endswith((".mp4", ".avi")):
        candidates.append(f"{clean_name}.mp4")

    for name in candidates:
        p_raw = raw_dir / name
        if p_raw.is_file():
            return p_raw.resolve()
        p_nv = raw_dir / "NonViolence" / name
        if p_nv.is_file():
            return p_nv.resolve()
        p_v = raw_dir / "Violence" / name
        if p_v.is_file():
            return p_v.resolve()
        p_pub = public_samples_dir / name
        if p_pub.is_file():
            return p_pub.resolve()

    for search_dir in [raw_dir / "NonViolence", raw_dir / "Violence", public_samples_dir]:
        if search_dir.is_dir():
            for f in search_dir.glob("*.*"):
                if (
                    f.name.lower() == clean_name.lower()
                    or f.stem.lower() == clean_name.lower()
                ):
                    return f.resolve()

    raise FileNotFoundError(f"Sample video '{sample_name}' not found in samples or raw data.")


import core.scene_gating

# Monkey-patch SceneClassifier to prevent loading MobileNetV3 into memory (saving ~100MB RAM for 512MB cloud limits)
class LightweightCloudSceneClassifier:
    """Zero-memory dummy scene classifier for lightweight 512MB RAM cloud deployments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = None

    def analyze_frame(self, frame: Any) -> bool:
        return False

core.scene_gating.SceneClassifier = LightweightCloudSceneClassifier


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _classifier, _incident_logger
    log.info("Initializing Lightweight Cloud Pose + ST-GCN Batch Inference Service (512MB RAM Profile)...")
    _classifier = PoseSTGCNDetector(
        intra_op_threads=1,
        pose_stride=3,
        enable_scene_gating=False,
    )
    _incident_logger = IncidentLogger()
    log.info("Lightweight Cloud Model loaded successfully (threads=1, stride=3, scene_gating=DISABLED). Ready for batch analysis.")
    try:
        yield
    finally:
        log.info("Shutting down inference service...")
        _classifier = None
        _incident_logger = None


app = FastAPI(
    title="AI Video Kinematics & Incident Analysis Service",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "running" if _classifier is not None else "starting",
        "mode": "batch-analysis",
        "violence_threshold": _classifier.threshold if _classifier else None,
        "pose_model": str(_classifier.pose_model_path) if _classifier else None,
        "stgcn_model": str(_classifier.model_path) if _classifier else None,
        "lstm_model": str(_classifier.model_path) if _classifier else None,
        "ort_threads": _classifier.intra_op_threads if _classifier else None,
        "alert_hold_frames": ALERT_HOLD_FRAMES,
        "consecutive_triggers_required": CONSECUTIVE_TRIGGERS_REQUIRED,
    }


@app.get("/api/samples")
async def list_samples() -> dict[str, Any]:
    """Return available sample clips for quick testing."""
    samples = [
        {
            "id": "NV_1.mp4",
            "name": "Normal Activity (NV_1)",
            "category": "Safe",
            "description": "Standard multi-person movement and benign interaction.",
        },
        {
            "id": "V_1.mp4",
            "name": "Physical Altercation (V_1)",
            "category": "Violence",
            "description": "Rapid strike kinematic acceleration and interaction turbulence.",
        },
        {
            "id": "safe_sample.mp4",
            "name": "Embedded Safe Benchmark",
            "category": "Safe",
            "description": "Everyday physical activity benchmark.",
        },
        {
            "id": "violence_sample.mp4",
            "name": "Embedded Violence Incident",
            "category": "Violence",
            "description": "Verified aggressive engagement clip.",
        },
    ]
    return {"samples": samples}


class AnalyzeRequest(BaseModel):
    sample_name: Optional[str] = None


@app.post("/api/analyze_video")
async def analyze_video_endpoint(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    sample_query: Optional[str] = Query(None, alias="sample"),
) -> dict[str, Any]:
    """Process an entire video clip sequentially using Pose + LSTM kinematics.

    Accepts an uploaded .mp4 file or a sample name from data/raw or frontend/public/samples.
    Returns:
    {
        "overall_verdict": "Violence" | "Safe",
        "confidence": 0.89,
        "incident_timestamps": [1.4, 1.6, ...],
        "processing_time_sec": 4.2,
        "frame_count": 103,
        "video_duration_sec": 3.43,
        "video_fps": 30.0,
        "peak_probability": 0.89,
        "timeline": [...]
    }
    """
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Inference model is not loaded yet.")

    target_video_path: Optional[Path] = None
    is_temporary_file = False

    chosen_sample = sample_name or sample_query

    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{suffix}'. Please upload an MP4 video.",
            )

        temp_dir = Path(tempfile.gettempdir()) / "edge-video-batch"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dest = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".mp4", prefix="batch_upload_", dir=temp_dir, delete=False
        )
        temp_path = Path(temp_dest.name)
        try:
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
                    )
                temp_dest.write(chunk)
            temp_dest.flush()
            temp_dest.close()

            if total_bytes == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            target_video_path = temp_path
            is_temporary_file = True
        except Exception:
            temp_dest.close()
            temp_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    elif chosen_sample:
        try:
            target_video_path = resolve_sample_video(chosen_sample)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    else:
        raise HTTPException(
            status_code=400,
            detail="Either a video file upload or a sample_name must be provided.",
        )

    try:
        log.info("Starting sequential batch analysis for: %s", target_video_path)
        # Run inference in a thread pool to avoid blocking the asyncio event loop
        async with _analysis_lock:
            report = await asyncio.to_thread(_classifier.analyze_video, target_video_path)

        log.info(
            "Batch analysis finished for %s: verdict=%s, confidence=%.2f, time=%.2fs",
            target_video_path.name,
            report["overall_verdict"],
            report["confidence"],
            report["processing_time_sec"],
        )

        # Log incident if violence detected
        if report["overall_verdict"] == "Violence" and _incident_logger is not None:
            await asyncio.to_thread(
                _incident_logger.log_batch_incident,
                video_source=target_video_path.name,
                probability=report["confidence"],
                incident_timestamps=report["incident_timestamps"],
                details=f"Batch analysis identified incidents at timestamps: {report['incident_timestamps'][:5]}",
            )

        return report

    except Exception as exc:
        log.exception("Error processing video %s: %s", target_video_path, exc)
        raise HTTPException(
            status_code=500, detail=f"Analysis pipeline error: {str(exc)}"
        )
    finally:
        if is_temporary_file and target_video_path is not None:
            target_video_path.unlink(missing_ok=True)


@app.get("/api/incidents")
async def get_incidents(
    limit: int = Query(default=50, ge=1, le=500),
    since: Optional[float] = Query(default=None),
) -> dict[str, Any]:
    if _incident_logger is None:
        raise HTTPException(status_code=503, detail="Logger is not initialized.")
    records = await asyncio.to_thread(
        _incident_logger.get_recent_incidents, limit, since
    )
    return {"incidents": records, "count": len(records)}


@app.get("/api/stats")
async def get_stats(hours: float = Query(default=24.0, ge=0.1, le=720)) -> dict[str, Any]:
    if _incident_logger is None:
        raise HTTPException(status_code=503, detail="Logger is not initialized.")
    return await asyncio.to_thread(_incident_logger.get_incident_count, hours)


# Serve built Next.js export if present
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
    async def root() -> dict[str, Any]:
        return {
            "service": "AI Video Kinematics & Incident Analysis Service",
            "status": "running",
            "docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=LOG_LEVEL.lower(),
    )
