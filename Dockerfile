# ════════════════════════════════════════════════════════════════════
# Violence Detection System — Production Dockerfile
# Optimized for Hugging Face Spaces (Docker SDK, Port 7860)
# ════════════════════════════════════════════════════════════════════

# ── Stage 1: Build Next.js Static Export ──
FROM node:20-slim AS frontend-builder
WORKDIR /build

# Copy package manifests and install dependencies
COPY frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit

# Copy frontend source and build static export (HTML/CSS/JS in /build/out)
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Production Python & ML Runtime ──
FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    STREAM_SOURCE="0" \
    FRAME_WIDTH=640 \
    FRAME_HEIGHT=480 \
    FRAME_FPS=15 \
    MODEL_PATH="yolov8n-pose.pt" \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (FFmpeg for ingestion, libgl for OpenCV, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces security specification: create non-root user with UID 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install Python dependencies as user
COPY requirements.txt .
RUN chown -R user:user /app
USER user

RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-cache YOLOv8n-pose weights during build to ensure instant container startup
WORKDIR /app/backend
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

# Copy application source code
USER root
COPY --chown=user:user backend/ /app/backend/
COPY --chown=user:user --from=frontend-builder /build/out /app/frontend/out/

# Create runtime data directory with proper write permissions
RUN mkdir -p /app/backend/data && chown -R user:user /app

USER user

# Expose required Hugging Face Spaces port
EXPOSE 7860

# Healthcheck monitoring the FastAPI stats endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Launch the unified FastAPI + WebSocket server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--ws-ping-interval", "30", "--ws-ping-timeout", "10"]
