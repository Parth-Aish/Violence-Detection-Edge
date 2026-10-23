FROM node:20-slim AS frontend-builder
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --no-audit --prefer-offline
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    STREAM_SOURCE=0 \
    FRAME_WIDTH=640 \
    FRAME_HEIGHT=480 \
    FRAME_FPS=30 \
    ONNX_MODEL_PATH=/app/backend/violence_lstm.onnx \
    POSE_MODEL_PATH=yolov8n-pose.pt \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY backend/ /app/backend/
COPY --from=frontend-builder /build/out /app/frontend/out/
RUN test -s /app/backend/violence_lstm.onnx \
    && mkdir -p /app/backend/data \
    && chown -R user:user /app

USER user
WORKDIR /app/backend
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

CMD ["uvicorn", "--app-dir", "/app/backend", "main:app", "--host", "0.0.0.0", "--port", "7860", "--ws-ping-interval", "30", "--ws-ping-timeout", "10"]
