FROM node:20-slim AS frontend-builder
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --no-audit --prefer-offline
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    STREAM_SOURCE=idle \
    FRAME_WIDTH=640 \
    FRAME_HEIGHT=480 \
    FRAME_FPS=15 \
    POSE_STRIDE=2 \
    ONNX_MODEL_PATH=/app/backend/violence_lstm.onnx \
    POSE_MODEL_PATH=/app/backend/yolov8n-pose.onnx \
    ONNX_INTRA_OP_THREADS=1 \
    ORT_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENCV_OPENCL_RUNTIME=disabled \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/backend/
COPY --from=frontend-builder /build/out /app/frontend/out/
RUN test -s /app/backend/violence_lstm.onnx \
    && test -s /app/backend/yolov8n-pose.onnx \
    && mkdir -p /app/backend/data \
    && chown -R user:user /app

USER user
WORKDIR /app/backend

EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen('http://127.0.0.1:%s/api/health' % os.getenv('PORT', '10000'), timeout=3)"

CMD ["sh", "-c", "exec uvicorn --app-dir /app/backend main:app --host 0.0.0.0 --port ${PORT:-10000} --ws-ping-interval 30 --ws-ping-timeout 10"]
