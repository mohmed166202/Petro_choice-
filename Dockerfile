# ==============================================================================
# Production Dockerfile - PPE Safety Detection Microservice
# ==============================================================================
# - Base: python:3.11-slim for minimal security surface
# - Multi-threaded ONNX Runtime CPU inference
# - Principle of least privilege: non-root user (appuser, UID 10001)
# - Container healthcheck probe configured for orchestrators
# ==============================================================================

FROM python:3.11-slim

# Build arguments & Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install minimal OS dependencies for healthchecks & image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install lightweight inference dependencies (pure ONNX Runtime, no PyTorch/MLflow bloat)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy source code and exported ONNX model
COPY src/ ./src/
COPY models/model.onnx ./models/model.onnx

# Ensure permissions are owned by unprivileged appuser
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose internal service port
EXPOSE 8000

# Orchestrator Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with single worker for deterministic CPU threading
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
