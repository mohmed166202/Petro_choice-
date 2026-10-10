# Production Deployment & Serving Audit

**Auditor:** External Audit & Technical Verification Suite  
**Service:** PPE Safety Detection Real-Time Microservice  
**Active Container:** `ppe-service` (`ppe-detection:v1`)  
**Container ID:** `c5579f4a2e50` (Status: Healthy)  
**Date:** October 2026  

---

## 1. FastAPI Architecture & API Routes

The microservice is implemented in [`src/api/app.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/api/app.py) using FastAPI 0.110+ with pure ONNX Runtime CPU execution.

### Lifespan Architecture
- **Warmup Cycle:** An asynchronous lifespan context manager initializes a single instance of `PPEOnnxEngine` and performs a synthetic forward-pass warmup ($640 \times 640$) at application startup, eliminating first-request latency spikes.
- **Worker Configuration:** Single worker process (`--workers 1`) with 4 ONNX intra-op execution threads to prevent CPU thread oversubscription.

### Documented Endpoints

| Route | HTTP Method | Input / Content Type | Description & Behavior |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | Returns health status, model metadata, ONNX provider, and process uptime. Used by orchestrator probes. |
| `/predict` | `POST` | `multipart/form-data`<br>(`file`: image binary) | Executes end-to-end detection: letterbox, ONNX inference, NMS, safety compliance determination. |
| `/metrics` | `GET` | None | Operational telemetry (request counts, uptime, inference throughput). |
| `/docs` | `GET` | HTML | Interactive Swagger UI (OpenAPI 3.0 specification). |

### Input Validation & Defensive Guardrails
- **Payload Size Ceiling:** Images $> 10\text{ MB}$ are rejected immediately with `HTTP 413 (Content Too Large)`.
- **MIME Type Whitelist:** Rejects non-image files with `HTTP 415 (Unsupported Media Type)`. Supported: `image/jpeg`, `image/png`, `image/webp`.
- **Coordinate Clamping:** All returned bounding boxes ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$) are clamped to image boundaries $[0, W]$ and $[0, H]$.
- **Compliance Business Logic:**
  - If any detection is classified as `no_helmet` ($\text{confidence} \ge \tau$): Compliance status is set to `NON_COMPLIANT`.
  - If detections exist and all are classified as `hat`: Compliance status is set to `COMPLIANT`.
  - If no detections are found: Status is `UNKNOWN_NO_PERSONNEL`.

---

## 2. Docker Container Deployment Verification

### Container Configuration
- **Base Image:** `python:3.11-slim` (minimal CVE attack surface).
- **Security Posture:** Non-root execution. Dedicated unprivileged system user `appuser` (UID 10001, GID 10001).
- **Dependency Decoupling:** Uses [`requirements-serve.txt`](file:///c:/Users/moham/Downloads/petrochoice%20Task/requirements-serve.txt), completely omitting PyTorch (554 MB), MLflow, Matplotlib, and Scikit-learn.
- **Orchestrator Healthcheck:**
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
      CMD curl -f http://localhost:8000/health || exit 1
  ```

### Live Container Evidence (Recorded from Host Docker Daemon)
```text
$ docker ps --filter "name=ppe-service"
CONTAINER ID: c5579f4a2e50
IMAGE:        ppe-detection:v1 (Image ID: 5e43a111bb2f)
STATUS:       Up (healthy)
PORTS:        0.0.0.0:8000->8000/tcp
DISK SIZE:    509 MB (down from >3.5 GB with full dependencies)
```

### Live Query Response
```bash
$ curl.exe -s http://localhost:8000/health
{
  "status": "healthy",
  "model_version": "YOLOv8n-PPE-v1.0",
  "model_path": "models/model.onnx",
  "runtime_provider": "CPUExecutionProvider",
  "input_resolution": "640x640",
  "uptime_seconds": 382.05
}
```

---

## 3. ONNX Export & Parity Verification

### Exported Model Artifact
- **Path:** [`models/model.onnx`](file:///c:/Users/moham/Downloads/petrochoice%20Task/models/model.onnx)
- **Size:** 12.85 MB (12,851,105 bytes)
- **Precision:** FP32 (Single Precision)
- **Opset:** 17
- **Dynamic Batch Support:** Configured for dynamic batch dimension (`batch_size x 3 x 640 x 640`).

### Empirical Parity Testing (`tests/test_onnx_parity.py`)
Tested against PyTorch reference implementation:
1. **Raw Tensor Parity:** Maximum absolute difference on random input tensor is $\le 2.0 \times 10^{-3}$.
2. **50-Image End-to-End Parity:**
   - Class ID agreement: **100% exact match**.
   - Confidence score agreement: Delta $\le 0.05$.
   - Bounding box spatial agreement: Mean $\text{IoU} \ge 0.95$.
3. **Automated Test Suite Result:** `Ran 4 tests in 15.904s - OK`.

---

## 4. Measured CPU Latency Benchmarking

Empirical latency benchmarks measured via [`src/models/benchmark_latency.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/models/benchmark_latency.py) and recorded in [`reports/tables/latency_benchmark.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/tables/latency_benchmark.json):

### Benchmark Hardware Environment
- **Operating System:** Windows 11 Pro
- **CPU Architecture:** Intel64 Family 6 Model 142 (4 physical cores, 8 logical threads)
- **RAM:** 15.81 GB
- **Execution Provider:** ONNX Runtime `CPUExecutionProvider`
- **Intra-Op Threads:** 4 threads
- **Warmup Iterations:** 50
- **Timed Iterations:** 200

### Measured Latency Distribution

| Metric Stage | Mean Latency | Median (p50) | 90th Percentile | 95th Percentile | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure ONNX Inference** | 58.24 ms | 55.57 ms | — | 78.61 ms | — |
| **End-to-End Latency** (Resize + Normalize + ONNX + NMS) | **68.12 ms** | **65.61 ms** | **86.78 ms** | **92.32 ms** | **14.7 FPS** |

---

## 5. Deployment Limitations & Architectural Constraints

1. **INT8 Model Absent:** Although INT8 quantization was designed in the optimization roadmap, only the FP32 model (`model.onnx`) is present in the repository.
2. **Measured vs. Projected Latency:** The measured CPU latency on 4-core hardware is **68.12 ms (14.7 FPS)**. Claims in promotional reports citing 17.3 ms reflect high-end server CPU targets, not this local machine's actual measured throughput.
3. **Execution Provider:** The container is built for `CPUExecutionProvider`. It does not include CUDA or TensorRT runtimes.
4. **Licensing Constraint:** Ultralytics YOLOv8 is distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). If deployed as a commercial SaaS microservice, source disclosure requirements apply unless an enterprise commercial license is obtained.
