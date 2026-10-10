"""
app.py - Production FastAPI Service for PPE Detection
------------------------------------------------------
Serves real-time PPE detection via pure ONNX Runtime engine:
1. Lifespan context manager: Single engine initialization + dummy warmup.
2. GET /health: Model metadata, engine status, and uptime metrics.
3. POST /predict: Multipart image upload, size validation, MIME validation,
   safety compliance logic, and microsecond-level timing.
4. OpenAPI 3.0 interactive documentation at /docs and /redoc.
"""

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse, RedirectResponse

from src.models.onnx_engine import PPEOnnxEngine
from src.api.schemas import (
    PredictionResponse,
    HealthResponse,
    Detection,
    BoundingBox,
    ImageDimensions,
    ComplianceStatus
)

# Configuration Constants
DEFAULT_MODEL_PATH = "models/model.onnx"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB strict upload limit
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp"
}

# Global State Container (Managed via Lifespan)
state = {
    "engine": None,
    "start_time": 0.0,
    "model_path": DEFAULT_MODEL_PATH
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager. Loads ONNX model and executes warmup pass
    before serving requests to eliminate first-request cold start latency.
    """
    model_path = Path(state["model_path"])
    if not model_path.exists():
        fallback = Path("models/exp8_best_combo/weights/best.onnx")
        if fallback.exists():
            model_path = fallback
        else:
            print(f"[WARN] Expected model '{model_path}' not found at startup.")
            
    print(f"\n[API LIFESPAN] Initializing PPEOnnxEngine from {model_path}...")
    state["engine"] = PPEOnnxEngine(model_path=str(model_path), num_threads=4)
    state["start_time"] = time.time()
    
    # Warmup inference
    print("[API LIFESPAN] Executing engine warmup inference...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = state["engine"].predict(dummy, conf_threshold=0.40, iou_threshold=0.45)
    print("[API LIFESPAN] PPE Detection API initialized successfully.\n")
    
    yield
    
    print("[API LIFESPAN] Shutting down PPE Detection API...")
    state["engine"] = None


app = FastAPI(
    title="PPE Safety Detection API",
    description="High-performance, decoupled ONNX Runtime inference service for industrial PPE compliance (Safety Helmet vs Bare Head).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Service Health Check",
    description="Returns service status, model path, uptime, and ONNX Runtime execution details."
)
async def health():
    """Health check endpoint for liveness and readiness probes."""
    engine = state.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model engine is not loaded."
        )
        
    uptime = time.time() - state["start_time"]
    return HealthResponse(
        status="healthy",
        model_version="YOLOv8n-PPE-v1.0",
        model_path=str(state["model_path"]),
        runtime_provider="CPUExecutionProvider",
        input_resolution=f"{engine.img_w}x{engine.img_h}",
        uptime_seconds=round(uptime, 2)
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Detect PPE in Image",
    description="Uploads an image, detects PPE items ('hat' vs 'no_helmet'), returns bounding boxes and compliance status."
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, or BMP, max 10MB)"),
    conf_threshold: float = Query(
        0.40,
        ge=0.05,
        le=0.95,
        description="Confidence cutoff threshold. Default 0.40 minimizes safety risk under 5x FN cost matrix."
    ),
    iou_threshold: float = Query(
        0.45,
        ge=0.10,
        le=0.90,
        description="Non-Maximum Suppression (NMS) IoU threshold."
    )
):
    """
    Performs end-to-end PPE detection and site compliance classification on uploaded image.
    """
    # 1. Content-Type Validation  (runs before engine check so 415 fires cleanly in tests)
    content_type = file.content_type or ""
    if content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type}'. Allowed MIME types: {sorted(list(ALLOWED_MIME_TYPES))}"
        )

    # 2. File Size Validation (read full body once; guard before engine so 413 fires cleanly)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Payload too large ({len(contents) / (1024 * 1024):.2f} MB). Maximum allowed size is 10 MB."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Received empty image file."
        )

    # 3. Engine Availability Guard
    engine: Optional[PPEOnnxEngine] = state.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine unavailable."
        )

    # 4. Image Decoding
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unable to decode image file. File may be corrupted or not a valid image."
        )

    orig_h, orig_w = img_bgr.shape[:2]

    # 5. High-Performance Model Inference
    t0 = time.perf_counter()
    raw_detections = engine.predict(
        img_bgr,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    t1 = time.perf_counter()
    processing_time_ms = round((t1 - t0) * 1000.0, 2)

    # 5. Format Detections and Compute Safety Metrics
    detections = []
    count_hat = 0
    count_no_helmet = 0

    for d in raw_detections:
        cls_id = d["class_id"]
        cls_name = d["class_name"]
        if cls_id == 0 or cls_name == "hat":
            count_hat += 1
        elif cls_id == 1 or cls_name == "no_helmet":
            count_no_helmet += 1

        detections.append(
            Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=d["confidence"],
                bbox=BoundingBox(
                    xmin=d["bbox"]["xmin"],
                    ymin=d["bbox"]["ymin"],
                    xmax=d["bbox"]["xmax"],
                    ymax=d["bbox"]["ymax"]
                )
            )
        )

    count_total = count_hat + count_no_helmet

    # Compliance Logic:
    # - If ANY person without a helmet is detected: VIOLATION (Highest Priority)
    # - If at least one person with a helmet and ZERO without helmet: COMPLIANT
    # - If no persons detected: NO_PERSONS
    if count_no_helmet > 0:
        compliance_status = ComplianceStatus.VIOLATION
    elif count_hat > 0:
        compliance_status = ComplianceStatus.COMPLIANT
    else:
        compliance_status = ComplianceStatus.NO_PERSONS

    return PredictionResponse(
        status="success",
        compliance_status=compliance_status,
        count_total=count_total,
        count_hat=count_hat,
        count_no_helmet=count_no_helmet,
        processing_time_ms=processing_time_ms,
        image_dimensions=ImageDimensions(width=orig_w, height=orig_h),
        detections=detections
    )
