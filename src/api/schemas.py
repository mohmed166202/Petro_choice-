"""
schemas.py - Pydantic v2 Data Transfer Objects for PPE Detection API
--------------------------------------------------------------------
Defines strict input/output models for FastAPI endpoints:
1. BoundingBox: normalized and absolute coordinate representations.
2. Detection: individual predicted PPE instance (hat vs no_helmet).
3. PredictionResponse: full inference result with compliance classification.
4. HealthResponse: runtime health and environment status.
"""

from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"       # Persons detected, all wearing helmets
    VIOLATION = "VIOLATION"       # At least one person without helmet detected
    NO_PERSONS = "NO_PERSONS"     # No persons or helmets detected in frame


class BoundingBox(BaseModel):
    """Bounding box coordinates in original image pixel space [xmin, ymin, xmax, ymax]."""
    xmin: float = Field(..., description="Top-left horizontal coordinate (pixels)", ge=0.0)
    ymin: float = Field(..., description="Top-left vertical coordinate (pixels)", ge=0.0)
    xmax: float = Field(..., description="Bottom-right horizontal coordinate (pixels)", ge=0.0)
    ymax: float = Field(..., description="Bottom-right vertical coordinate (pixels)", ge=0.0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "xmin": 124.5,
                "ymin": 45.0,
                "xmax": 210.2,
                "ymax": 180.8
            }
        }
    )


class Detection(BaseModel):
    """Individual PPE detection instance."""
    class_id: int = Field(..., description="Class identifier (0: hat, 1: no_helmet)")
    class_name: str = Field(..., description="Human-readable class name ('hat' or 'no_helmet')")
    confidence: float = Field(..., description="Model confidence score in [0.0, 1.0]", ge=0.0, le=1.0)
    bbox: BoundingBox = Field(..., description="Detected bounding box in pixel coordinates")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "class_id": 0,
                "class_name": "hat",
                "confidence": 0.8924,
                "bbox": {
                    "xmin": 124.5,
                    "ymin": 45.0,
                    "xmax": 210.2,
                    "ymax": 180.8
                }
            }
        }
    )


class ImageDimensions(BaseModel):
    """Source image dimensions."""
    width: int = Field(..., description="Original image width in pixels", gt=0)
    height: int = Field(..., description="Original image height in pixels", gt=0)


class PredictionResponse(BaseModel):
    """Complete prediction response schema returned by /predict endpoint."""
    status: str = Field(default="success", description="Request status flag")
    compliance_status: ComplianceStatus = Field(
        ...,
        description="Site safety determination: COMPLIANT, VIOLATION, or NO_PERSONS"
    )
    count_total: int = Field(..., description="Total detections in frame", ge=0)
    count_hat: int = Field(..., description="Count of compliant helmets ('hat')", ge=0)
    count_no_helmet: int = Field(..., description="Count of violations ('no_helmet')", ge=0)
    processing_time_ms: float = Field(..., description="Total inference pipeline processing time (ms)")
    image_dimensions: ImageDimensions = Field(..., description="Dimensions of uploaded image")
    detections: List[Detection] = Field(default_factory=list, description="List of detected objects")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "compliance_status": "VIOLATION",
                "count_total": 3,
                "count_hat": 2,
                "count_no_helmet": 1,
                "processing_time_ms": 68.42,
                "image_dimensions": {"width": 1920, "height": 1080},
                "detections": [
                    {
                        "class_id": 1,
                        "class_name": "no_helmet",
                        "confidence": 0.824,
                        "bbox": {"xmin": 500.0, "ymin": 200.0, "xmax": 620.0, "ymax": 380.0}
                    }
                ]
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check diagnostic response schema."""
    status: str = Field(default="healthy", description="Service health indicator")
    model_version: str = Field(..., description="Model version tag or architecture")
    model_path: str = Field(..., description="Path to loaded ONNX weights")
    runtime_provider: str = Field(default="CPUExecutionProvider", description="ONNX Runtime execution provider")
    input_resolution: str = Field(default="640x640", description="Model input image resolution")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_version": "YOLOv8n-PPE-v1.0",
                "model_path": "models/model.onnx",
                "runtime_provider": "CPUExecutionProvider",
                "input_resolution": "640x640",
                "uptime_seconds": 128.4
            }
        }
    )
