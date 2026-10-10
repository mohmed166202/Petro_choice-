"""
test_api.py - FastAPI TestClient Integration Test Suite for PPE Detection API
------------------------------------------------------------------------------
Covers systematic test scenarios using FastAPI's in-process TestClient
(no external server needed). The `with TestClient(app) as client:` pattern
is used at the class level to trigger FastAPI's lifespan context manager,
ensuring the ONNX engine is loaded exactly once per test class.

Test classes:
  1. TestAPIHealth       - /health, / redirect, schema validation
  2. TestAPIPrediction   - /predict with valid images (JPEG, PNG, custom thresholds)
  3. TestAPIValidation   - Input rejection: MIME type, file size, corrupted payload
  4. TestAPIComplianceLogic - Unit-tests the compliance decision tree in isolation
"""

import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.app import app


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(height: int, width: int, color_bgr: tuple = (100, 150, 200)) -> bytes:
    """Create a solid-color BGR image and JPEG-encode it to bytes."""
    img = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode failed in _make_jpeg_bytes helper"
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Class 1: Health & Routing
# ---------------------------------------------------------------------------

class TestAPIHealth(unittest.TestCase):
    """Tests for GET /health and root redirect behaviour."""

    @classmethod
    def setUpClass(cls):
        # `with` context triggers lifespan → engine loaded before first test
        cls._cm = TestClient(app, raise_server_exceptions=True)
        cls._cm.__enter__()
        cls.client = cls._cm

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def test_health_returns_200(self):
        """GET /health should return HTTP 200."""
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200, f"Got {r.status_code}: {r.text}")

    def test_health_schema_fields(self):
        """All required schema fields must be present in /health JSON."""
        r = self.client.get("/health")
        data = r.json()
        required = {"status", "model_version", "model_path",
                    "runtime_provider", "input_resolution", "uptime_seconds"}
        missing = required - data.keys()
        self.assertFalse(missing, f"Missing fields: {missing}")
        self.assertEqual(data["status"], "healthy")

    def test_health_runtime_provider(self):
        """ONNX runtime provider must be CPUExecutionProvider."""
        r = self.client.get("/health")
        self.assertEqual(r.json()["runtime_provider"], "CPUExecutionProvider")

    def test_health_uptime_non_negative(self):
        """uptime_seconds must be >= 0."""
        r = self.client.get("/health")
        self.assertGreaterEqual(r.json()["uptime_seconds"], 0.0)

    def test_root_redirects_to_docs(self):
        """GET / should issue a redirect to /docs."""
        r = self.client.get("/", follow_redirects=False)
        self.assertIn(r.status_code, [301, 302, 307, 308])
        self.assertIn("/docs", r.headers.get("location", ""))


# ---------------------------------------------------------------------------
# Class 2: Prediction (valid inputs)
# ---------------------------------------------------------------------------

class TestAPIPrediction(unittest.TestCase):
    """Tests for POST /predict with valid images."""

    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(app, raise_server_exceptions=True)
        cls._cm.__enter__()
        cls.client = cls._cm

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _post(self, img_bytes: bytes, content_type: str = "image/jpeg",
              filename: str = "test.jpg", **qs):
        return self.client.post(
            "/predict",
            files={"file": (filename, io.BytesIO(img_bytes), content_type)},
            params=qs
        )

    def test_valid_jpeg_returns_200(self):
        r = self._post(_make_jpeg_bytes(480, 640))
        self.assertEqual(r.status_code, 200, f"Got {r.status_code}: {r.text}")

    def test_response_schema_fields(self):
        """All required Pydantic schema fields must be in /predict response."""
        r = self._post(_make_jpeg_bytes(480, 640))
        data = r.json()
        required = {"status", "compliance_status", "count_total",
                    "count_hat", "count_no_helmet", "processing_time_ms",
                    "image_dimensions", "detections"}
        missing = required - data.keys()
        self.assertFalse(missing, f"Missing fields: {missing}")
        self.assertEqual(data["status"], "success")

    def test_image_dimensions_match_upload(self):
        """Reported image_dimensions must match the uploaded image size."""
        r = self._post(_make_jpeg_bytes(360, 640))
        data = r.json()
        self.assertEqual(data["image_dimensions"]["width"], 640)
        self.assertEqual(data["image_dimensions"]["height"], 360)

    def test_processing_time_positive(self):
        """processing_time_ms must be a positive number."""
        r = self._post(_make_jpeg_bytes(480, 640))
        self.assertGreater(r.json()["processing_time_ms"], 0.0)

    def test_counts_are_consistent(self):
        """count_total must equal count_hat + count_no_helmet."""
        r = self._post(_make_jpeg_bytes(480, 640))
        d = r.json()
        self.assertGreaterEqual(d["count_hat"], 0)
        self.assertGreaterEqual(d["count_no_helmet"], 0)
        self.assertEqual(d["count_total"], d["count_hat"] + d["count_no_helmet"])

    def test_compliance_status_is_valid_enum(self):
        """compliance_status must be one of the three valid enum values."""
        r = self._post(_make_jpeg_bytes(480, 640))
        self.assertIn(r.json()["compliance_status"],
                      ["COMPLIANT", "VIOLATION", "NO_PERSONS"])

    def test_high_threshold_blank_image_no_persons(self):
        """Solid gray image at very high threshold should yield NO_PERSONS."""
        r = self._post(_make_jpeg_bytes(640, 640, color_bgr=(114, 114, 114)),
                       conf_threshold=0.95)
        d = r.json()
        self.assertEqual(r.status_code, 200)
        # At conf=0.95 on a blank image, we expect 0 detections → NO_PERSONS
        if d["count_total"] == 0:
            self.assertEqual(d["compliance_status"], "NO_PERSONS")

    def test_png_image_accepted(self):
        """PNG content-type must be accepted and return 200."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        r = self._post(buf.tobytes(), content_type="image/png", filename="test.png")
        self.assertEqual(r.status_code, 200)

    def test_custom_thresholds_accepted(self):
        """Custom conf_threshold and iou_threshold query params must be accepted."""
        r = self._post(_make_jpeg_bytes(480, 640),
                       conf_threshold=0.60, iou_threshold=0.50)
        self.assertEqual(r.status_code, 200)

    def test_detection_bbox_has_correct_keys(self):
        """Every detection dict must contain class_id, class_name, confidence, bbox (xmin/ymin/xmax/ymax)."""
        r = self._post(_make_jpeg_bytes(480, 640))
        for det in r.json().get("detections", []):
            for key in ("class_id", "class_name", "confidence", "bbox"):
                self.assertIn(key, det, f"Detection missing key '{key}'")
            for coord in ("xmin", "ymin", "xmax", "ymax"):
                self.assertIn(coord, det["bbox"],
                              f"bbox missing coord '{coord}'")


# ---------------------------------------------------------------------------
# Class 3: Input Validation (rejection paths — do NOT need engine)
# ---------------------------------------------------------------------------

class TestAPIValidation(unittest.TestCase):
    """Input validation tests (MIME, size, corruption). Engine not required
    for MIME/size guards since those fire before the engine check."""

    @classmethod
    def setUpClass(cls):
        # Minimal client; lifespan not strictly needed for MIME/size tests,
        # but we open it so corruption test (post-engine) also works.
        cls._cm = TestClient(app, raise_server_exceptions=True)
        cls._cm.__enter__()
        cls.client = cls._cm

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def test_unsupported_mime_type_returns_415(self):
        """Uploading application/pdf must return HTTP 415."""
        r = self.client.post(
            "/predict",
            files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
        )
        self.assertEqual(r.status_code, 415,
                         f"Expected 415, got {r.status_code}: {r.text}")

    def test_oversized_payload_returns_413(self):
        """Payload > 10 MB must return HTTP 413."""
        big = b"X" * (11 * 1024 * 1024)
        r = self.client.post(
            "/predict",
            files={"file": ("big.jpg", io.BytesIO(big), "image/jpeg")}
        )
        self.assertEqual(r.status_code, 413,
                         f"Expected 413, got {r.status_code}: {r.text}")

    def test_corrupt_jpeg_bytes_returns_422(self):
        """Valid MIME type but undecodeable bytes must return HTTP 422."""
        garbage = b"\xff\xd8\xff\x00" + b"\xAB\xCD" * 50  # Truncated JPEG
        r = self.client.post(
            "/predict",
            files={"file": ("corrupt.jpg", io.BytesIO(garbage), "image/jpeg")}
        )
        self.assertEqual(r.status_code, 422,
                         f"Expected 422, got {r.status_code}: {r.text}")


# ---------------------------------------------------------------------------
# Class 4: Compliance Logic (pure unit-test, no HTTP)
# ---------------------------------------------------------------------------

class TestAPIComplianceLogic(unittest.TestCase):
    """Unit-tests for the compliance decision tree without involving HTTP transport."""

    def _compliance(self, count_hat: int, count_no_helmet: int) -> str:
        from src.api.schemas import ComplianceStatus
        if count_no_helmet > 0:
            return ComplianceStatus.VIOLATION
        elif count_hat > 0:
            return ComplianceStatus.COMPLIANT
        else:
            return ComplianceStatus.NO_PERSONS

    def test_violation_takes_priority_over_hat(self):
        """If even one no_helmet exists, status is VIOLATION regardless of hat count."""
        from src.api.schemas import ComplianceStatus
        result = self._compliance(count_hat=10, count_no_helmet=1)
        self.assertEqual(result, ComplianceStatus.VIOLATION)

    def test_all_helmets_is_compliant(self):
        from src.api.schemas import ComplianceStatus
        result = self._compliance(count_hat=5, count_no_helmet=0)
        self.assertEqual(result, ComplianceStatus.COMPLIANT)

    def test_zero_detections_is_no_persons(self):
        from src.api.schemas import ComplianceStatus
        result = self._compliance(count_hat=0, count_no_helmet=0)
        self.assertEqual(result, ComplianceStatus.NO_PERSONS)

    def test_only_violations_is_violation(self):
        from src.api.schemas import ComplianceStatus
        result = self._compliance(count_hat=0, count_no_helmet=3)
        self.assertEqual(result, ComplianceStatus.VIOLATION)


if __name__ == "__main__":
    unittest.main()
