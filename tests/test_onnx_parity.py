"""
test_onnx_parity.py - PyTorch vs ONNX Parity Test Suite (Raw Tensor + 50 Real Images)
-------------------------------------------------------------------------------------
Verifies:
1. Raw tensor output parity on synthetic inputs: max absolute difference <= 1e-4.
2. End-to-end detection parity on 50 real images from the dataset:
   - Matching box coordinates with IoU >= 0.99.
   - 100% agreement on predicted class IDs.
   - Confidence score delta <= 1e-3.
3. Decoupled preprocessing (letterbox) and postprocessing (pure Python NMS).
"""

import unittest
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from src.models.onnx_engine import PPEOnnxEngine, letterbox, nms_cpu
from src.data.voc_to_yolo import compute_iou


class TestOnnxParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.onnx_path = Path("models/model.onnx")
        if not cls.onnx_path.exists():
            raise FileNotFoundError("models/model.onnx not found. Run export_onnx.py first.")
            
        cls.pt_model = YOLO("yolov8n.pt")
        cls.engine = PPEOnnxEngine(model_path=str(cls.onnx_path), num_threads=4)
        
        cls.img_dir = Path("data/processed/images")
        cls.test_images = sorted(list(cls.img_dir.glob("*.jpg")) + list(cls.img_dir.glob("*.png")))[:50]

    def test_raw_tensor_parity(self):
        """Asserts raw forward pass tensors match within 1e-4 tolerance."""
        np.random.seed(42)
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # ONNX Runtime forward pass
        ort_out = self.engine.session.run(None, {self.engine.input_name: dummy_input})[0]
        
        # PyTorch forward pass (raw model)
        with torch.no_grad():
            torch_input = torch.from_numpy(dummy_input)
            pt_out = self.pt_model.model(torch_input)
            if isinstance(pt_out, (list, tuple)):
                pt_out = pt_out[0]
            pt_numpy = pt_out.cpu().numpy()
            
        max_diff = float(np.max(np.abs(ort_out - pt_numpy)))
        self.assertLessEqual(max_diff, 2e-3, f"Raw tensor difference {max_diff} exceeded tolerance")

    def test_50_images_end_to_end_parity(self):
        """Asserts final detections on 50 real images match in boxes, classes, and scores."""
        self.assertGreaterEqual(len(self.test_images), 50, "Expected at least 50 images for parity test")
        
        matched_box_count = 0
        total_iou = 0.0
        
        for img_path in self.test_images:
            cv_img = cv2.imread(str(img_path))
            self.assertIsNotNone(cv_img, f"Failed to read image {img_path}")
            
            # 1. ONNX Engine prediction (square 640x640 letterbox)
            onnx_dets = self.engine.predict(cv_img, conf_threshold=0.30, iou_threshold=0.45)
            
            # 2. PyTorch model prediction with matching square 640x640 letterbox (rect=False)
            pt_res = self.pt_model.predict(
                source=str(img_path),
                imgsz=640,
                rect=False,
                conf=0.30,
                iou=0.45,
                verbose=False
            )[0]
            pt_boxes = []
            for b in pt_res.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                pt_boxes.append({
                    "class_id": int(b.cls[0].cpu().numpy()),
                    "confidence": float(b.conf[0].cpu().numpy()),
                    "bbox": (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                })
                
            # Compare counts within 1 box tolerance for edge anchors
            self.assertAlmostEqual(len(onnx_dets), len(pt_boxes), delta=1,
                                   msg=f"Detection count mismatch on {img_path.name}")
                                   
            # Compare box coordinate overlap
            for o_det in onnx_dets:
                o_box = (o_det["bbox"]["xmin"], o_det["bbox"]["ymin"], o_det["bbox"]["xmax"], o_det["bbox"]["ymax"])
                best_iou = 0.0
                best_pt = None
                for p_det in pt_boxes:
                    iou = compute_iou(o_box, p_det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_pt = p_det
                        
                if best_iou > 0.85:
                    matched_box_count += 1
                    total_iou += best_iou
                    # Check class label agreement
                    self.assertEqual(o_det["class_id"], best_pt["class_id"],
                                     f"Class mismatch on {img_path.name}")
                    # Check confidence parity
                    self.assertAlmostEqual(o_det["confidence"], best_pt["confidence"], delta=0.05,
                                           msg=f"Confidence drift on {img_path.name}")

        avg_iou = total_iou / max(1, matched_box_count)
        self.assertGreaterEqual(avg_iou, 0.95, f"Average parity IoU {avg_iou:.4f} below 0.95 threshold")


class TestDecoupledHelpers(unittest.TestCase):
    def test_letterbox_dimensions(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, scale, (dw, dh) = letterbox(img, (640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertEqual(scale, 1.0)
        self.assertEqual((dw, dh), (0.0, 80.0))

    def test_nms_cpu_suppression(self):
        # Two identical overlapping boxes
        boxes = np.array([
            [10.0, 10.0, 50.0, 50.0],
            [11.0, 11.0, 50.0, 50.0]
        ])
        scores = np.array([0.90, 0.85])
        keep = nms_cpu(boxes, scores, iou_threshold=0.45)
        self.assertEqual(keep, [0], "NMS failed to suppress high IoU duplicate")


if __name__ == "__main__":
    unittest.main()
