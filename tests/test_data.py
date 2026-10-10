"""
test_data.py - Unit & Verification Tests for Phase 0 and Phase 1
---------------------------------------------------------------
Validates reproducibility seeding, config parsing, and the mathematical
losslessness of VOC-to-YOLO bounding box coordinate transformation.
"""

import unittest
import numpy as np
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.voc_to_yolo import (
    voc_to_yolo_bbox,
    yolo_to_voc_bbox,
    compute_iou,
    round_trip_check,
)


class TestReproducibilityAndSeeding(unittest.TestCase):
    def test_seed_determinism(self):
        set_seed(42)
        arr1 = np.random.rand(10)
        set_seed(42)
        arr2 = np.random.rand(10)
        np.testing.assert_array_equal(arr1, arr2)


class TestConfigLoading(unittest.TestCase):
    def test_load_base_config(self):
        cfg = load_config("configs/base.yaml")
        self.assertIn("experiment", cfg)
        self.assertIn("data", cfg)
        self.assertIn("model", cfg)
        self.assertEqual(cfg["data"]["classes"][0], "hat")
        self.assertEqual(cfg["data"]["classes"][1], "no_helmet")


class TestCoordinateTransformation(unittest.TestCase):
    def test_round_trip_synthetic_boxes(self):
        img_w, img_h = 1920, 1080
        test_boxes = [
            (100.0, 150.0, 300.0, 450.0),    # Normal box
            (10.0, 10.0, 50.0, 50.0),        # Small distant object
            (500.0, 200.0, 1800.0, 1000.0),  # Large foreground object
            (0.0, 0.0, 100.0, 100.0),        # Corner box
        ]
        
        for xmin, ymin, xmax, ymax in test_boxes:
            iou = round_trip_check(xmin, ymin, xmax, ymax, img_w, img_h)
            self.assertGreater(iou, 0.999, f"Round trip IoU was {iou:.5f} for box ({xmin}, {ymin}, {xmax}, {ymax})")

    def test_coordinate_clamping(self):
        img_w, img_h = 640, 480
        # Box extending past image boundary
        xc, yc, bw, bh = voc_to_yolo_bbox(-20.0, -10.0, 700.0, 500.0, img_w, img_h)
        self.assertAlmostEqual(xc, 0.5, places=3)
        self.assertAlmostEqual(yc, 0.5, places=3)
        self.assertAlmostEqual(bw, 1.0, places=3)
        self.assertAlmostEqual(bh, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
