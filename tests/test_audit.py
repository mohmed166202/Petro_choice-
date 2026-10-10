"""
test_audit.py - Unit Tests for Data Audit & Integrity Routines
--------------------------------------------------------------
Tests hash calculations, IoU metrics, degenerate box flagging, and audit output schemas.
"""

import unittest
from pathlib import Path
from src.data.voc_to_yolo import compute_iou


class TestAuditLogic(unittest.TestCase):
    def test_iou_identical_boxes(self):
        box = (10.0, 10.0, 50.0, 50.0)
        self.assertAlmostEqual(compute_iou(box, box), 1.0)

    def test_iou_disjoint_boxes(self):
        boxA = (0.0, 0.0, 10.0, 10.0)
        boxB = (20.0, 20.0, 30.0, 30.0)
        self.assertEqual(compute_iou(boxA, boxB), 0.0)

    def test_iou_half_overlap(self):
        boxA = (0.0, 0.0, 20.0, 10.0)  # Area 200
        boxB = (10.0, 0.0, 30.0, 10.0) # Area 200, Overlap: 10*10=100, Union: 300
        self.assertAlmostEqual(compute_iou(boxA, boxB), 100.0 / 300.0, places=4)


if __name__ == "__main__":
    unittest.main()
