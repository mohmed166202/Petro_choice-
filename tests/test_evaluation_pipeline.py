"""
test_evaluation_pipeline.py - Unit Tests for Phase 5 Evaluation and Failure Analysis
------------------------------------------------------------------------------------
Tests:
1. Optimal threshold justification file structure and valid range [0, 1].
2. Test set isolation guard permits evaluate.py.
3. Failure analysis generates at least 15 failure diagnostic images and markdown table.
4. Failure taxonomy definitions and engineering remediation rankings.
"""

import unittest
import json
from pathlib import Path
import pandas as pd

from src.utils.guard import guard_test_set_access
from src.models.threshold_optimizer import load_optimal_threshold
from src.models.failure_analysis import FAILURE_TAXONOMY


class TestThresholdOptimization(unittest.TestCase):
    def test_threshold_file_validity(self):
        reports_dir = Path("reports")
        json_path = reports_dir / "optimal_threshold_justification.json"
        self.assertTrue(json_path.exists())
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("optimal_threshold", data)
        self.assertGreaterEqual(data["optimal_threshold"], 0.05)
        self.assertLessEqual(data["optimal_threshold"], 0.95)
        self.assertIn("cost_matrix_parameters", data)

    def test_load_optimal_threshold_helper(self):
        tau = load_optimal_threshold(Path("reports"))
        self.assertIsInstance(tau, float)
        self.assertTrue(0.0 <= tau <= 1.0)


class TestEvaluationAccessGuard(unittest.TestCase):
    def test_guard_permits_evaluate_caller(self):
        try:
            guard_test_set_access("evaluate.py")
        except PermissionError:
            self.fail("guard_test_set_access unexpectedly blocked evaluate.py")


class TestFailureAnalysis(unittest.TestCase):
    def test_failure_images_count(self):
        failures_dir = Path("reports/failures")
        self.assertTrue(failures_dir.exists())
        imgs = list(failures_dir.glob("failure_*.jpg"))
        self.assertGreaterEqual(len(imgs), 15, f"Expected >= 15 failure images, found {len(imgs)}")

    def test_failure_tables_exist(self):
        csv_file = Path("reports/tables/failure_analysis.csv")
        md_file = Path("reports/tables/failure_analysis.md")
        self.assertTrue(csv_file.exists())
        self.assertTrue(md_file.exists())
        df = pd.read_csv(csv_file)
        self.assertGreaterEqual(len(df), 15)
        self.assertIn("Root Cause", df.columns)
        self.assertIn("Proposed Fix", df.columns)
        self.assertIn("Impact Rank", df.columns)

    def test_failure_taxonomy_coverage(self):
        required_keys = {"small_distant", "occlusion", "lighting_glare", "crowd_density", "headwear_ambiguity", "label_noise"}
        self.assertTrue(required_keys.issubset(set(FAILURE_TAXONOMY.keys())))


if __name__ == "__main__":
    unittest.main()
