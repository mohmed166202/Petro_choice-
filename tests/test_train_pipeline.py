"""
test_train_pipeline.py - Unit & Security Tests for Phase 3 Training Pipeline
----------------------------------------------------------------------------
Tests:
1. dataset.yaml structure and absence of test-set definition.
2. Config loading and parameter parsing for baseline and seed sensitivity.
3. Safety guard enforcement preventing test set leaks during training.
4. Seeding determinism and distinct streams for Seed 42 vs Seed 1337.
"""

import unittest
from pathlib import Path
import yaml
import numpy as np

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.models.train import verify_dataset_config_safety
from src.utils.guard import guard_test_set_access


class TestDatasetConfig(unittest.TestCase):
    def test_dataset_yaml_structure(self):
        cfg_path = Path("data/dataset.yaml")
        self.assertTrue(cfg_path.exists())
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("train", cfg)
        self.assertIn("val", cfg)
        self.assertNotIn("test", cfg, "CRITICAL: 'test' must not be in dataset.yaml to prevent training leaks!")
        self.assertEqual(cfg["names"][0], "hat")
        self.assertEqual(cfg["names"][1], "no_helmet")


class TestBaselineConfigs(unittest.TestCase):
    def test_baseline_seed42_config(self):
        cfg = load_config("configs/exp0_baseline.yaml")
        self.assertEqual(cfg["experiment"]["seed"], 42)
        self.assertEqual(cfg["experiment"]["run_id"], "exp0_baseline_seed42")
        self.assertEqual(cfg["data"]["img_size"], 640)

    def test_baseline_seed1337_config(self):
        cfg = load_config("configs/exp0_baseline_seed1337.yaml")
        self.assertEqual(cfg["experiment"]["seed"], 1337)
        self.assertEqual(cfg["experiment"]["run_id"], "exp0_baseline_seed1337")


class TestSafetyGuard(unittest.TestCase):
    def test_guard_blocks_leaky_dataset_yaml(self):
        # Create a temporary leaky config
        leaky_yaml = Path("tests/temp_leaky_data.yaml")
        try:
            with open(leaky_yaml, "w") as f:
                yaml.dump({"train": "train.txt", "val": "val.txt", "test": "test.txt"}, f)
            with self.assertRaises(PermissionError):
                verify_dataset_config_safety(str(leaky_yaml))
        finally:
            if leaky_yaml.exists():
                leaky_yaml.unlink()

    def test_guard_blocks_train_script(self):
        with self.assertRaises(PermissionError):
            guard_test_set_access("train.py")


class TestSeedDifferentiation(unittest.TestCase):
    def test_seeds_produce_distinct_streams(self):
        set_seed(42)
        stream_42 = np.random.rand(5)
        set_seed(1337)
        stream_1337 = np.random.rand(5)
        self.assertFalse(np.array_equal(stream_42, stream_1337))


if __name__ == "__main__":
    unittest.main()
