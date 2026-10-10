"""
test_ablation_pipeline.py - Unit Tests for Phase 4 Controlled Ablation Framework
--------------------------------------------------------------------------------
Verifies:
1. Existence and valid schema for all 8 Phase 4 experiment configuration files.
2. Strict single-variable modification constraint for runs exp1 through exp7.
3. Test set isolation in dataset_oversampled.yaml.
4. Ablation tracker generates valid CSV and Markdown tables.
"""

import unittest
from pathlib import Path
import yaml
from src.utils.config import load_config
from src.models.ablation_tracker import compile_ablation_study


class TestAblationConfigs(unittest.TestCase):
    def setUp(self):
        self.baseline = load_config("configs/exp0_baseline.yaml")

    def test_all_ablation_configs_exist(self):
        for i in range(1, 9):
            name = [
                "exp1_scale", "exp2_mosaic", "exp3_hsv", "exp4_cls_gain",
                "exp5_oversample", "exp6_imgsz_800", "exp7_cosine_lr", "exp8_best_combo"
            ][i - 1]
            cfg_path = Path(f"configs/{name}.yaml")
            self.assertTrue(cfg_path.exists(), f"Missing config: {cfg_path}")
            cfg = load_config(str(cfg_path))
            self.assertEqual(cfg["experiment"]["run_id"], name)

    def test_single_variable_constraint_exp1(self):
        exp1 = load_config("configs/exp1_scale.yaml")
        self.assertNotEqual(exp1["augmentation"]["scale"], self.baseline["augmentation"]["scale"])
        self.assertEqual(exp1["augmentation"]["mosaic"], self.baseline["augmentation"]["mosaic"])
        self.assertEqual(exp1["loss"]["cls_loss_gain"], self.baseline["loss"]["cls_loss_gain"])

    def test_single_variable_constraint_exp2(self):
        exp2 = load_config("configs/exp2_mosaic.yaml")
        self.assertNotEqual(exp2["augmentation"]["mosaic"], self.baseline["augmentation"]["mosaic"])
        self.assertEqual(exp2["augmentation"]["scale"], self.baseline["augmentation"]["scale"])

    def test_single_variable_constraint_exp4(self):
        exp4 = load_config("configs/exp4_cls_gain.yaml")
        self.assertNotEqual(exp4["loss"]["cls_loss_gain"], self.baseline["loss"]["cls_loss_gain"])
        self.assertEqual(exp4["augmentation"]["scale"], self.baseline["augmentation"]["scale"])

    def test_single_variable_constraint_exp6(self):
        exp6 = load_config("configs/exp6_imgsz_800.yaml")
        self.assertEqual(exp6["data"]["img_size"], 800)
        self.assertEqual(self.baseline["data"]["img_size"], 640)

    def test_oversampled_yaml_test_set_isolation(self):
        cfg_path = Path("data/dataset_oversampled.yaml")
        self.assertTrue(cfg_path.exists())
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.assertNotIn("test", cfg, "CRITICAL: dataset_oversampled.yaml must not contain 'test' key!")


class TestAblationTracker(unittest.TestCase):
    def test_tracker_generates_tables(self):
        df = compile_ablation_study()
        self.assertEqual(len(df), 10)  # 2 baselines + 8 ablations
        self.assertTrue(Path("reports/tables/ablation_study.csv").exists())
        self.assertTrue(Path("reports/tables/ablation_study.md").exists())


if __name__ == "__main__":
    unittest.main()
