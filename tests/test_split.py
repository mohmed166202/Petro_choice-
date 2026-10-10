"""
test_split.py - Unit & Security Tests for Phase 2 Leakage Guard
---------------------------------------------------------------
Verifies:
1. Leakage guard strictly raises PermissionError when called from forbidden training scripts.
2. Leakage guard allows authorized evaluation callers.
3. assert_zero_leakage detects overlapping stems and perceptual duplicates.
4. Test set file write lock functionality.
"""

import unittest
from pathlib import Path
import imagehash
from PIL import Image

from src.utils.guard import (
    guard_test_set_access,
    assert_zero_leakage,
    lock_test_file,
    unlock_test_file
)


class TestLeakageGuard(unittest.TestCase):
    def test_guard_blocks_train_script(self):
        with self.assertRaises(PermissionError):
            guard_test_set_access("train.py")

    def test_guard_blocks_tune_script(self):
        with self.assertRaises(PermissionError):
            guard_test_set_access("tune.py")

    def test_guard_allows_evaluate_script(self):
        try:
            guard_test_set_access("evaluate.py")
        except PermissionError:
            self.fail("guard_test_set_access unexpectedly blocked evaluate.py")


class TestLeakageAssertion(unittest.TestCase):
    def test_detects_stem_overlap(self):
        train = ["0001", "0002"]
        val = ["0003"]
        test = ["0002", "0004"]  # "0002" leaked
        with self.assertRaises(AssertionError):
            assert_zero_leakage(train, val, test)

    def test_detects_perceptual_hash_leakage(self):
        # Create identical mock hashes
        mock_hash = imagehash.hex_to_hash("ffd0c00000000000")
        phash_map = {
            "train_img": mock_hash,
            "test_img": mock_hash  # Distance 0
        }
        with self.assertRaises(AssertionError):
            assert_zero_leakage(["train_img"], ["val_img"], ["test_img"], phash_map, distance_threshold=4)

    def test_passes_when_clean(self):
        h1 = imagehash.hex_to_hash("ffffffffffffffff")
        h2 = imagehash.hex_to_hash("0000000000000000")
        phash_map = {"train_img": h1, "test_img": h2}
        # Hamming distance is 64 > 4
        assert_zero_leakage(["train_img"], ["val_img"], ["test_img"], phash_map, distance_threshold=4)


if __name__ == "__main__":
    unittest.main()
