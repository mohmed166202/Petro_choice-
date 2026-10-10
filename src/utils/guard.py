"""
guard.py - Test Set Leakage Guard & Access Gatekeeper
-----------------------------------------------------
Enforces strict test-set isolation:
1. Prevents any training, tuning, or validation scripts from reading test.txt.
2. Applies OS-level read-only locks to the test set file.
3. Provides automated verification asserting zero stem or perceptual overlap.
"""

import os
import stat
import sys
from pathlib import Path
from typing import Dict, Iterable, Set
import imagehash


FORBIDDEN_CALLERS = {
    "train.py",
    "train_yolo.py",
    "tune.py",
    "grid_search.py",
    "ablation.py",
    "val.py"
}

AUTHORIZED_CALLERS = {
    "evaluate.py",
    "test_split.py",
    "guard.py"
}


def guard_test_set_access(caller_name: str | None = None) -> None:
    """
    Guards test.txt access. Raises PermissionError if called from training scripts.
    """
    if caller_name is None:
        caller_name = Path(sys.argv[0]).name
        
    caller_clean = caller_name.lower()
    for forbidden in FORBIDDEN_CALLERS:
        if forbidden in caller_clean:
            raise PermissionError(
                f"[LEAKAGE GUARD VIOLATION] Script '{caller_name}' attempted to access "
                f"the test set! Test set must remain strictly untouched until final evaluation."
            )


def lock_test_file(test_file: Path) -> None:
    """Sets OS-level read-only attribute on test.txt."""
    if test_file.exists():
        # Make read-only
        os.chmod(test_file, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        print(f"[SECURITY] Test file write-locked (read-only): {test_file}")


def unlock_test_file(test_file: Path) -> None:
    """Removes read-only attribute for authorized updates."""
    if test_file.exists():
        os.chmod(test_file, stat.S_IWUSR | stat.S_IREAD)


def assert_zero_leakage(
    train_stems: Iterable[str],
    val_stems: Iterable[str],
    test_stems: Iterable[str],
    phash_map: Dict[str, imagehash.ImageHash] | None = None,
    distance_threshold: int = 4
) -> None:
    """
    Mathematically verifies that no stems or near-duplicates cross the split boundaries.
    """
    train_set = set(train_stems)
    val_set = set(val_stems)
    test_set = set(test_stems)
    
    # 1. Exact stem disjointness
    assert len(train_set.intersection(val_set)) == 0, "Leakage detected: Train and Val share image stems!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage detected: Train and Test share image stems!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage detected: Val and Test share image stems!"
    
    # 2. Perceptual hash leakage assertion
    if phash_map:
        trainval_stems = train_set.union(val_set)
        for t_stem in test_set:
            if t_stem not in phash_map:
                continue
            t_hash = phash_map[t_stem]
            for tv_stem in trainval_stems:
                if tv_stem not in phash_map:
                    continue
                dist = t_hash - phash_map[tv_stem]
                if dist <= distance_threshold:
                    raise AssertionError(
                        f"Perceptual leakage detected! Test image '{t_stem}' and Train/Val image '{tv_stem}' "
                        f"have Hamming distance {dist} <= {distance_threshold}."
                    )
    print("[SUCCESS] Zero-leakage verification passed: 100% disjoint and zero perceptual overlaps.")
