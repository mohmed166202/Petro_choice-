"""
oversample.py - Targeted Minority Class Oversampling Pipeline
--------------------------------------------------------------
Generates a balanced training split by oversampling images containing the
minority class ('hat', class 0) at 2x frequency, while leaving the validation set
completely untouched and isolated.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def create_oversampled_dataset(
    train_split_path: Path = Path("data/splits/train.txt"),
    labels_dir: Path = Path("data/processed/labels"),
    output_train_path: Path = Path("data/splits/train_oversampled.txt"),
    output_yaml_path: Path = Path("data/dataset_oversampled.yaml"),
    oversample_factor: int = 2,
    seed: int = 42
) -> Dict[str, int]:
    """
    Creates an oversampled train list and a corresponding dataset YAML config.
    """
    if not train_split_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_split_path}")
        
    with open(train_split_path, "r", encoding="utf-8") as f:
        train_lines = [line.strip() for line in f if line.strip()]
        
    random.seed(seed)
    oversampled_lines = []
    hat_image_count = 0
    no_helmet_only_count = 0
    
    for line in train_lines:
        img_path = Path(line)
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        
        has_hat = False
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as lf:
                for l in lf:
                    parts = l.strip().split()
                    if parts and int(parts[0]) == 0:
                        has_hat = True
                        break
                        
        if has_hat:
            hat_image_count += 1
            # Add oversample_factor copies
            for _ in range(oversample_factor):
                oversampled_lines.append(line)
        else:
            no_helmet_only_count += 1
            oversampled_lines.append(line)
            
    # Deterministic shuffle
    random.shuffle(oversampled_lines)
    
    # Write oversampled train split
    output_train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(oversampled_lines) + "\n")
        
    # Write dataset_oversampled.yaml
    dataset_cfg = {
        "path": ".",
        "train": str(output_train_path.as_posix()),
        "val": "data/splits/val.txt",
        "names": {0: "hat", 1: "no_helmet"}
    }
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)
        
    stats = {
        "original_train_images": len(train_lines),
        "hat_containing_images": hat_image_count,
        "no_helmet_only_images": no_helmet_only_count,
        "oversampled_train_images": len(oversampled_lines),
        "oversample_factor": oversample_factor
    }
    
    print(f"\n[SUCCESS] Oversampled dataset created:")
    print(f"  Original Images:     {stats['original_train_images']}")
    print(f"  Images with 'hat':   {stats['hat_containing_images']} (repeated {oversample_factor}x)")
    print(f"  Oversampled Images:  {stats['oversampled_train_images']}")
    print(f"  Saved YAML to:       {output_yaml_path}")
    return stats


if __name__ == "__main__":
    create_oversampled_dataset()
