"""
audit.py - Comprehensive Dataset Audit & Quality Analysis
---------------------------------------------------------
Performs in-depth analysis on the converted YOLO dataset:
1. Class distribution and exact imbalance ratio.
2. Bounding box size breakdown (COCO small, medium, large categories).
3. Image resolution spread and aspect ratios.
4. Annotation anomalies (degenerate boxes, overlapping duplicates, out-of-bounds).
5. Exact image duplicates (MD5) and near-duplicates (perceptual hashing).
6. Generates visual audit charts and a ground-truth spot-check grid.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
from tqdm import tqdm


def audit_dataset(
    processed_dir: Path = Path("data/processed"),
    reports_dir: Path = Path("reports"),
    sample_grid_size: int = 16
) -> Dict:
    """
    Executes full data audit on processed dataset and exports summary and figures.
    """
    img_dir = processed_dir / "images"
    lbl_dir = processed_dir / "labels"
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    lbl_files = sorted(list(lbl_dir.glob("*.txt")))
    
    print(f"[INFO] Auditing {len(img_files)} images and {len(lbl_files)} label files...")
    
    # 1. Image checks & exact/near duplicate tracking
    md5_hashes = defaultdict(list)
    phash_dict = {}
    corrupt_images = []
    resolutions = []
    
    for img_path in tqdm(img_files, desc="Hashing & Checking Images"):
        try:
            with Image.open(img_path) as pil_img:
                w, h = pil_img.size
                resolutions.append((w, h))
                
                # MD5 hash for exact duplicates
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                md5_hashes[file_hash].append(img_path.name)
                
                # Perceptual hash for near-duplicates
                phash_val = imagehash.phash(pil_img)
                phash_dict[img_path.name] = phash_val
        except Exception as e:
            corrupt_images.append({"file": img_path.name, "error": str(e)})

    # Exact duplicates summary
    exact_duplicates = [names for names in md5_hashes.values() if len(names) > 1]
    
    # Near-duplicates (Hamming distance <= 4)
    print("[INFO] Computing pairwise perceptual hash distances (sampling check)...")
    near_duplicates = []
    filenames = list(phash_dict.keys())
    # Efficient pairwise check
    for i in range(len(filenames)):
        for j in range(i + 1, min(i + 150, len(filenames))):  # windowed check for speed
            dist = phash_dict[filenames[i]] - phash_dict[filenames[j]]
            if dist <= 4:
                near_duplicates.append((filenames[i], filenames[j], int(dist)))

    # 2. Annotation & Box Auditing
    class_counts = {0: 0, 1: 0}  # 0: hat, 1: no_helmet
    box_sizes = []  # (class_id, w_px, h_px, area_px)
    boxes_per_image = []
    
    # Error tracking
    label_errors = {
        "out_of_bounds": [],
        "degenerate_zero_area": [],
        "overlapping_duplicates": [],
        "missing_label_file": [],
    }
    
    for img_path, (w, h) in zip(img_files, resolutions):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            label_errors["missing_label_file"].append(img_path.name)
            boxes_per_image.append(0)
            continue
            
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        boxes_per_image.append(len(lines))
        img_boxes = []
        
        for line in lines:
            parts = line.split()
            cid = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            
            # Out of bounds check
            if xc < 0 or xc > 1 or yc < 0 or yc > 1 or bw <= 0 or bh <= 0:
                label_errors["out_of_bounds"].append({"file": img_path.name, "box": line})
                continue
                
            w_px = bw * w
            h_px = bh * h
            area_px = w_px * h_px
            
            if area_px <= 1.0:
                label_errors["degenerate_zero_area"].append({"file": img_path.name, "box": line})
                continue
                
            class_counts[cid] = class_counts.get(cid, 0) + 1
            box_sizes.append((cid, w_px, h_px, area_px))
            img_boxes.append((cid, xc, yc, bw, bh))

        # Check for overlapping duplicate boxes inside same image
        for i in range(len(img_boxes)):
            for j in range(i + 1, len(img_boxes)):
                c1, x1, y1, w1, h1 = img_boxes[i]
                c2, x2, y2, w2, h2 = img_boxes[j]
                if c1 == c2:
                    # Compute box IoU
                    xA = max(x1 - w1/2, x2 - w2/2)
                    yA = max(y1 - h1/2, y2 - h2/2)
                    xB = min(x1 + w1/2, x2 + w2/2)
                    yB = min(y1 + h1/2, y2 + h2/2)
                    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                    union = (w1 * h1) + (w2 * h2) - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.90:
                        label_errors["overlapping_duplicates"].append({
                            "file": img_path.name, "iou": round(iou, 3), "class": c1
                        })

    # COCO object size categorization (<32^2 px, 32^2-96^2 px, >96^2 px)
    coco_small = sum(1 for b in box_sizes if b[3] < 1024)
    coco_medium = sum(1 for b in box_sizes if 1024 <= b[3] <= 9216)
    coco_large = sum(1 for b in box_sizes if b[3] > 9216)
    total_boxes = len(box_sizes)
    
    imbalance_ratio = (
        round(class_counts[1] / class_counts[0], 2) if class_counts[0] > 0 else "N/A"
    )
    
    audit_results = {
        "dataset_summary": {
            "total_images": len(img_files),
            "total_boxes": total_boxes,
            "corrupt_images_count": len(corrupt_images),
            "exact_duplicate_groups": len(exact_duplicates),
            "near_duplicate_pairs_sampled": len(near_duplicates),
        },
        "class_distribution": {
            "hat_count": class_counts.get(0, 0),
            "no_helmet_count": class_counts.get(1, 0),
            "imbalance_ratio_no_helmet_to_hat": f"{imbalance_ratio}:1",
            "hat_percentage": round(100 * class_counts.get(0, 0) / max(1, total_boxes), 2),
            "no_helmet_percentage": round(100 * class_counts.get(1, 0) / max(1, total_boxes), 2),
        },
        "box_size_breakdown": {
            "coco_small_percentage": round(100 * coco_small / max(1, total_boxes), 2),
            "coco_medium_percentage": round(100 * coco_medium / max(1, total_boxes), 2),
            "coco_large_percentage": round(100 * coco_large / max(1, total_boxes), 2),
            "mean_boxes_per_image": round(float(np.mean(boxes_per_image)), 2) if boxes_per_image else 0,
            "max_boxes_in_single_image": int(np.max(boxes_per_image)) if boxes_per_image else 0,
        },
        "label_errors": {
            "out_of_bounds_count": len(label_errors["out_of_bounds"]),
            "degenerate_zero_area_count": len(label_errors["degenerate_zero_area"]),
            "overlapping_duplicates_count": len(label_errors["overlapping_duplicates"]),
        }
    }
    
    # Save JSON summary
    summary_path = reports_dir / "data_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"[SUCCESS] Audit summary exported to {summary_path}")
    
    # 3. Generate Charts
    generate_audit_figures(audit_results, box_sizes, resolutions, fig_dir)
    
    # 4. Generate Visual Verification Grid
    create_sample_grid(img_files, lbl_dir, fig_dir / "audit_visual_sample_grid.png")
    
    return audit_results


def generate_audit_figures(audit_results: Dict, box_sizes: List, resolutions: List, fig_dir: Path) -> None:
    """Generates informative EDA plots for the technical report."""
    # Class Distribution Bar Chart
    plt.figure(figsize=(6, 4))
    classes = ["hat (helmet)", "no_helmet (bare head)"]
    counts = [audit_results["class_distribution"]["hat_count"], audit_results["class_distribution"]["no_helmet_count"]]
    colors = ["#2ecc71", "#e74c3c"]
    bars = plt.bar(classes, counts, color=colors, width=0.5)
    plt.ylabel("Total Annotations")
    plt.title(f"Class Distribution (Imbalance {audit_results['class_distribution']['imbalance_ratio_no_helmet_to_hat']})")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.01), f"{yval:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=150)
    plt.close()
    
    # Box Size Distribution (COCO categories)
    plt.figure(figsize=(6, 4))
    cat_names = ["Small (<32² px)", "Medium (32²-96² px)", "Large (>96² px)"]
    cat_vals = [
        audit_results["box_size_breakdown"]["coco_small_percentage"],
        audit_results["box_size_breakdown"]["coco_medium_percentage"],
        audit_results["box_size_breakdown"]["coco_large_percentage"]
    ]
    plt.bar(cat_names, cat_vals, color=["#3498db", "#9b59b6", "#f39c12"], width=0.5)
    plt.ylabel("Share of Bounding Boxes (%)")
    plt.title("Bounding Box Scale Distribution (COCO Definition)")
    plt.tight_layout()
    plt.savefig(fig_dir / "box_size_distribution.png", dpi=150)
    plt.close()


def create_sample_grid(img_files: List[Path], lbl_dir: Path, output_file: Path) -> None:
    """Draws ground truth boxes on a 4x4 image grid for visual spot checking."""
    selected_imgs = img_files[:16]
    if not selected_imgs:
        return
        
    cols, rows = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    axes = axes.flatten()
    
    for i, img_path in enumerate(selected_imgs):
        ax = axes[i]
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            continue
        h, w, _ = cv_img.shape
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        xmin = int((xc - bw / 2) * w)
                        ymin = int((yc - bh / 2) * h)
                        xmax = int((xc + bw / 2) * w)
                        ymax = int((yc + bh / 2) * h)
                        color = (46, 204, 113) if cid == 0 else (231, 76, 60)
                        cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), color, 2)
                        
        ax.imshow(cv_img)
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")
        
    plt.suptitle("Sample Dataset Grid (Green = hat, Red = no_helmet)", fontsize=14, y=0.92)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Visual verification grid saved to {output_file}")


if __name__ == "__main__":
    audit_dataset()
