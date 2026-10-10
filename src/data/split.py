"""
split.py - Group-Aware Stratified Leakage-Free Dataset Splitter
--------------------------------------------------------------
Clusters near-duplicate images (pHash distance <= 4) into connected components (groups).
Performs stratified group splitting (70% Train, 15% Val, 15% Test) ensuring:
1. No cluster of duplicate or near-duplicate scenes ever spans across splits.
2. The severe class imbalance (7.5% hat vs 92.5% no_helmet) is preserved across splits.
3. Automatically locks test.txt as read-only.
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

from src.utils.guard import lock_test_file, assert_zero_leakage


def load_annotations(labels_dir: Path) -> Dict[str, Tuple[int, int]]:
    """Loads box counts (hat_count, no_helmet_count) for each image stem."""
    ann_map = {}
    for lbl_file in labels_dir.glob("*.txt"):
        hat_cnt = 0
        no_helmet_cnt = 0
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    if cid == 0:
                        hat_cnt += 1
                    else:
                        no_helmet_cnt += 1
        ann_map[lbl_file.stem] = (hat_cnt, no_helmet_cnt)
    return ann_map


def cluster_images_by_similarity(
    img_files: List[Path],
    hamming_threshold: int = 4
) -> Tuple[List[List[str]], Dict[str, imagehash.ImageHash]]:
    """
    Groups images into connected components where edges represent
    exact MD5 equality or pHash Hamming distance <= threshold.
    """
    print(f"[INFO] Computing perceptual hashes for {len(img_files)} images...")
    md5_dict = {}
    phash_dict = {}
    stem_to_idx = {img.stem: i for i, img in enumerate(img_files)}
    stems = [img.stem for img in img_files]
    
    for img_path in tqdm(img_files, desc="Hashing Images"):
        stem = img_path.stem
        with open(img_path, "rb") as f:
            md5_dict[stem] = hashlib.md5(f.read()).hexdigest()
        with Image.open(img_path) as pil_img:
            phash_dict[stem] = imagehash.phash(pil_img)
            
    # Build Adjacency Graph
    adj = defaultdict(set)
    
    # 1. Exact MD5 duplicates
    md5_groups = defaultdict(list)
    for stem, h in md5_dict.items():
        md5_groups[h].append(stem)
    for group in md5_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                adj[group[i]].add(group[j])
                adj[group[j]].add(group[i])
                
    # 2. Perceptual near-duplicates via Pigeonhole Multi-Index Hashing
    # 64 bits partitioned into 5 chunks (13, 13, 13, 13, 12 bits).
    # If distance <= 4, by Pigeonhole Principle, at least 1 chunk MUST match exactly!
    print("[INFO] Building guaranteed near-duplicate graph using Multi-Index Pigeonhole Hashing...")
    pigeonhole_buckets = defaultdict(list)
    for stem, ph in phash_dict.items():
        bits = ph.hash.flatten()
        chunks = [
            (0, tuple(bits[0:13])),
            (1, tuple(bits[13:26])),
            (2, tuple(bits[26:39])),
            (3, tuple(bits[39:52])),
            (4, tuple(bits[52:64])),
        ]
        for chunk in chunks:
            pigeonhole_buckets[chunk].append(stem)
            
    # Connect candidate pairs
    checked_pairs = set()
    for bucket in pigeonhole_buckets.values():
        if len(bucket) <= 1:
            continue
        for i in range(len(bucket)):
            s1 = bucket[i]
            for j in range(i + 1, len(bucket)):
                s2 = bucket[j]
                pair_key = (min(s1, s2), max(s1, s2))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                dist = phash_dict[s1] - phash_dict[s2]
                if dist <= hamming_threshold:
                    adj[s1].add(s2)
                    adj[s2].add(s1)

    # 3. Find Connected Components via BFS
    visited = set()
    clusters = []
    for stem in stems:
        if stem in visited:
            continue
        cluster = []
        queue = [stem]
        visited.add(stem)
        while queue:
            node = queue.pop(0)
            cluster.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append(cluster)
        
    print(f"[INFO] Clustered {len(stems)} images into {len(clusters)} disjoint groups.")
    multi_img_clusters = [c for c in clusters if len(c) > 1]
    print(f"       Found {len(multi_img_clusters)} duplicate/near-duplicate groups containing {sum(len(c) for c in multi_img_clusters)} images.")
    return clusters, phash_dict


def stratified_group_split(
    clusters: List[List[str]],
    ann_map: Dict[str, Tuple[int, int]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Distributes clusters into Train, Val, and Test to maintain
    image counts and class balance while guaranteeing group isolation.
    """
    np.random.seed(seed)
    
    # Calculate group metadata
    group_stats = []
    total_hat = 0
    total_no_helmet = 0
    total_images = 0
    
    for c in clusters:
        h_cnt = sum(ann_map.get(stem, (0, 0))[0] for stem in c)
        nh_cnt = sum(ann_map.get(stem, (0, 0))[1] for stem in c)
        group_stats.append({
            "stems": c,
            "img_count": len(c),
            "hat_count": h_cnt,
            "no_helmet_count": nh_cnt
        })
        total_hat += h_cnt
        total_no_helmet += nh_cnt
        total_images += len(c)
        
    # Target allocations
    targets = {
        "train": {"images": total_images * train_ratio, "hat": total_hat * train_ratio},
        "val": {"images": total_images * val_ratio, "hat": total_hat * val_ratio},
        "test": {"images": total_images * test_ratio, "hat": total_hat * test_ratio},
    }
    
    # Sort groups descending by box volume for greedy balanced packing
    group_stats.sort(key=lambda x: (x["hat_count"] + x["no_helmet_count"], x["img_count"]), reverse=True)
    
    allocations = {"train": [], "val": [], "test": []}
    current = {
        "train": {"images": 0, "hat": 0, "no_helmet": 0},
        "val": {"images": 0, "hat": 0, "no_helmet": 0},
        "test": {"images": 0, "hat": 0, "no_helmet": 0},
    }
    
    # Multi-objective greedy allocation
    for grp in group_stats:
        # Score candidate splits: distance to remaining capacity
        best_split = None
        best_score = float("inf")
        
        for split in ["train", "val", "test"]:
            # Image score and hat ratio score
            img_diff = (current[split]["images"] + grp["img_count"]) / targets[split]["images"]
            hat_diff = (current[split]["hat"] + grp["hat_count"]) / max(1, targets[split]["hat"])
            score = 0.5 * img_diff + 0.5 * hat_diff
            
            if score < best_score:
                best_score = score
                best_split = split
                
        allocations[best_split].extend(grp["stems"])
        current[best_split]["images"] += grp["img_count"]
        current[best_split]["hat"] += grp["hat_count"]
        current[best_split]["no_helmet"] += grp["no_helmet_count"]

    return allocations["train"], allocations["val"], allocations["test"]


def generate_leakage_free_splits(
    processed_dir: Path = Path("data/processed"),
    splits_dir: Path = Path("data/splits"),
    reports_dir: Path = Path("reports")
) -> Dict:
    """
    Orchestrates leakage-free split generation, writes files, locks test set, and exports summary.
    """
    img_dir = processed_dir / "images"
    lbl_dir = processed_dir / "labels"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    ann_map = load_annotations(lbl_dir)
    
    # 1. Cluster images into groups
    clusters, phash_dict = cluster_images_by_similarity(img_files)
    
    # 2. Stratified group splitting
    train_stems, val_stems, test_stems = stratified_group_split(clusters, ann_map)
    
    # 3. Mathematically assert zero leakage
    assert_zero_leakage(train_stems, val_stems, test_stems, phash_dict, distance_threshold=4)
    
    # 4. Write split files formatted with standard image paths
    split_paths = {
        "train": splits_dir / "train.txt",
        "val": splits_dir / "val.txt",
        "test": splits_dir / "test.txt"
    }
    
    for split_name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        file_path = split_paths[split_name]
        # Unlock if it was previously locked
        if split_name == "test" and file_path.exists():
            from src.utils.guard import unlock_test_file
            unlock_test_file(file_path)
            
        lines = [f"data/processed/images/{s}.jpg" for s in sorted(stems)]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
    # 5. Security write-lock on test.txt
    lock_test_file(split_paths["test"])
    
    # 6. Compute statistics summary
    total_imgs = len(img_files)
    summary = {
        "splits_summary": {
            "train": {
                "images": len(train_stems),
                "percentage": round(100 * len(train_stems) / total_imgs, 2),
                "hat_boxes": sum(ann_map.get(s, (0, 0))[0] for s in train_stems),
                "no_helmet_boxes": sum(ann_map.get(s, (0, 0))[1] for s in train_stems),
            },
            "val": {
                "images": len(val_stems),
                "percentage": round(100 * len(val_stems) / total_imgs, 2),
                "hat_boxes": sum(ann_map.get(s, (0, 0))[0] for s in val_stems),
                "no_helmet_boxes": sum(ann_map.get(s, (0, 0))[1] for s in val_stems),
            },
            "test": {
                "images": len(test_stems),
                "percentage": round(100 * len(test_stems) / total_imgs, 2),
                "hat_boxes": sum(ann_map.get(s, (0, 0))[0] for s in test_stems),
                "no_helmet_boxes": sum(ann_map.get(s, (0, 0))[1] for s in test_stems),
            }
        },
        "leakage_verification": {
            "stem_cross_overlap": 0,
            "perceptual_hash_cross_leaks": 0,
            "test_isolation_status": "LOCKED (Read-Only Guard Enabled)"
        }
    }
    
    # Compute per-split class ratios
    for s_name in ["train", "val", "test"]:
        s_data = summary["splits_summary"][s_name]
        t_boxes = s_data["hat_boxes"] + s_data["no_helmet_boxes"]
        s_data["hat_ratio"] = f"{round(100 * s_data['hat_boxes'] / max(1, t_boxes), 2)}%"
        s_data["no_helmet_ratio"] = f"{round(100 * s_data['no_helmet_boxes'] / max(1, t_boxes), 2)}%"

    summary_file = reports_dir / "leakage_free_split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[SUCCESS] Leakage-free splits generated successfully at {splits_dir}.")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    generate_leakage_free_splits()
