"""
leakage_audit.py - Cross-Split Data Leakage Auditor for SHWD
------------------------------------------------------------
Measures data contamination in the author's original trainval vs test split:
- Exact duplicate images (MD5 collision) crossing the split boundary.
- Perceptual near-duplicates (pHash Hamming distance <= 4) crossing the split boundary.
- Quantifies the test contamination rate to provide hard evidence for the technical report.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image
import imagehash
from tqdm import tqdm


def load_split_stems(split_file: Path) -> Set[str]:
    """Reads image filename stems from a split file."""
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def compute_hashes(img_paths: List[Path]) -> Tuple[Dict[str, str], Dict[str, imagehash.ImageHash]]:
    """Computes MD5 and perceptual hash for a list of image paths."""
    md5_map = {}
    phash_map = {}
    for p in tqdm(img_paths, desc="Hashing Images for Leakage Audit"):
        with open(p, "rb") as f:
            md5_map[p.stem] = hashlib.md5(f.read()).hexdigest()
        with Image.open(p) as pil_img:
            phash_map[p.stem] = imagehash.phash(pil_img)
    return md5_map, phash_map


def audit_author_split(
    author_split_dir: Path = Path("data/raw/VOC2028/ImageSets/Main"),
    images_dir: Path = Path("data/processed/images"),
    reports_dir: Path = Path("reports")
) -> Dict:
    """
    Audits the original author's trainval.txt vs test.txt for cross-split leakage.
    """
    trainval_stems = load_split_stems(author_split_dir / "trainval.txt")
    test_stems = load_split_stems(author_split_dir / "test.txt")
    
    print(f"[INFO] Author Split: {len(trainval_stems)} trainval images, {len(test_stems)} test images.")
    
    # Verify exact stem overlap
    stem_overlap = trainval_stems.intersection(test_stems)
    
    # Locate images
    all_imgs = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    stem_to_path = {img.stem: img for img in all_imgs}
    
    trainval_paths = [stem_to_path[s] for s in trainval_stems if s in stem_to_path]
    test_paths = [stem_to_path[s] for s in test_stems if s in stem_to_path]
    
    print(f"[INFO] Hashing {len(trainval_paths)} trainval and {len(test_paths)} test images...")
    tv_md5, tv_phash = compute_hashes(trainval_paths)
    test_md5, test_phash = compute_hashes(test_paths)
    
    # 1. Exact MD5 duplicates across splits
    md5_to_tv_stems = {}
    for stem, h in tv_md5.items():
        md5_to_tv_stems.setdefault(h, []).append(stem)
        
    cross_split_exact_duplicates = []
    for test_stem, h in test_md5.items():
        if h in md5_to_tv_stems:
            for tv_stem in md5_to_tv_stems[h]:
                cross_split_exact_duplicates.append((test_stem, tv_stem))
                
    # 2. Near-duplicate perceptual hash leakage (Hamming distance <= 4)
    print("[INFO] Checking cross-split perceptual hash distances...")
    cross_split_near_duplicates = []
    contaminated_test_images = set()
    
    # Vectorized / pairwise check
    tv_stems_list = list(tv_phash.keys())
    for test_stem, t_hash in tqdm(test_phash.items(), desc="Auditing Cross-Split Leaks"):
        for tv_stem in tv_stems_list:
            dist = t_hash - tv_phash[tv_stem]
            if dist <= 4:
                cross_split_near_duplicates.append({
                    "test_image": test_stem,
                    "trainval_image": tv_stem,
                    "hamming_distance": int(dist)
                })
                contaminated_test_images.add(test_stem)

    test_contamination_pct = round(100.0 * len(contaminated_test_images) / max(1, len(test_stems)), 2)

    leakage_summary = {
        "author_split_counts": {
            "trainval_images": len(trainval_stems),
            "test_images": len(test_stems),
            "total_images": len(trainval_stems) + len(test_stems),
            "stem_exact_overlap": len(stem_overlap)
        },
        "leakage_findings": {
            "cross_split_exact_duplicates_count": len(cross_split_exact_duplicates),
            "cross_split_near_duplicate_pairs": len(cross_split_near_duplicates),
            "contaminated_test_images_count": len(contaminated_test_images),
            "test_set_contamination_percentage": f"{test_contamination_pct}%"
        },
        "sample_leaks": cross_split_near_duplicates[:10]
    }
    
    out_file = reports_dir / "author_split_leakage.json"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(leakage_summary, f, indent=2)
        
    print(f"\n[RESULT] Author Split Leakage Audit complete.")
    print(f"  Exact cross-split duplicate pairs: {len(cross_split_exact_duplicates)}")
    print(f"  Near-duplicate cross-split pairs:  {len(cross_split_near_duplicates)}")
    print(f"  Contaminated test set images:      {len(contaminated_test_images)} / {len(test_stems)} ({test_contamination_pct}%)")
    print(f"  Saved findings to {out_file}")
    
    return leakage_summary


if __name__ == "__main__":
    audit_author_split()
