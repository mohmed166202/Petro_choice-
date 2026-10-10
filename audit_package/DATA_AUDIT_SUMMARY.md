# Data Audit & Split Integrity Summary

**Dataset:** Safety Helmet Wearing Dataset (SHWD / VOC2028 format)  
**Total Images:** 7,581  
**Total Annotations:** 120,558 bounding boxes  
**Audit Artifacts:** [`reports/data_audit_summary.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/data_audit_summary.json), [`reports/author_split_leakage.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/author_split_leakage.json), [`reports/leakage_free_split_summary.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/leakage_free_split_summary.json)  

---

## 1. Class Distribution & Extreme Imbalance

The dataset defines two target detection classes:
- **Class 0 (`hat`):** Worker wearing an industrial safety helmet / hard hat.
- **Class 1 (`no_helmet`):** Worker / pedestrian without a safety helmet (bare head).

| Class | Absolute Count | Percentage | Ratio Relative to Minority (`hat`) |
| :--- | :--- | :--- | :--- |
| `hat` | 9,044 | 7.50% | 1.00 : 1 |
| `no_helmet` | 111,514 | 92.50% | **12.33 : 1** |
| **Total** | **120,558** | **100.00%** | — |

> **Critical Note on Domain Scope:** The dataset only annotates hard hats and bare heads. It does **not** annotate full-body Personal Protective Equipment (such as reflective safety vests, safety goggles, steel-toe boots, or fall harnesses). The system is an industrial helmet compliance detector.

---

## 2. Bounding Box Geometry & Scale Breakdown

Bounding boxes were evaluated against standard COCO scale criteria:
- **Small ($< 32 \times 32\text{ px}$):** **69.60%** (83,908 boxes)
- **Medium ($32 \times 32\text{ to }96 \times 96\text{ px}$):** **22.11%** (26,655 boxes)
- **Large ($> 96 \times 96\text{ px}$):** **8.29%** (9,995 boxes)

- **Mean boxes per image:** 15.9 boxes
- **Maximum boxes in a single image:** 162 boxes (dense crowd scenes)
- **Corrupt images:** 0
- **Degenerate / zero-area boxes:** 0
- **Out-of-bounds coordinates:** 0 (all clamped to $[0, 1]$ in normalized YOLO format)
- **Duplicate overlapping boxes:** 2 (resolved during VOC-to-YOLO conversion)

---

## 3. Duplicate Detection & Author Split Contamination

An image hash audit was performed on the raw dataset:
- **Exact duplicate groups:** 37 groups identified via MD5 / SHA-256 byte hashing.
- **Near-duplicate pairs:** 8,475 pairs identified via perceptual hashing ($\text{Hamming distance } d \le 4$).

### Original Author Split Audit
In the repository's source distribution, the author provided a default split of 6,064 trainval images and 1,517 test images. Auditing this split revealed critical data leakage:
- **Cross-split exact duplicates:** 9 identical images shared across trainval and test.
- **Cross-split near-duplicate pairs:** 17,315 pairs shared across trainval and test.
- **Contaminated test set images:** **409 out of 1,517 test images (26.96%)** were perceptual duplicates of training images.
- **Implication:** Any benchmark reported using the author's original split suffers from severe test set memorization and inflated mAP metrics.

---

## 4. Leakage-Free Split Design & Verification

To establish an uncompromised evaluation baseline, a connected-component duplicate-grouping algorithm with stratified partitioning was executed:

| Split Partition | Image Count | Image % | Hat Boxes | No-Helmet Boxes | Hat Box % | No-Helmet Box % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 4,964 | 65.48% | 6,740 | 61,303 | 9.91% | 90.09% |
| **Validation** | 1,264 | 16.67% | 1,205 | 23,760 | 4.83% | 95.17% |
| **Test** | 1,353 | 17.85% | 1,099 | 26,451 | 3.99% | 96.01% |
| **Total** | **7,581** | **100.00%** | **9,044** | **111,514** | **7.50%** | **92.50%** |

### Verified Split Invariants:
1. **Zero Exact Stem Overlap:** $\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$.
2. **Zero Perceptual Hash Overlap:** $d(\text{Test}, \text{Train} \cup \text{Val}) > 4$ across all 1,353 test images.
3. **OS-Level Test Set Lock:** The file [`data/splits/test.txt`](file:///c:/Users/moham/Downloads/petrochoice%20Task/data/splits/test.txt) has its OS read-only attribute set (`-ar---`).
4. **Caller Gatekeeping:** [`src/utils/guard.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/utils/guard.py) enforces caller inspection, actively rejecting training or hyperparameter tuning scripts if they attempt to open `test.txt`.
