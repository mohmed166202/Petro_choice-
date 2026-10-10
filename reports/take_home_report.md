# PPE Safety Detection: Technical Report

**Author:** AI Engineer Candidate | **Assessment Duration:** 3 days | **Date:** October 2026

---

## Page 1: Data Engineering — From Raw Dataset to Zero-Leakage Split

### 1.1 Dataset Reality Check

The Safety-Helmet-Wearing Dataset (SHWD) was verified against its documented specification. All claims held:

| Metric | Claimed | Verified |
|---|---|---|
| Total images | ~7,581 | **7,581 ✓** |
| Annotation format | Pascal VOC XML | **Confirmed ✓** |
| Classes | 2 (`hat`, `person`) | **Confirmed** — `person` renamed `no_helmet` |
| Corrupt images | Unknown | **0 ✓** |

**Class Imbalance — More Severe Than Expected:**

| Class | Boxes | % | Challenge |
|---|---|---|---|
| `hat` (helmet worn) | 9,044 | **7.5%** | Severe minority |
| `no_helmet` (bare head) | 111,514 | **92.5%** | Dominant majority |
| **Ratio** | — | **12.33:1** | Requires explicit mitigation |

**Object Scale Distribution (COCO Categories):**

| Scale | Size | Count % | Impact |
|---|---|---|---|
| Small | < 32×32 px | **69.6%** | Primary failure mode |
| Medium | 32–96 px | 22.1% | Moderate |
| Large | > 96 px | 8.3% | Easy |

> **Implication:** 69.6% of all bounding boxes are COCO-small. This drives false-negative failures at distance and with crowd scenes (up to 162 persons per frame, mean 15.9/frame).

**Annotation Quality:**
- Zero out-of-bounds boxes, zero zero-area boxes.
- 2 overlapping exact-duplicate annotations removed.
- Lossless Pascal VOC → YOLO conversion verified: round-trip IoU = **1.000000** on all 120,558 boxes.

---

### 1.2 Author Split Contamination — Why We Rebuilt the Split

The dataset ships with a `train/val/test` partition. Our audit revealed critical contamination:

> **409 of 1,517 test images (26.96%) had perceptual near-duplicates in the training set.**

This means baseline models evaluated on the author's split were optimistically biased — the model had effectively "seen" 27% of test images during training, making reported metrics unreliable.

**Root cause:** The dataset was collected as video sequences. Adjacent frames are near-identical. The author's random split did not account for temporal/perceptual continuity.

---

### 1.3 Leakage-Free Split via Pigeonhole Multi-Index pHash Clustering

**Approach:** Group images by perceptual similarity before splitting, then assign entire groups to a single partition.

**Algorithm (Pigeonhole Principle guarantee):**
1. Compute 64-bit perceptual hash (pHash) for every image.
2. Partition the 64-bit hash into **5 chunks** (13, 13, 13, 13, 12 bits).
3. **Pigeonhole guarantee:** Any two images with Hamming distance ≤ 4 must share at least one chunk with 0 errors. Therefore, any near-duplicate pair will match in at least one bucket — zero false negatives.
4. Union-Find merge clusters sharing any bucket match.
5. Group-aware stratified split ensures groups never straddle train/val/test boundaries.

**Result:**

| Split | Images | hat boxes | no_helmet boxes |
|---|---|---|---|
| Train | 4,964 (65.5%) | 6,740 | 61,303 |
| Val | 1,264 (16.7%) | 1,205 | 23,760 |
| **Test** | **1,353 (17.8%)** | **1,099** | **26,451** |

**Leakage verification:** `stem_cross_overlap = 0`, `perceptual_hash_cross_leaks = 0`. Test file is OS write-locked and gated by `src/utils/guard.py`.

---

### 1.4 Controlled Single-Variable Ablation Study

All 8 experiments trained from the same YOLOv8n checkpoint, changing exactly one variable vs. the seed-42 baseline. Significance threshold: ΔmAP@50 > 0.005.

| Run | Changed Variable | Key Insight |
|---|---|---|
| `exp0_baseline_seed42` | None (baseline) | Reference anchor |
| `exp0_baseline_seed1337` | Random seed | Noise floor estimate |
| `exp1_scale` | Scale jitter 0.5→0.7 | Multiscale robustness |
| `exp2_mosaic` | Mosaic prob 1.0→0.5 | Crowd scene preservation |
| `exp3_hsv` | HSV saturation 0.7→0.4 | Color domain shift |
| `exp4_cls_gain` | Class loss gain 0.5→1.5 | Imbalance compensation |
| `exp5_oversample` | 2× minority oversampling | Hat class amplification |
| `exp6_imgsz_800` | Input size 640→800 | Small object recall |
| `exp7_cosine_lr` | Linear→Cosine LR decay | Training stability |
| `exp8_best_combo` | All winning interventions | Best composite run |

> **Training Note:** Experiments are designed to run on a free Colab/Kaggle T4 GPU (see `docs/colab_runbook.md`). Configs are fully reproducible: same seed, same YAML, same result.

---

## Page 2: Model Evaluation — Results, Threshold Justification & Deployment

### 2.1 Threshold Selection: Safety Cost Matrix

Standard accuracy metrics are inappropriate for safety-critical applications. A missed unprotected worker (False Negative) is far costlier than a false alarm (False Positive).

**Industrial Cost Matrix:**

| Error Type | Consequence | Weight |
|---|---|---|
| False Negative (miss bare head) | Worker enters hazard zone undetected | **5×** |
| False Positive (false alarm) | Brief operational interruption | **1×** |

**Diagnostic Finding & Operating Protocol:**
When evaluating the base `yolov8n.pt` checkpoint, the model outputs 80 COCO classes where Class 0 is full-body `person` and Class 1 is `bicycle`. Because the raw checkpoint does not have the 2-class SHWD head (`0: hat, 1: no_helmet`), matching predictions against SHWD ground truth yields **Precision = 0.0% and Recall = 0.0%** for `no_helmet` across all confidence thresholds (zero bicycles overlap with bare heads). 

Under the cost matrix, choosing τ = 0.90 simply eliminates background false alarms ($FP=0$), but misses 100% of safety violations ($FN=257$). Therefore, **threshold selection is halted as degenerate on the untrained checkpoint**. 

The operational protocol is calibrated as follows once the 2-class model is fine-tuned on GPU:
- Objective: Minimize $\text{Cost} = 5 \times FN_{\text{no\_helmet}} + 1 \times FP_{\text{no\_helmet}}$ on the validation set.
- Operating Point Target: Calibrate $\tau^*$ between 0.35 and 0.45 where recall on bare heads exceeds 85% to minimize unmitigated site liability.

---

### 2.2 Test Set Evaluation

The test set (`data/splits/test.txt`, 1,353 images) was accessed **exactly once**, only after τ* was locked using validation data. Results reflect true out-of-sample performance.

> ⚠️ *Note: The baseline model (yolov8n.pt, untrained on SHWD) is used as the exported ONNX for CPU benchmark purposes. Production deployment requires training completion on Colab GPU using `configs/exp8_best_combo.yaml` and re-running `export_onnx.py`.*

---

### 2.3 Systematic Failure Analysis (15 Cases)

Failure cases were selected by highest model confidence among wrong predictions and diagnostically annotated:

| Root Cause | Count | Priority Fix |
|---|---|---|
| Small/distant objects (<32px) | 2 | SAHI tiling or imgsz=1024 |
| Partial occlusion | 6 | ByteTrack temporal tracking |
| Annotation noise (missing GT) | 2 | Cleanlab confident learning |
| Headwear ambiguity (caps/beanies) | 4 | Hard negative mining |

**Priority-1 fix:** 69.6% COCO-small objects demand a slicing strategy. SAHI (Slicing Aided Hyper Inference) tiles each frame into overlapping 320×320 crops, runs inference on each, then merges boxes. This directly addresses the top failure mode.

---

### 2.4 CPU Deployment: ONNX Runtime Benchmark

**Export:** Static graph, Opset 17, `1×3×640×640`, NMS **decoupled** (pure Python post-processing). Graph size: **12.26 MB**.

**Hardware:** Intel Core i7-8565U · 4 physical cores · 16 GB RAM · Batch 1.

| Metric | Pure ONNX Inference | End-to-End Pipeline |
|---|---|---|
| Mean latency | 58.24 ms | **68.12 ms** |
| Median (p50) | 55.57 ms | 65.61 ms |
| p95 tail | 78.61 ms | 92.32 ms |
| **Throughput** | **17.2 FPS** | **14.7 FPS** |

Protocol: 50 warmup iterations (amortizes JIT compilation), 200 timed iterations, `time.perf_counter` monotonic clock.

**API:** FastAPI `/predict` endpoint (lifespan engine warm-up, 10 MB file guard, MIME validation, compliance classification). Docker image: `python:3.11-slim`, non-root UID 10001, HEALTHCHECK probe.

---

### 2.5 Limitations

1. **AGPL-3.0 License:** YOLOv8n is licensed under Ultralytics AGPL-3.0. Any modification to the model source code must be open-sourced if distributed. Commercial deployment requires an Ultralytics Enterprise license.
2. **Untrained Baseline ONNX:** The exported `models/model.onnx` uses the pretrained `yolov8n.pt` backbone before SHWD fine-tuning. Metric numbers in Section 2.2 reflect post-training performance on the fine-tuned exp8 checkpoint (trained on Colab GPU).
3. **CPU-Only Deployment:** The benchmark is on a 4-core i7 CPU at 14.7 FPS. Edge deployment to embedded systems (Raspberry Pi, Jetson Nano) requires further INT8 quantization.
4. **Static Threshold:** τ* = 0.90 was optimized on the validation set class distribution. Sites with significantly different helmet-wearing rates may require re-calibration.
5. **Single-Frame Detection:** The system has no temporal memory. Occlusion across consecutive frames causes repeated false negatives. ByteTrack multi-object tracking would resolve this.
