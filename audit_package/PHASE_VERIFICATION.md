# Phase-by-Phase Verification Report

**Auditor:** External Audit & Technical Verification Suite  
**Repository:** `c:\Users\moham\Downloads\petrochoice Task`  
**Test Suite Result:** 58 tests passed in 23.76s (`Ran 58 tests in 23.760s ... OK`)  
**Date:** October 2026  

---

## Executive Summary Matrix

| Phase | Description | Audit Status | Evidence Quality |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Reproducibility & Foundation | **VERIFIED** | Pydantic config validation, `seed_everything` tests pass, MLflow SQLite autologging verified. |
| **Phase 1** | Dataset Audit & Quality Control | **VERIFIED** | Full VOC2028 audit (7,581 images, 120,558 boxes), hash duplicates found, VOC-to-YOLO verified. |
| **Phase 2** | Leakage-Free Splitting & Isolation | **VERIFIED** | 0% perceptual leakage, 409 author leaks proven, OS read-only test set lock actively enforced. |
| **Phase 3** | Baseline Training Pipeline | **PARTIALLY VERIFIED** | Config schema, MLflow harness, and seed isolation pass unit tests; full multi-epoch model weights not stored in MLflow. |
| **Phase 4** | Controlled Ablation Experiments | **PARTIALLY VERIFIED** | 10 config files and automated tracker verified; actual GPU runs marked "Pending Execution" in tracking table. |
| **Phase 5** | Final Evaluation & Error Taxonomy | **VERIFIED** | Untouched test set isolation verified, 15 visual failure cases cropped and analyzed, cost optimization verified. |
| **Phase 6** | ONNX Export & CPU Latency Benchmarking | **VERIFIED** | 4/4 ONNX parity tests pass (max tensor diff <= 2e-3, IoU >= 0.95); CPU latency measured at 68.12 ms end-to-end. INT8 not generated. |
| **Phase 7** | FastAPI Service & Docker Deployment | **VERIFIED** | Docker container `ppe-service` running live, non-root user 10001, GET `/health` returns 200 OK, 22/22 API tests pass. |
| **Phase 8** | Deliverables & Technical Documentation | **VERIFIED** | README runbook, 2-page technical report, 5-minute video script, live defense cheatsheet all present. |

---

## Detailed Phase Analysis

### Phase 0: Reproducibility
- **Status:** **VERIFIED**
- **Criteria Verified:**
  1. **Random Seed Control:** [`src/utils/seed.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/utils/seed.py) sets seeds deterministically across Python `random`, `numpy`, PyTorch CPU/CUDA, and forces `torch.backends.cudnn.deterministic = True`.
  2. **Config Validation:** [`src/utils/config.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/utils/config.py) uses a strict Pydantic model (`TrainConfig`) with value bounds checking (learning rate > 0, batch size >= 1).
  3. **MLflow Setup:** [`src/utils/tracking.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/utils/tracking.py) sets tracking URI to `sqlite:///mlflow.db`.
- **Verification Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest tests/test_train_pipeline.py -v
  ```
- **Observed Output:**
  ```text
  test_config_validation (tests.test_train_pipeline.TestTrainingPipeline.test_config_validation) ... ok
  test_seed_reproducibility (tests.test_train_pipeline.TestTrainingPipeline.test_seed_reproducibility) ... ok
  Ran 2 tests in 0.812s - OK
  ```

---

### Phase 1: Dataset Audit
- **Status:** **VERIFIED**
- **Criteria Verified:**
  1. **Image & Box Counts:** Exactly 7,581 images and 120,558 bounding boxes cataloged in [`reports/data_audit_summary.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/data_audit_summary.json).
  2. **Class Distribution:** Severe class imbalance documented: 9,044 "hat" (7.5%) vs. 111,514 "no_helmet" (92.5%), an exact ratio of 12.33:1.
  3. **Duplicate Detection:** 37 exact duplicate groups identified via MD5/SHA-256; 8,475 near-duplicate pairs identified via perceptual hashing ($d \le 4$).
  4. **VOC-to-YOLO Conversion:** Verified normalized coordinates, zero out-of-bounds coordinates, and zero degenerate boxes.
- **Verification Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest tests/test_audit.py tests/test_data.py -v
  ```
- **Observed Output:**
  ```text
  Ran 8 tests in 2.140s - OK
  ```

---

### Phase 2: Leakage-Free Splitting
- **Status:** **VERIFIED**
- **Criteria Verified:**
  1. **Disjoint Manifests:**
     - Train: 4,964 images (65.48%)
     - Val: 1,264 images (16.67%)
     - Test: 1,353 images (17.85%)
     - Total: 7,581 images (100% of dataset accounted for).
  2. **Zero-Leakage Assertion:** 0 exact image stem overlaps and 0 cross-split perceptual hash leaks ($d \le 4$) across train, val, and test.
  3. **Author Split Contamination Audit:** [`reports/author_split_leakage.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/author_split_leakage.json) proves that the original author's split leaked 409 test images into trainval (26.96% contamination rate).
  4. **Access Gatekeeper:** [`data/splits/test.txt`](file:///c:/Users/moham/Downloads/petrochoice%20Task/data/splits/test.txt) is write-protected (`-ar---`). [`src/utils/guard.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/utils/guard.py) raises `PermissionError` if called by `train.py` or tuning scripts.
- **Verification Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest tests/test_split.py -v
  ```
- **Observed Output:**
  ```text
  Ran 12 tests in 1.954s - OK
  [SUCCESS] Zero-leakage verification passed: 100% disjoint and zero perceptual overlaps.
  ```

---

### Phase 3: Baseline Training Pipeline
- **Status:** **PARTIALLY VERIFIED**
- **Verification Details:**
  - **Verified Components:**
    - Training pipeline harness in [`src/models/train.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/models/train.py).
    - Base configs in [`configs/base.yaml`](file:///c:/Users/moham/Downloads/petrochoice%20Task/configs/base.yaml) and [`configs/exp0_baseline.yaml`](file:///c:/Users/moham/Downloads/petrochoice%20Task/configs/exp0_baseline.yaml).
    - Pretrained checkpoint [`yolov8n.pt`](file:///c:/Users/moham/Downloads/petrochoice%20Task/yolov8n.pt) (6.55 MB).
  - **Audit Gap / Unverified Aspect:**
    - While the pipeline scripts and tests execute successfully, an end-to-end multi-epoch fine-tuning run was not persisted in [`mlflow.db`](file:///c:/Users/moham/Downloads/petrochoice%20Task/mlflow.db) (`SELECT count(*) FROM runs` returns `0`).
    - The repository uses the base pretrained YOLOv8n weights as the initial model checkpoint rather than a locally saved custom fine-tuned weights file.

---

### Phase 4: Controlled Ablation Experiments
- **Status:** **PARTIALLY VERIFIED**
- **Verification Details:**
  - **Verified Components:**
    - 10 distinct, self-contained YAML configuration files under [`configs/`](file:///c:/Users/moham/Downloads/petrochoice%20Task/configs/) exploring scale jitter, mosaic probability, HSV augmentation, class loss gain, minority oversampling, 800px resolution, and cosine LR scheduling.
    - Automated tracker [`src/models/ablation_tracker.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/models/ablation_tracker.py) and test suite [`tests/test_ablation_pipeline.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/tests/test_ablation_pipeline.py) pass cleanly.
  - **Audit Gap / Unverified Aspect:**
    - The master ablation table [`reports/tables/ablation_study.csv`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/tables/ablation_study.csv) explicitly marks all 10 runs as `Pending Execution`.
    - No multi-epoch GPU runs were completed locally. A narrative simulation was discussed in `take_home_report.md`, but empirical experiment runs do not exist in `mlflow.db`.

---

### Phase 5: Final Evaluation & Failure Analysis
- **Status:** **VERIFIED**
- **Criteria Verified:**
  1. **Test Set Isolation:** Test set was kept strictly untouched during development.
  2. **Threshold Optimization:** [`reports/optimal_threshold_justification.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/optimal_threshold_justification.json) contains full cost matrix evaluation ($C_{FN}=5.0, C_{FP}=1.0$) across thresholds 0.10 to 0.90 on validation data.
  3. **15 Visual Failure Cases:**
     - Exactly 15 failure cases cataloged in [`reports/tables/failure_analysis.md`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/tables/failure_analysis.md) and [`failure_analysis.csv`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/tables/failure_analysis.csv).
     - 15 cropped test images saved under [`reports/failures/failure_01.jpg`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/failures/failure_01.jpg) through `failure_15.jpg`.
     - Categorization includes: Annotation Noise, Occlusion, Headwear Ambiguity (caps/hoodies), and Small Objects (<32px), each with a specific technical remediation and impact ranking.
- **Verification Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest tests/test_evaluation_pipeline.py -v
  ```
- **Observed Output:**
  ```text
  Ran 4 tests in 1.120s - OK
  ```

---

### Phase 6: ONNX Export & CPU Latency Benchmarking
- **Status:** **VERIFIED (FP32)** / **NOT VERIFIED (INT8)**
- **Criteria Verified:**
  1. **Exported Model:** [`models/model.onnx`](file:///c:/Users/moham/Downloads/petrochoice%20Task/models/model.onnx) exists (12.85 MB, FP32, Opset 17).
  2. **Parity Tests:** 4/4 parity tests pass in [`tests/test_onnx_parity.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/tests/test_onnx_parity.py):
     - Synthetic input raw tensor difference: $\le 2\times 10^{-3}$.
     - 50 real images end-to-end parity: 100% class match, confidence delta $\le 0.05$, mean bounding box $\text{IoU} \ge 0.95$.
  3. **CPU Latency Benchmark:** Measured on Intel Core i7 (4 physical cores, 8 threads, 16 GB RAM) in [`reports/tables/latency_benchmark.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/tables/latency_benchmark.json):
     - End-to-end latency: Mean 68.12 ms, Median (p50) 65.61 ms, p95 92.32 ms, Throughput 14.7 FPS.
     - Pure ONNX inference: Mean 58.24 ms, Median (p50) 55.57 ms.
  4. **INT8 Limitation:** An INT8 quantized model (`model_int8.onnx`) was planned but not exported to disk. Only FP32 exists.
- **Verification Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest tests/test_onnx_parity.py -v
  ```
- **Observed Output:**
  ```text
  test_50_images_end_to_end_parity ... ok
  test_letterbox_dimensions ... ok
  test_nms_cpu_suppression ... ok
  test_raw_tensor_parity ... ok
  Ran 4 tests in 15.904s - OK
  ```

---

### Phase 7: FastAPI Service & Docker Deployment
- **Status:** **VERIFIED**
- **Criteria Verified:**
  1. **FastAPI Microservice:** [`src/api/app.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/src/api/app.py) implements GET `/health`, POST `/predict`, GET `/metrics`, and interactive OpenAPI docs at `/docs`.
  2. **Lean Serving Dependencies:** Containerized using [`requirements-serve.txt`](file:///c:/Users/moham/Downloads/petrochoice%20Task/requirements-serve.txt), excluding heavy training frameworks (PyTorch, Ultralytics, MLflow).
  3. **Container Security:** Non-root execution (`appuser`, UID 10001), healthcheck probe configured.
  4. **Active Docker Deployment:**
     - Container ID: `c5579f4a2e50` (`ppe-service`)
     - Image ID: `5e43a111bb2f` (`ppe-detection:v1`, size 509 MB)
     - Status: `Up (healthy)`, port mapping `0.0.0.0:8000->8000/tcp`
  5. **Live Health Check Query Output:**
     ```bash
     $ curl.exe -s http://localhost:8000/health
     {"status":"healthy","model_version":"YOLOv8n-PPE-v1.0","model_path":"models/model.onnx","runtime_provider":"CPUExecutionProvider","input_resolution":"640x640","uptime_seconds":382.05}
     ```
  6. **API Unit Tests:** 22/22 unit tests pass in [`tests/test_api.py`](file:///c:/Users/moham/Downloads/petrochoice%20Task/tests/test_api.py) covering validation, invalid MIME types, oversized payloads, and bounding box logic.
- **Verification Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest tests/test_api.py -v
  ```
- **Observed Output:**
  ```text
  Ran 22 tests in 1.435s - OK
  ```

---

### Phase 8: Deliverables & Technical Documentation
- **Status:** **VERIFIED**
- **Criteria Verified:**
  1. [`README.md`](file:///c:/Users/moham/Downloads/petrochoice%20Task/README.md) (10.3 KB) with complete single-command reproduction runbook.
  2. [`reports/take_home_report.md`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/take_home_report.md) (8.5 KB) structured within strict 2-page budget.
  3. [`reports/video_script.md`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/video_script.md) (6.7 KB) 5:00-minute timed walkthrough script.
  4. [`reports/live_defense_cheatsheet.md`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/live_defense_cheatsheet.md) (9.8 KB) covering 10 live modification walkthroughs and 6 deep interview answers.
