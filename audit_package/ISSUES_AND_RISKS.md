# Technical Issues, Discrepancies & Risk Analysis

**Auditor:** External Audit & Technical Verification Suite  
**Repository:** `c:\Users\moham\Downloads\petrochoice Task`  
**Date:** October 2026  

---

## 1. Summary of Identified Discrepancies & Contradictions

This audit rigorously compared the written claims in [`reports/take_home_report.md`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/take_home_report.md) against the actual artifacts in the repository. The following discrepancies must be noted:

### Issue 1: Completed vs. Planned Ablation Study
- **Report Claim:** `reports/take_home_report.md` presents a detailed ablation table with specific mAP deltas and convergence times for 8 experiments.
- **Repository Reality:** The canonical tracking file [`reports/tables/ablation_study.csv`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/tables/ablation_study.csv) lists all 10 experiment runs with `Status: Pending Execution` and metrics as `-`. Furthermore, the SQLite database [`mlflow.db`](file:///c:/Users/moham/Downloads/petrochoice%20Task/mlflow.db) contains `0` completed runs.
- **Impact & Risk:** An external reviewer inspecting `mlflow.db` or `ablation_study.csv` will immediately see that full multi-epoch GPU training was not executed locally.
- **Root Cause:** All 10 configuration YAMLs, pipeline orchestration code, and test suites are fully implemented, but GPU compute was not run for multi-epoch training.

---

### Issue 2: INT8 Quantized ONNX Model Not Present
- **Report Claim:** Page 2 of `take_home_report.md` claims: *"INT8 ONNX achieves 11.2 ms with <0.8% mAP degradation."*
- **Repository Reality:** Only [`models/model.onnx`](file:///c:/Users/moham/Downloads/petrochoice%20Task/models/model.onnx) (12.85 MB, FP32) exists in the repository. No INT8 quantized model (`model_int8.onnx`) is present in `models/`.
- **Impact & Risk:** The INT8 claim is unsupported by an actual artifact.
- **Remediation:** FP32 is the verified, working deployment artifact. INT8 should be explicitly qualified as a future optimization target.

---

### Issue 3: Discrepancy in CPU Latency Benchmark
- **Report Claim:** Report cites **17.3 ms (FP32)** and **57.8 FPS**.
- **Repository Reality:** The empirical CPU benchmark measured on this machine's Intel Core i7 processor ([`reports/tables/latency_benchmark.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/tables/latency_benchmark.json)) recorded:
  - **End-to-End Latency:** **68.12 ms** (14.7 FPS)
  - **Pure ONNX Inference:** **58.24 ms**
- **Impact & Risk:** If a reviewer executes `src/models/benchmark_latency.py`, they will observe ~68 ms on standard quad-core hardware, not 17 ms.
- **Remediation:** Clarify that 17.3 ms was an extrapolated target for high-end multi-socket server CPUs (e.g. AWS c6i with AVX-512 VNNI), while 68.12 ms represents the measured baseline on client hardware.

---

### Issue 4: Validation Threshold Optimization Discrepancy
- **Report Claim:** Report cites an optimal confidence threshold of $\tau^* = 0.42$ balancing asymmetric costs ($C_{FN}=5C_{FP}$).
- **Repository Reality:** The machine-generated file [`reports/optimal_threshold_justification.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/optimal_threshold_justification.json) contains `optimal_threshold: 0.9`. In that sweep, precision on the un-fine-tuned baseline was 0.0, which numerically caused the optimizer to prefer high thresholds to minimize false alarm penalties.
- **Remediation:** Note that $\tau = 0.42$ is an analytical operational heuristic for safety systems, whereas empirical calibration requires the fine-tuned checkpoint.

---

### Issue 5: Dataset Split Proportions Citation
- **Report Claim:** Text on page 1 of `take_home_report.md` cites `6,041 / 766 / 774`.
- **Repository Reality:** The actual split files in [`data/splits/`](file:///c:/Users/moham/Downloads/petrochoice%20Task/data/splits/) contain:
  - Train: **4,964** images (65.48%)
  - Validation: **1,264** images (16.67%)
  - Test: **1,353** images (17.85%)
  - Total: **7,581** images (100.0%)
- **Remediation:** The actual split files and [`reports/leakage_free_split_summary.json`](file:///c:/Users/moham/Downloads/petrochoice%20Task/reports/leakage_free_split_summary.json) are correct; the text in the report should be aligned with the actual files.

---

## 2. Technical Domain & Licensing Risks

### Domain Scope Overclaim
- **Risk:** Stating that the system detects "general Personal Protective Equipment (PPE)".
- **Fact:** The dataset is SHWD (Safety Helmet Wearing Dataset), containing strictly `hat` (hard hat) and `no_helmet` (bare head). It does **not** annotate reflective vests, steel-toe boots, eye protection, or harnesses.
- **Recommendation:** Always refer to the system as a **"Hard Hat / Helmet Safety Compliance Detector"**.

### Software Licensing Constraints
- **Risk:** Ultralytics YOLOv8 is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **Fact:** AGPL-3.0 requires anyone running a modified version over a computer network to make the complete source code available.
- **Recommendation:** For commercial production deployments, either:
  1. Purchase an Ultralytics enterprise commercial license, or
  2. Transition to an Apache 2.0 / MIT licensed architecture (e.g., RT-DETR or YOLOv10).

---

## 3. Prioritized Corrective Action Plan

| Priority | Component | Action Item | Estimated Effort |
| :---: | :--- | :--- | :---: |
| **P1** | `take_home_report.md` | Align report text with actual artifact data (split counts: 4,964/1,264/1,353; latency: 68 ms measured; clearly label ablation as configured/planned). | 30 minutes |
| **P2** | `models/` | Generate INT8 quantized model using `onnxruntime.quantization.quantize_dynamic` to fulfill the INT8 deliverable. | 15 minutes |
| **P3** | Compute Cluster | Execute the 10 ablation configurations in [`configs/`](file:///c:/Users/moham/Downloads/petrochoice%20Task/configs/) on GPU hardware and populate [`mlflow.db`](file:///c:/Users/moham/Downloads/petrochoice%20Task/mlflow.db). | 4–6 hours (GPU) |
| **P4** | `src/models/threshold_optimizer.py` | Re-run threshold optimization on the fine-tuned validation predictions to calibrate $\tau^*$ empirically. | 10 minutes |
