# Live Defense Cheatsheet
## PPE Safety Detection — Interview Q&A + Live Modification Guide

**Purpose:** Be ready to explain every design choice and make live code modifications with confidence.

---

## Top 10 Live Modification Requests

### MOD-1: "Change the confidence threshold"
**File:** [`src/api/app.py`](../src/api/app.py) — default in `Query(0.40, ...)`  
**Or via URL:** `POST /predict?conf_threshold=0.60`  
**Code pointer:**
```python
# app.py line ~139
conf_threshold: float = Query(
    0.40,   # ← Change this default
    ...
)
```
**Answer:** "The default 0.40 is the API default. The research-optimal threshold τ*=0.90 was selected by the cost-matrix sweep. These are separate by design — the API is flexible, the research pipeline uses the validated value from `reports/optimal_threshold_justification.json`."

---

### MOD-2: "Add a new class (e.g., 'vest')"
**Files to change:**
1. `src/models/onnx_engine.py` line 18 — add `2: "vest"` to `ID_TO_CLASS`
2. `src/data/voc_to_yolo.py` — add `"vest"` to `CLASS_MAP`
3. `data/dataset.yaml` — add `vest` to `names` list
4. Re-annotate training data with the new class
5. Re-train and re-export ONNX

**Answer:** "YOLOv8n's output head is dynamic in class count. Adding a class requires retraining — but the pipeline is fully modular. I'd add the class to `CLASS_MAP`, update `dataset.yaml`, and re-run the full pipeline from Phase 1."

---

### MOD-3: "Make the API reject images smaller than 100×100 pixels"
**File:** [`src/api/app.py`](../src/api/app.py) — add after image decoding  
**Code pointer (~line 193):**
```python
# After: img_bgr = cv2.imdecode(...)
orig_h, orig_w = img_bgr.shape[:2]
if orig_h < 100 or orig_w < 100:
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=f"Image too small ({orig_w}×{orig_h}). Minimum 100×100 px required."
    )
```

---

### MOD-4: "Explain the Pigeonhole NMS / why you use pure Python NMS"
**File:** [`src/models/onnx_engine.py`](../src/models/onnx_engine.py) lines 63–109  
**Answer:** 
> "Embedding NMS inside the ONNX graph requires the `NonMaxSuppression` ONNX operator. This operator is frequently unsupported on edge runtimes — TensorRT, OpenVINO, ONNX.js, mobile iOS/Android CoreML. By decoupling NMS into Python, our ONNX graph is purely convolution and matrix ops — universally supported. The Python NMS is also fully transparent: you can read every line during a live interview, and it can be tuned without re-exporting the graph."

---

### MOD-5: "Show me the Pigeonhole leakage proof"
**File:** [`src/data/split.py`](../src/data/split.py) — the `_pigeonhole_cluster` function  
**Theoretical explanation:**  
> "A 64-bit pHash is split into 5 chunks (sizes 13-13-13-13-12). For any two images with Hamming distance ≤ 4, at least one of the 5 chunks must agree completely — by the Pigeonhole Principle: 4 errors distributed among 5 chunks means at least one chunk has 0 errors. So any near-duplicate pair will collide in at least one bucket, guaranteed. This is a mathematical proof — not a heuristic. The result: `perceptual_hash_cross_leaks = 0` in `reports/leakage_free_split_summary.json`."

---

### MOD-6: "Run a new ablation experiment with dropout"
**Steps:**
1. Copy `configs/exp7_cosine_lr.yaml` → `configs/exp9_dropout.yaml`
2. Change `run_id: "exp9_dropout"`
3. Change `changed_variable: "dropout_0.1"`
4. Add `dropout: 0.1` under `model:` section
5. Run: `python src/models/train.py --config configs/exp9_dropout.yaml`
6. Results tracked automatically in `mlflow.db`

**Answer:** "The ablation framework in `src/models/ablation_tracker.py` reads all `configs/exp*.yaml` files and auto-populates the comparison table in `reports/tables/ablation_study.md`. One new YAML file is all that's needed."

---

### MOD-7: "Why did you use static ONNX shape (1×3×640×640)?"
**File:** [`src/models/export_onnx.py`](../src/models/export_onnx.py) line 66 — `dynamic=False`  
**Answer:**  
> "For CPU edge deployments, dynamic tensor shapes cause ONNX Runtime to re-allocate memory buffers on every call. With a static shape, the memory allocator can pre-allocate and reuse buffers across calls — reducing both latency and p95 tail variance. Our benchmark shows p50=65ms vs p95=92ms — the tail is caused by OS scheduling, not buffer allocation. For a production camera feed, static shape is the right choice."

---

### MOD-8: "Change NMS IoU threshold from 0.45 to 0.30"
**API:** `POST /predict?iou_threshold=0.30` — no code change needed  
**Code default:** [`src/api/app.py`](../src/api/app.py) line ~145  
**Engine default:** [`src/models/onnx_engine.py`](../src/models/onnx_engine.py) line ~242  
**Answer:** "Lower IoU threshold is more aggressive suppression — fewer overlapping boxes. This is useful in crowd scenes where adjacent heads have high overlap. The threshold is a runtime parameter, not baked into the model — it can be changed per-request via the query string."

---

### MOD-9: "Explain the safety compliance logic"
**File:** [`src/api/app.py`](../src/api/app.py) lines ~238–243  
**Code pointer:**
```python
if count_no_helmet > 0:
    compliance_status = ComplianceStatus.VIOLATION   # Any bare head → VIOLATION
elif count_hat > 0:
    compliance_status = ComplianceStatus.COMPLIANT   # All helmets → COMPLIANT
else:
    compliance_status = ComplianceStatus.NO_PERSONS  # Empty frame
```
**Answer:** "VIOLATION takes absolute priority. Even if 99 workers are wearing helmets but one is not — the site is non-compliant. This reflects the safety-first design principle from our cost matrix (5× FN penalty). It's better to raise one false alarm than to miss one real violation."

---

### MOD-10: "How would you scale this to multiple cameras?"
**Answer:**  
> "The current FastAPI service runs single-threaded with one ONNX engine. To scale horizontally:  
> 1. **Short term:** Run multiple Docker containers behind a load balancer (nginx or AWS ALB). Each container has its own ONNX session — no shared state.  
> 2. **Medium term:** Add a Kafka or Redis queue per camera stream. Each camera publishes frames; a worker pool consumes and runs inference.  
> 3. **Long term:** Switch to GPU inference (TensorRT or CUDA OnnxRuntime) for 10–100× throughput. The ONNX graph is already provider-agnostic — just change `CPUExecutionProvider` to `CUDAExecutionProvider` in `onnx_engine.py` line 134."

---

## Critical Q&A — Expected Hard Interview Questions

### Q: "Your threshold is 0.90, but the API default is 0.40. Why the inconsistency?"
**A:** "They serve different purposes. The research threshold τ*=0.90 was optimized on our specific validation set class distribution for the cost matrix objective. The API default 0.40 is a conservative starting point for general use — lower thresholds detect more objects at the cost of more false alarms. Any deployment team would re-run the cost sweep for their site-specific FN/FP ratio."

### Q: "How do you know the test set was never touched during development?"
**A:** "Three mechanisms: (1) `data/splits/test.txt` is OS write-locked using `os.chmod`. (2) `src/utils/guard.py` maintains a blocklist of training script names — any call from `train.py`, `tune.py`, or `val.py` to `guard_test_set_access()` raises `PermissionError`. (3) `tests/test_split.py` asserts this behaviour automatically in CI. These are not conventions — they're enforced by code."

### Q: "You said 26.96% leakage in the author's split. Show me proof."
**A:** "Open `reports/author_split_leakage.json`. Field `contaminated_test_images_count: 409`, `test_set_contamination_percentage: 26.96%`. The script that generated this is `src/data/leakage_audit.py` — it uses the same pHash pipeline and is fully reproducible."

### Q: "YOLOv8n is very small. Why not YOLOv8s or YOLOv8m?"
**A:** "The assessment specifies CPU deployment at the edge. YOLOv8n achieves 14.7 FPS on a 4-core i7 CPU at 12 MB model size. YOLOv8s is roughly 2× larger and slower — it would drop below 10 FPS, which is marginal for real-time CCTV. The architecture choice is justified by the deployment constraint, not by benchmark overfitting."

### Q: "Explain why you didn't use data augmentation more aggressively."
**A:** "Each augmentation was tested in isolation (Phases 1–7 in the ablation). The ablation shows that mosaic at probability 0.5 outperforms 1.0 — full mosaic obscures the small helmet objects we need to detect. This is an empirical finding, not an assumption. Aggressive augmentation that degrades the minority class visibility is counterproductive given our 12.33:1 imbalance."

### Q: "What's your production monitoring strategy?"
**A:** "Three layers: (1) Infra — the Docker HEALTHCHECK probe hits `/health` every 30s. (2) Application — the `processing_time_ms` field in every response enables latency SLO monitoring. (3) Model — we'd log prediction distributions over time and alert if the ratio of VIOLATION:COMPLIANT drifts significantly, which indicates concept drift or camera malfunction."

---

## Key File Locations Reference

| What you want to show | File | Lines |
|---|---|---|
| Pigeonhole clustering proof | `src/data/split.py` | ~80–130 |
| Pure Python NMS | `src/models/onnx_engine.py` | 63–109 |
| Safety cost threshold optimizer | `src/models/threshold_optimizer.py` | ~40–90 |
| Test set guard | `src/utils/guard.py` | 34–47 |
| API compliance logic | `src/api/app.py` | ~234–243 |
| ONNX export (decoupled NMS) | `src/models/export_onnx.py` | 61–68 |
| Letterbox preprocessing | `src/models/onnx_engine.py` | 24–60 |
| Benchmark protocol | `src/models/benchmark_latency.py` | ~40–80 |
| Failure root causes | `reports/tables/failure_analysis.md` | All |
| Threshold sweep data | `reports/optimal_threshold_justification.json` | All |
