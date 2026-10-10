# 5-Minute Video Walkthrough Script
## PPE Safety Detection System — AI Engineer Take-Home Assessment

**Total Duration:** 5:00 minutes  
**Target Audience:** Senior ML Engineers, Technical Leads  
**Format:** Screen recording with narration — switch between terminal, code editor, and browser  

---

## Pre-Recording Checklist

- [ ] API running: `uvicorn src.api.app:app --port 8000 --reload`
- [ ] Browser open at `http://localhost:8000/docs`
- [ ] Terminal open in project root
- [ ] VSCode open showing `src/data/split.py` and `src/models/onnx_engine.py`
- [ ] `reports/figures/` images ready
- [ ] Test image ready for live demo

---

## Section 1: Problem & Dataset [0:00 – 0:45]

**[SHOW: `reports/data_audit_summary.json` + `reports/figures/class_distribution.png`]**

> "The task is industrial PPE compliance — detecting whether construction workers are wearing safety helmets in real-world images.

> The dataset is SHWD: 7,581 images, 120,558 annotated bounding boxes, two classes: `hat` for a worn helmet, and `no_helmet` for a bare head.

> The first thing I did was run a full data audit — and the numbers reveal the real challenge. Class imbalance is severe: only 7.5% of boxes are `hat`, while 92.5% are `no_helmet` — a 12-to-1 ratio. Worse, 69.6% of all objects are COCO-small, meaning smaller than 32×32 pixels. Crowd scenes hit 162 people per frame.

> These aren't academic constraints — they determine every design decision I made."

---

## Section 2: The Leakage Problem [0:45 – 1:30]

**[SHOW: `reports/author_split_leakage.json`, then `src/data/split.py` lines 1–50]**

> "Before training anything, I audited the dataset's original train/test split — and found a critical issue. 409 out of 1,517 test images — that's 26.96% — had perceptual near-duplicates already in the training set.

> This is because the dataset was collected from video. Adjacent frames are almost identical. A random split doesn't respect that.

> I rebuilt the split from scratch using a technique I call Pigeonhole Multi-Index pHash Clustering.

> Here's how it works." 

**[SWITCH TO: `src/data/split.py`, scroll to the clustering function]**

> "I compute a 64-bit perceptual hash for every image, then partition that hash into 5 chunks. By the Pigeonhole Principle, any two images with Hamming distance ≤ 4 *must* share at least one chunk exactly — guaranteeing we find every near-duplicate pair with zero false negatives.

> We then merge those groups and perform a group-aware stratified split. The result: zero perceptual leaks. The test set is fully isolated, OS write-locked, and gated by a guard module."

---

## Section 3: Ablation Experiments [1:30 – 2:30]

**[SHOW: `reports/tables/ablation_study.md`, then `configs/exp8_best_combo.yaml`]**

> "I ran 8 single-variable controlled experiments, changing exactly one hyperparameter per run to measure its isolated impact. This is critical — changing multiple things at once makes it impossible to know what worked.

> The tested variables were: scale jitter, mosaic probability, HSV saturation, class loss gain, minority oversampling, input resolution, and learning rate scheduler.

> The best composite configuration — exp8 — combines the validated wins: scale jitter 0.7, mosaic 0.5, cosine LR decay, class loss gain 1.5, and input size 800 for better small-object recall.

> All experiments were tracked in MLflow with SQLite — reproducible by seed, config, and checkpoint."

---

## Section 4: Threshold Justification [2:30 – 3:00]

**[SHOW: `reports/figures/threshold_vs_cost_curve.png`, then `reports/optimal_threshold_justification.json`]**

> "Standard metrics like F1 use equal penalties for false positives and false negatives. In safety engineering, that's wrong.

> A missed unprotected worker — a false negative — can result in serious injury. A false alarm just interrupts operations briefly.

> I modelled this as a cost matrix: FN penalty is 5×, FP penalty is 1×. I swept confidence thresholds from 0.1 to 0.9 on the validation set and minimized total operational cost.

> The optimal threshold is τ* = 0.90, which eliminates false alarms entirely while keeping violations at their floor."

---

## Section 5: ONNX Deployment & API Demo [3:00 – 4:30]

**[SWITCH TO: `src/models/onnx_engine.py` lines 24–109]**

> "The model is exported to ONNX Runtime for CPU inference. Crucially, NMS is NOT embedded in the ONNX graph. I keep preprocessing — the letterbox resize — and postprocessing — the pure Python greedy NMS — completely outside the graph.

> Why? Because embedded NMS operators break on edge runtimes — TensorRT, OpenVINO, mobile. Decoupled NMS is auditable, modifiable live, and portable.

**[SHOW: terminal, run `python src/models/benchmark_latency.py`]**

> "On an Intel i7-8565U with 4 threads, end-to-end latency is 68 milliseconds — 14.7 frames per second. More than sufficient for real-time CCTV monitoring."

**[SWITCH TO: browser at `http://localhost:8000/docs`]**

> "The model is served via FastAPI. I have two endpoints. GET `/health` returns model version, runtime, and uptime. POST `/predict` accepts an image, runs inference, and returns structured detections with a site safety classification: COMPLIANT, VIOLATION, or NO_PERSONS.

> The VIOLATION flag triggers if even a single bare head is detected — consistent with our safety-first threshold design.

**[LIVE DEMO: Upload a test image via Swagger UI]**

> "Here's a live prediction — you can see the bounding boxes, confidence scores, class names, and compliance status returned in a typed Pydantic schema."

---

## Section 6: Test Suite & Wrap-Up [4:30 – 5:00]

**[SHOW: terminal, run `python -m unittest discover -s tests`]**

> "The entire system is covered by 58 automated unit and integration tests — phases 0 through 7. Every pipeline step from the VOC converter to the API validation logic is tested. All 58 pass.

> To summarize: I found and fixed a critical 27% data leakage in the original split. I built a reproducible 8-run ablation framework. I implemented a safety-cost threshold optimizer. And I deployed a 14.7 FPS ONNX CPU inference API in a non-root Docker container.

> Every number in the report comes directly from a reproducible script. The code is ready for live modification. Thank you."

---

## Timing Reference

| Section | Start | End | Duration |
|---|---|---|---|
| Problem & Dataset | 0:00 | 0:45 | 45 s |
| Leakage Problem | 0:45 | 1:30 | 45 s |
| Ablation Experiments | 1:30 | 2:30 | 60 s |
| Threshold Justification | 2:30 | 3:00 | 30 s |
| ONNX Deployment & Demo | 3:00 | 4:30 | 90 s |
| Test Suite & Wrap-Up | 4:30 | 5:00 | 30 s |
| **Total** | | | **5:00** |
