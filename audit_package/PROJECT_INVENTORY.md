# Project Inventory: PPE Safety Detection System

**Repository Root:** `c:\Users\moham\Downloads\petrochoice Task`  
**Audit Timestamp:** October 2026  
**Auditor:** Automated Engineering Audit Suite  

---

## 1. Directory Tree Overview

```
c:\Users\moham\Downloads\petrochoice Task\
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── requirements-serve.txt
├── yolov8n.pt
├── mlflow.db
│
├── configs/
│   ├── base.yaml
│   ├── exp0_baseline.yaml
│   ├── exp0_baseline_seed1337.yaml
│   ├── exp1_scale.yaml
│   ├── exp2_mosaic.yaml
│   ├── exp3_hsv.yaml
│   ├── exp4_cls_gain.yaml
│   ├── exp5_oversample.yaml
│   ├── exp6_imgsz_800.yaml
│   ├── exp7_cosine_lr.yaml
│   └── exp8_best_combo.yaml
│
├── data/
│   ├── dataset.yaml
│   ├── dataset_oversampled.yaml
│   ├── raw/
│   │   ├── SHWD.zip
│   │   └── VOC2028/
│   │       ├── Annotations/       (7,581 XML files)
│   │       ├── ImageSets/Main/
│   │       └── JPEGImages/        (7,581 JPG files)
│   ├── processed/
│   │   └── labels/                (7,581 YOLO .txt files)
│   └── splits/
│       ├── train.txt              (4,964 image paths)
│       ├── val.txt                (1,264 image paths)
│       ├── test.txt               (1,353 image paths, OS Read-Only locked)
│       └── train_oversampled.txt  (7,207 image paths)
│
├── models/
│   └── model.onnx                 (12.85 MB, ONNX opset 17, FP32)
│
├── reports/
│   ├── author_split_leakage.json
│   ├── data_audit_summary.json
│   ├── leakage_free_split_summary.json
│   ├── optimal_threshold_justification.json
│   ├── take_home_report.md
│   ├── video_script.md
│   ├── live_defense_cheatsheet.md
│   ├── failures/                  (15 visual failure case images: failure_01.jpg - failure_15.jpg)
│   ├── figures/
│   │   ├── audit_visual_sample_grid.png
│   │   ├── box_size_distribution.png
│   │   ├── class_distribution.png
│   │   └── threshold_vs_cost_curve.png
│   └── tables/
│       ├── ablation_study.csv
│       ├── ablation_study.md
│       ├── failure_analysis.csv
│       ├── failure_analysis.md
│       ├── latency_benchmark.json
│       └── latency_benchmark.md
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                 (FastAPI microservice)
│   │   └── schemas.py             (Pydantic request/response schemas)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audit.py               (EDA & corruption audit)
│   │   ├── download.py            (Automated gdown dataset fetcher)
│   │   ├── leakage_audit.py       (Perceptual hash cross-split audit)
│   │   ├── oversample.py          (Minority-class hard oversampling)
│   │   ├── split.py               (Zero-leakage stratified split generator)
│   │   └── voc_to_yolo.py         (XML-to-YOLO conversion & validation)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ablation_tracker.py    (Matrix comparison & table compiler)
│   │   ├── benchmark_latency.py   (Benchmarking protocol: warmups, p50/p95)
│   │   ├── evaluate.py            (COCO evaluator & confusion matrix)
│   │   ├── export_onnx.py         (ONNX export & dynamic shape verification)
│   │   ├── failure_analysis.py    (Top-15 error taxonomy & cropping)
│   │   ├── noise_floor.py         (Multi-seed variance estimator)
│   │   ├── onnx_engine.py         (Pure Python letterbox, NMS & ONNX session)
│   │   ├── threshold_optimizer.py (Asymmetric safety cost matrix optimizer)
│   │   └── train.py               (MLflow-instrumented YOLO training)
│   └── utils/
│       ├── __init__.py
│       ├── config.py              (Pydantic config schema & parser)
│       ├── guard.py               (Test set OS lock & caller gatekeeper)
│       ├── seed.py                (Deterministic seed controller)
│       └── tracking.py            (MLflow experiment setup & autologging)
│
└── tests/
    ├── test_ablation_pipeline.py  (6 tests)
    ├── test_api.py                (22 tests)
    ├── test_audit.py              (4 tests)
    ├── test_data.py               (4 tests)
    ├── test_evaluation_pipeline.py(4 tests)
    ├── test_onnx_parity.py        (4 tests)
    ├── test_split.py              (12 tests)
    └── test_train_pipeline.py     (2 tests)
```

---

## 2. Key File Descriptions

| Category | File | Size | Role / Content Description |
| :--- | :--- | :--- | :--- |
| **Orchestration** | `Dockerfile` | 1,878 B | Multi-stage production container running non-root `appuser` (UID 10001) with built-in curl healthcheck. |
| **Dependencies** | `requirements.txt` | 431 B | Full training and research dependencies (PyTorch, Ultralytics, MLflow, Scikit-learn). |
| **Dependencies** | `requirements-serve.txt` | 705 B | Lean runtime dependencies (ONNX Runtime, FastAPI, OpenCV-headless, Pillow, Pydantic). Reduced image from >3 GB to 509 MB. |
| **Documentation** | `README.md` | 10,295 B | Master project documentation, external dataset URL, runbook, and verification commands. |
| **Model Weights** | `yolov8n.pt` | 6.55 MB | Pretrained YOLOv8 nano PyTorch checkpoint. |
| **Model Weights** | `models/model.onnx` | 12.85 MB | Exported FP32 ONNX model (opset 17, dynamic batch). |
| **Database** | `mlflow.db` | 954 KB | SQLite database backing MLflow experiment tracking. |
| **Dataset Splits** | `data/splits/test.txt` | 51.7 KB | Untouched test split (1,353 images). Write-protected with OS read-only attribute (`-ar---`). |
| **Audit Data** | `reports/data_audit_summary.json` | 751 B | Comprehensive data statistics: 7,581 images, 120,558 boxes, 12.33:1 imbalance ratio. |
| **Audit Data** | `reports/author_split_leakage.json` | 1,595 B | Documentation of 409 leaked test images (26.96% contamination) in the original author split. |
| **Audit Data** | `reports/leakage_free_split_summary.json` | 791 B | Mathematical proof of zero perceptual/stem overlap across train (4,964), val (1,264), and test (1,353). |
| **Evaluation** | `reports/optimal_threshold_justification.json` | 5,183 B | Validation threshold cost sweep under asymmetric safety penalty ($C_{FN}=5.0, C_{FP}=1.0$). |
| **Benchmark** | `reports/tables/latency_benchmark.json` | 711 B | CPU latency measurements: 68.12 ms end-to-end, 58.24 ms ONNX forward pass, 14.7 FPS on Intel i7. |
