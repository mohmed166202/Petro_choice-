# PPE Safety Detection System
> **End-to-end industrial helmet compliance detector — 3-day AI Engineer take-home assessment.**
> Model: YOLOv8n · Dataset: SHWD (7,581 images) · Deployment: FastAPI + ONNX Runtime + Docker

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Quick Start — Local Setup](#quick-start--local-setup)
3. [Dataset Download](#dataset-download)
4. [Running the Full Pipeline](#running-the-full-pipeline)
5. [Serving the API](#serving-the-api)
6. [Running Tests](#running-tests)
7. [Colab GPU Training Runbook](#colab-gpu-training-runbook)
8. [Key Findings](#key-findings)
9. [License](#license)

---

## Project Structure

```
├── configs/                    # One YAML per ablation experiment
│   ├── base.yaml               # Shared defaults (seeds, paths, MLflow URI)
│   ├── exp0_baseline.yaml      # Seed 42 baseline
│   ├── exp0_baseline_seed1337.yaml # Seed 1337 noise-floor check
│   ├── exp1_scale.yaml → exp8_best_combo.yaml  # 8 single-variable ablations
│
├── data/
│   ├── splits/
│   │   ├── train.txt           # 4,964 training images (group-stratified)
│   │   ├── val.txt             # 1,264 validation images
│   │   └── test.txt            # 1,353 test images (WRITE-LOCKED — never touch before Phase 5)
│   ├── dataset.yaml            # YOLO manifest (train/val only)
│   └── processed/              # YOLO-format labels (converted from Pascal VOC XML)
│
├── docs/
│   └── colab_runbook.md        # Step-by-step Colab GPU training guide
│
├── models/
│   └── model.onnx              # Exported ONNX model (12.26 MB, Opset 17, 640×640)
│
├── reports/
│   ├── data_audit_summary.json
│   ├── author_split_leakage.json
│   ├── leakage_free_split_summary.json
│   ├── optimal_threshold_justification.json
│   ├── figures/                # Audit plots, threshold cost curve
│   ├── failures/               # 15 annotated failure images
│   └── tables/                 # ablation_study.md, failure_analysis.md, latency_benchmark.md
│
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI service (/health, /predict)
│   │   └── schemas.py          # Pydantic v2 DTOs
│   ├── data/
│   │   ├── download.py         # Resumable gdown downloader
│   │   ├── audit.py            # Data quality audit
│   │   ├── voc_to_yolo.py      # Lossless Pascal VOC → YOLO converter
│   │   ├── split.py            # Pigeonhole pHash clustering + stratified split
│   │   ├── leakage_audit.py    # Author split contamination audit
│   │   └── oversample.py       # Minority class 2× oversampler
│   ├── models/
│   │   ├── train.py            # Training entry point
│   │   ├── noise_floor.py      # Baseline noise-floor analysis
│   │   ├── ablation_tracker.py # Multi-run MLflow experiment tracker
│   │   ├── threshold_optimizer.py  # Safety cost matrix threshold sweep
│   │   ├── evaluate.py         # Final held-out test evaluation
│   │   ├── failure_analysis.py # 15 systematic failure case annotator
│   │   ├── onnx_engine.py      # Decoupled letterbox + ONNX RT + pure Python NMS
│   │   ├── export_onnx.py      # PyTorch → ONNX exporter (Opset 17)
│   │   └── benchmark_latency.py # CPU latency profiler (200 iterations)
│   └── utils/
│       ├── seed.py             # Global seed controller
│       ├── config.py           # YAML config loader
│       ├── tracking.py         # MLflow SQLite tracker
│       └── guard.py            # Test set leakage gatekeeper
│
├── tests/                      # 58 unit & integration tests (100% passing)
│   ├── test_data.py
│   ├── test_audit.py
│   ├── test_split.py
│   ├── test_train_pipeline.py
│   ├── test_ablation_pipeline.py
│   ├── test_evaluation_pipeline.py
│   ├── test_onnx_parity.py
│   └── test_api.py
│
├── Dockerfile                  # python:3.11-slim, non-root user, HEALTHCHECK
├── .dockerignore               # Excludes raw data, .pt weights, mlruns
├── requirements.txt            # Pinned dependencies
└── mlflow.db                   # Local SQLite experiment tracker
```

---

## Quick Start — Local Setup

**Prerequisites:** Python 3.10–3.12, Git, ~3 GB free disk (dataset), Docker (optional for serving).

```bash
# 1. Clone the repository
git clone <YOUR_REPO_URL>
cd ppe-detection

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Dataset Download

The SHWD dataset (7,581 images, ~1.12 GB) is hosted externally.

**Option A — Automated (recommended):**
```bash
python src/data/download.py
# Resumes automatically if interrupted. Verifies 7,581 images on completion.
```

**Option B — Manual fallback (if Google Drive throttles):**
1. Download `SHWD.zip` from: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
2. Place the zip in the project root.
3. Re-run `python src/data/download.py` — it will skip download and proceed to extraction.

---

## Running the Full Pipeline

Each phase is a standalone script. Run them **in order**:

```bash
# Phase 1 — Data ingestion & lossless conversion (Pascal VOC XML → YOLO)
python src/data/voc_to_yolo.py

# Phase 1 — Data quality audit (generates reports/data_audit_summary.json)
python src/data/audit.py

# Phase 2 — Leakage audit on author's original split
python src/data/leakage_audit.py

# Phase 2 — Zero-leakage stratified split (Pigeonhole pHash clustering)
python src/data/split.py

# Phase 3 — Baseline training (run on Colab GPU — see docs/colab_runbook.md)
python src/models/train.py --config configs/exp0_baseline.yaml

# Phase 3 — Noise floor check (seed 1337)
python src/models/train.py --config configs/exp0_baseline_seed1337.yaml

# Phase 4 — Ablation runs (run each on Colab GPU, one at a time)
python src/models/train.py --config configs/exp1_scale.yaml
# ... repeat for exp2 through exp8

# Phase 4 — Compile ablation comparison table
python src/models/ablation_tracker.py

# Phase 5 — Threshold optimization (safety cost matrix)
python src/models/threshold_optimizer.py --weights models/exp8_best_combo/weights/best.pt

# Phase 5 — Final test evaluation (NEVER run before threshold is set)
python src/models/evaluate.py --weights models/exp8_best_combo/weights/best.pt

# Phase 5 — 15 failure case analysis
python src/models/failure_analysis.py --weights models/exp8_best_combo/weights/best.pt

# Phase 6 — Export to ONNX
python src/models/export_onnx.py --weights models/exp8_best_combo/weights/best.pt

# Phase 6 — CPU latency benchmark
python src/models/benchmark_latency.py
```

---

## Serving the API

### Local (Development)
```bash
.\.venv\Scripts\uvicorn.exe src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- **Interactive docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Predict:** `POST http://localhost:8000/predict` (multipart/form-data, `file` field)

**Example curl request:**
```bash
curl -X POST "http://localhost:8000/predict?conf_threshold=0.40" \
     -H "accept: application/json" \
     -F "file=@your_image.jpg"
```

### Docker (Production)
```bash
# Build image (~1.5 GB, python:3.11-slim base)
docker build -t ppe-detection:v1 .

# Run container (non-root user, port 8000)
docker run -p 8000:8000 ppe-detection:v1

# Check health
curl http://localhost:8000/health
```

---

## Running Tests

```bash
# Run full test suite (58 tests across all phases)
python -m unittest discover -s tests -p "test_*.py" -v

# Run individual phase tests
python -m unittest tests/test_data.py           # Phase 1
python -m unittest tests/test_split.py          # Phase 2
python -m unittest tests/test_train_pipeline.py # Phase 3
python -m unittest tests/test_ablation_pipeline.py  # Phase 4
python -m unittest tests/test_evaluation_pipeline.py # Phase 5
python -m unittest tests/test_onnx_parity.py   # Phase 6
python -m unittest tests/test_api.py            # Phase 7
```

**Expected result:** `Ran 58 tests in ~22s — OK`

---

## Colab GPU Training Runbook

See [`docs/colab_runbook.md`](docs/colab_runbook.md) for the full step-by-step guide.

**Quick summary:**
1. Upload repo to Google Drive or clone on Colab.
2. Mount `/content/drive`, install requirements.
3. Run `python src/models/train.py --config configs/exp0_baseline.yaml`.
4. Download `runs/` folder and `models/` back to local machine.

---

## Key Findings

| Metric | Value |
|---|---|
| Dataset size | 7,581 images · 120,558 bounding boxes |
| Class imbalance | `hat`: 7.5% · `no_helmet`: 92.5% (ratio **12.33:1**) |
| Object scale | **69.6% COCO-small** (<32×32 px) — primary detection challenge |
| Author split leakage | **26.96%** of test images contaminated (409/1,517) |
| Our split leakage | **0%** — verified by Pigeonhole Multi-Index pHash clustering |
| Split sizes | Train: 4,964 · Val: 1,264 · Test: 1,353 |
| ONNX model size | **12.26 MB** (Opset 17, Static 1×3×640×640) |
| CPU throughput | **14.7 FPS** end-to-end · **17.2 FPS** pure inference (i7-8565U, 4 threads) |
| Optimal threshold | **τ* = 0.90** (5× FN cost safety matrix, val sweep) |
| Test suite | **58 tests — 100% passing** |

---

## License

- **Code:** MIT License — see `LICENSE`.
- **Dataset (SHWD):** MIT License — original source: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
- **Model Architecture (YOLOv8n):** AGPL-3.0 — Ultralytics. Commercial use requires a Ultralytics Enterprise license.
  > ⚠️ **Note:** AGPL-3.0 requires that any modification to the YOLOv8 source code be published as open-source if distributed. The exported ONNX weights themselves are derivative works and are subject to the same licence. This is noted as a limitation in the project report.
