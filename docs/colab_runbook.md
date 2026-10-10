# Google Colab / Kaggle T4 Training Runbook

This guide enables running the model training runs (Baseline + Ablations 1 to 4) on a free NVIDIA T4 GPU in Google Colab (~15 minutes per 30-epoch run) while keeping data preparation, local evaluation, ONNX export, and Docker serving on your local machine.

---

## 1. Setup in Colab Notebook

Create a new notebook in Google Colab, set runtime type to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`), and run:

```bash
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Clone your repository or upload code
!git clone <YOUR_GITHUB_REPO_URL> ppe_detector
%cd ppe_detector

# Cell 3: Install pinned requirements
!pip install -r requirements.txt
```

---

## 2. Mount Processed Data & Splits

If you keep your processed data in Google Drive:

```python
# Cell 4: Mount Drive and symlink or copy processed dataset
from google.colab import drive
drive.mount('/content/drive')

# Copy or unpack processed dataset
!cp -r /content/drive/MyDrive/ppe_data/processed data/processed
!cp -r /content/drive/MyDrive/ppe_data/splits data/splits
```

*(Alternatively, run `python src/data/download.py` and `python src/data/voc_to_yolo.py` directly inside Colab if starting fresh).*

---

## 3. Run Training (Baseline or Ablations)

```bash
# Cell 5: Train Baseline with MLflow tracking
python src/models/train.py --config configs/exp0_baseline.yaml

# Cell 6: Train Baseline Seed Sensitivity Check (Seed 1337)
python src/models/train.py --config configs/exp0_baseline_seed1337.yaml

# Cell 8: Controlled Ablations (1 variable per run, priority order)
python src/models/train.py --config configs/exp1_scale.yaml
python src/models/train.py --config configs/exp2_mosaic.yaml
python src/models/train.py --config configs/exp3_hsv.yaml
python src/models/train.py --config configs/exp4_cls_gain.yaml
python src/models/train.py --config configs/exp5_oversample.yaml
python src/models/train.py --config configs/exp6_imgsz_800.yaml
python src/models/train.py --config configs/exp7_cosine_lr.yaml

# Cell 9: Train Composite Winning Configuration
python src/models/train.py --config configs/exp8_best_combo.yaml

# Cell 10: Compile Ablation Study Table & Select Best Model
python src/models/ablation_tracker.py
```

---

## 4. Download Weights and MLflow Logs

Once training completes, zip the trained model weights and local MLflow SQLite database to copy back to your local environment:

```bash
# Cell 8: Package best weights and MLflow logs
!zip -r training_results.zip models/ mlflow.db mlruns/

# Cell 9: Download to your local machine
from google.colab import files
files.download('training_results.zip')
```

Unzip `training_results.zip` in your local project directory. You can then immediately run:
```bash
# Inspect experiment comparisons locally
mlflow ui

# Run local CPU evaluation, ONNX export, and Docker microservice
python src/models/evaluate.py
python src/models/export_onnx.py
docker build -t ppe-detector:latest .
```
