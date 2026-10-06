# Wi-Fi CSI-Based Human Activity Recognition: Cross-User Generalization Experiment

This repository contains the complete research experimental framework for studying **Cross-User Generalization** in Wi-Fi CSI-based Human Activity Recognition (HAR) using PyTorch.

---

## 📌 Project Overview
- **Research Question**: *"How well do CNN, LSTM and CNN-LSTM models recognize Wi-Fi CSI-based human activities when the test users were not present during training?"*
- **Dataset**: WiAR Wi-Fi CSI Dataset (10 subjects, 16 activity classes).
- **Core Contribution**: Controlled empirical comparison between **Random Split** (Seen Users) and **Unseen-User Split** (Cross-User Evaluation) to quantify the exact generalization drop.

---

## 📁 Repository Structure
```
wifi_csi_experiment/
│
├── DATASET_REPORT.md         # Comprehensive dataset structure audit
├── REPORT.md                 # Research report draft & empirical findings
├── README.md                 # Reproducibility guide
├── requirements.txt          # Python dependencies
├── run_experiment.py         # Main experiment CLI runner
│
├── configs/
│   └── config.yaml           # Experiment settings & hyperparameters
│
├── src/
│   ├── inspect_dataset.py    # Dataset inspection & report generator
│   ├── csi_parser.py         # Intel 5300 CSI parser module
│   ├── preprocessing.py      # Noise filtering, normalization, interpolation
│   ├── create_dataset.py     # Preprocessing subset generator & metadata builder
│   ├── split_by_user.py      # Strict unseen-user vs. random splitter
│   ├── dataset.py            # PyTorch CSIDataset loader
│   ├── train.py              # PyTorch training engine
│   ├── evaluate.py           # Evaluation & metrics calculator
│   ├── visualize.py          # Figure generator (curves, bars, confusion matrices)
│   │
│   └── models/
│       ├── cnn.py            # 1D CNN architecture
│       ├── lstm.py           # LSTM sequence model
│       └── cnn_lstm.py       # Hybrid CNN-LSTM architecture
│
└── results/
    ├── models/               # Model checkpoints (.pth)
    ├── figures/              # Publication-quality charts & curves
    └── results.csv           # Quantitative metric results table
```

---

## 🚀 Step-by-Step Instructions (Windows PowerShell)

### 1. Set Up Environment
```powershell
cd c:\Users\manishkumar.COMPILERX\Downloads\WiAR-master\wifi_csi_experiment

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Inspect Dataset Structure
```powershell
python src/inspect_dataset.py
```
Outputs `DATASET_REPORT.md` and displays subject counts, sample counts, and packet shapes.

### 3. Test CSI Parser & Pipeline Validation
```powershell
python src/test_pipeline.py
```
Validates CSI parsing, preprocessing range, and model forward passes.

### 4. Run Full Experiment Suite
Execute both Random Split and Unseen-User Split across CNN, LSTM, and CNN-LSTM models:
```powershell
python run_experiment.py --epochs 20
```

To run a single model or specific experiment protocol:
```powershell
# Run only Unseen-User experiment for CNN
python run_experiment.py --experiment unseen_user --model cnn --epochs 20

# Run only Random Split experiment for all models
python run_experiment.py --experiment random --model all --epochs 20
```

---

## 📊 Experimental Results & Outputs
Running `run_experiment.py` automatically produces:
1. `results/results.csv`: Contains Accuracy, Precision, Recall, Macro F1, Weighted F1, Training Time, and Inference Time.
2. `results/figures/random_vs_unseen_accuracy.png`: Side-by-side accuracy comparison chart.
3. `results/figures/performance_drop_comparison.png`: Absolute and percentage performance drop chart.
4. `results/figures/training_curves_*.png`: Loss and accuracy curves per epoch.
5. `results/figures/cm_*.png`: Confusion matrices for each model.

---

## 🔬 Citation & Reference Notice
The primary reference paper (*"CSI-Based Human Activity Recognition Using Low-Cost Wi-Fi Modules and Deep Neural Networks"*, IEEE ISCIT 2025) serves as literature background only. All empirical results reported here are independently generated through our controlled experimental pipeline.
