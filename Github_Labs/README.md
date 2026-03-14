# GitHub Actions: ML Model Training & Calibration

This lab demonstrates how to automate machine learning model training, evaluation, versioning, and calibration using GitHub Actions.

## Project Overview

A **Gradient Boosting Classifier** is trained on sklearn's **Wine dataset** (178 samples, 13 features, 3 classes). Every push to `main` automatically triggers two workflows, one for training and one for calibration, storing versioned models and metrics in the repository.

---

## Repo Structure

```
MLOps_Labs/
├── .github/
│   └── workflows/
│       ├── model_retraining_on_push.yml    ← trains & evaluates model
│       └── model_calibration_on_push.yml   ← calibrates trained model
└── Github_Labs/
    ├── src/
    │   ├── train_model.py                  ← trains GBC, saves model + test data
    │   ├── evaluate_model.py               ← evaluates on held-out test set
    │   └── calibrate_model.py              ← applies Platt/isotonic calibration
    ├── models/                             ← versioned model files (auto-created)
    ├── metrics/                            ← JSON metric files (auto-created)
    ├── data/                               ← saved train/test splits (auto-created)
    ├── requirements.txt
    └── README.md
```

---

## How to Run Locally

**Prerequisites:** Python 3.9+, pip

```bash
# Install dependencies
pip install -r Github_Labs/requirements.txt

# Train model
cd Github_Labs
python src/train_model.py --timestamp 20240101120000

# Evaluate model
python src/evaluate_model.py --timestamp 20240101120000

# Calibrate model
python src/calibrate_model.py --timestamp 20240101120000
```

---

## Workflow Explanation

### 1. `model_retraining_on_push.yml`
Triggers on every push to `main` and runs the following steps:
1. Generates a timestamp for versioning
2. Trains a GBC model on the Wine dataset
3. Evaluates on a held-out test set
4. Commits versioned model and metrics back to the repo

### 2. `model_calibration_on_push.yml`
Also triggers on push to `main` and:
1. Retrains the base model
2. Tries both **Platt scaling** (sigmoid) and **isotonic regression**
3. Saves the better-performing calibrated model

---

## Sample Output

**`metrics/20240101120000_metrics.json`**
```json
{
    "timestamp": "20240101120000",
    "model": "GradientBoostingClassifier",
    "dataset": "Wine",
    "test_samples": 36,
    "accuracy": 0.9722,
    "f1_macro": 0.9725,
    "precision_macro": 0.9762,
    "recall_macro": 0.9697
}
```

**`metrics/20240101120000_calibration_metrics.json`**
```json
{
    "method": "isotonic",
    "accuracy": 1.0,
    "f1_macro": 1.0,
    "precision_macro": 1.0,
    "recall_macro": 1.0
}
```

---

## Dependencies
- `scikit-learn` — model training, calibration, metrics
- `mlflow` — experiment tracking
- `joblib` — model serialization
- `numpy`, `pandas` — data handling
 
