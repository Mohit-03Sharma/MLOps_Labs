# Diabetes Regression with scikit-learn (Docker Labs)

This small lab trains a scikit-learn regression pipeline on the built-in Diabetes dataset and demonstrates how to run the training both locally and inside Docker. The script saves a trained pipeline artifact (`diabetes_regressor.pkl`). Use this to learn Docker layering, dependency management, and reproducible runs.

**Files**
- `src/main.py`: Training script — loads the Diabetes dataset, trains a pipeline (scaling + RandomForestRegressor), evaluates, and saves `diabetes_regressor.pkl`.
- `requirements.txt`: Python dependencies (`scikit-learn`, `joblib`).
- `dockerfile`: Dockerfile that installs dependencies and runs `src/main.py`.

**Prerequisites**
- Python 3.8+ or Docker installed on your machine.

**Run locally (recommended for iteration)**
1. Create and activate a virtual environment, install dependencies, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\main.py
```

After running, `diabetes_regressor.pkl` will appear in the project root.

**Build and run with Docker**
- Build the image (from project root):

```powershell
docker build -t diabetes-regressor -f dockerfile .
```

- Run the container and persist the output artifact to the host (recommended while developing):

```powershell
docker run --rm -v ${PWD}:/app diabetes-regressor
```

- If you omit `-v ${PWD}:/app`, the artifact remains inside the container image or container filesystem.

**How this Dockerfile is structured (learning points)**
- We copy `requirements.txt` and run `pip install` first so Docker can cache installed dependencies when source code changes.
- We then copy `src/` to keep application code separate from dependency installation, speeding iterative builds.

**What the script does**
- Loads the Diabetes regression dataset (small, built-in).
- Trains a pipeline with `StandardScaler` and `RandomForestRegressor`.
- Evaluates on a held-out test set and prints MSE and R².
- Saves the trained pipeline to `diabetes_regressor.pkl`.


