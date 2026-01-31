import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_artifacts():
    root = project_root()
    model_path = root / "model" / "wine_model.pkl"
    metadata_path = root / "model" / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Run: cd src && python train.py"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. "
            f"Run: cd src && python train.py"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


def validate_features(features: List[float], expected_len: int) -> None:
    if not isinstance(features, list):
        raise ValueError("features must be a list of floats.")
    if len(features) != expected_len:
        raise ValueError(f"Expected {expected_len} features, got {len(features)}.")
    for i, v in enumerate(features):
        if not isinstance(v, (int, float)):
            raise ValueError(f"Feature at index {i} is not numeric: {v}")


def predict_class(features: List[float]) -> Dict[str, Any]:
    model, metadata = load_artifacts()
    expected_len = int(metadata["n_features"])
    validate_features(features, expected_len)

    pred = int(model.predict([features])[0])
    target_names = metadata.get("target_names", [])
    pred_label = target_names[pred] if pred < len(target_names) else str(pred)

    return {
        "prediction": pred,
        "prediction_label": pred_label
    }


def predict_proba(features: List[float]) -> Dict[str, Any]:
    model, metadata = load_artifacts()
    expected_len = int(metadata["n_features"])
    validate_features(features, expected_len)

    if not hasattr(model, "predict_proba"):
        raise ValueError("This model does not support predict_proba().")

    probs = model.predict_proba([features])[0].tolist()
    target_names = metadata.get("target_names", [])

    return {
        "probabilities": probs,
        "classes": target_names if target_names else list(range(len(probs)))
    }