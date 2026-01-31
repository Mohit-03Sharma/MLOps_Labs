import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn

from data import load_dataset


def project_root() -> Path:
        return Path(__file__).resolve().parents[1]


def main():
    X, y, feature_names, target_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    root = project_root()
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "wine_model.pkl"
    metadata_path = model_dir / "metadata.json"

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata (useful for /model-info and validation)
    metadata = {
        "dataset": "sklearn.datasets.load_wine",
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "target_names": target_names,
        "test_accuracy": float(acc),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "artifact": str(model_path.name),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Training complete")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()