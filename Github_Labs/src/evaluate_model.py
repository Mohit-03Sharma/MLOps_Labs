import os, pickle, json, argparse, sys
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score
)

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received: {timestamp}")

    # Load the trained model 
    model_path = f'models/model_{timestamp}_gbc.joblib'
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    # Load the SAME test data saved by train_model.py 
    try:
        with open('data/X_test.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data/y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)
        print(f"Test data loaded: {X_test.shape[0]} samples")
    except Exception as e:
        raise ValueError(f"Failed to load test data: {e}")

    # Evaluate 
    y_pred = model.predict(X_test)

    metrics = {
        "timestamp": timestamp,
        "model": "GradientBoostingClassifier",
        "dataset": "Wine",
        "test_samples": int(X_test.shape[0]),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average='macro'), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average='macro'), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average='macro'), 4)
    }

    print(f"Evaluation metrics: {metrics}")

    # Save metrics 
    os.makedirs('metrics', exist_ok=True)
    metrics_path = f'metrics/{timestamp}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")