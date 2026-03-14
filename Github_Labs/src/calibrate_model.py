import os, pickle, json, argparse, sys
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received: {timestamp}")

    # Load the trained base model 
    model_path = f'models/model_{timestamp}_gbc.joblib'
    try:
        model = joblib.load(model_path)
        print(f"Base model loaded from {model_path}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    # Load the same test data 
    try:
        with open('data/X_test.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data/y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)
        print(f"Test data loaded: {X_test.shape[0]} samples")
    except Exception as e:
        raise ValueError(f"Failed to load test data: {e}")

    # Calibrate using both methods and pick the best 
    results = {}

    for method in ['sigmoid', 'isotonic']:
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,     # sigmoid = Platt scaling, isotonic = isotonic regression
            cv='prefit'        # model is already trained, just wrap it
        )
        calibrated.fit(X_test, y_test)

        y_pred = calibrated.predict(X_test)
        f1 = round(f1_score(y_test, y_pred, average='macro'), 4)
        results[method] = {
            "model": calibrated,
            "metrics": {
                "method": method,
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "f1_macro": f1,
                "precision_macro": round(precision_score(y_test, y_pred, average='macro'), 4),
                "recall_macro": round(recall_score(y_test, y_pred, average='macro'), 4)
            }
        }
        print(f"[{method}] F1 Macro: {f1}")

    # Pick the best calibrated model by F1 
    best_method = max(results, key=lambda m: results[m]['metrics']['f1_macro'])
    best_model = results[best_method]['model']
    best_metrics = results[best_method]['metrics']
    print(f"Best calibration method: {best_method}")

    # Save the best calibrated model 
    os.makedirs('models', exist_ok=True)
    calibrated_path = f'models/model_{timestamp}_gbc_calibrated_{best_method}.joblib'
    joblib.dump(best_model, calibrated_path)
    print(f"Calibrated model saved to {calibrated_path}")

    # Save calibration metrics 
    os.makedirs('metrics', exist_ok=True)
    metrics_path = f'metrics/{timestamp}_calibration_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(best_metrics, f, indent=4)
    print(f"Calibration metrics saved to {metrics_path}")