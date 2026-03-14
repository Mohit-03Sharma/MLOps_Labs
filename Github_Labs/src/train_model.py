import os, pickle, argparse, sys
import mlflow
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received: {timestamp}")

    # Load Wine dataset 
    wine = load_wine()
    X, y = wine.data, wine.target
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")

    # Train/test split  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save split data so evaluate_model.py uses the SAME data 
    os.makedirs('data', exist_ok=True)
    with open('data/X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)
    with open('data/y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)
    print("Test data saved to data/")

    # Train Gradient Boosting Classifier 
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    }

    model = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=42
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_metrics = {
        "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
        "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro'), 4)
    }
    print(f"Training metrics: {train_metrics}")

    # Log to MLflow  
    try:
        mlflow.set_tracking_uri("./mlruns")
        experiment_name = f"Wine_GBC_{timestamp}"
        experiment_id = mlflow.create_experiment(experiment_name)

        with mlflow.start_run(experiment_id=experiment_id, run_name="GradientBoosting_Wine"):
            mlflow.log_params({
                "model": "GradientBoostingClassifier",
                "dataset": "Wine",
                **params,
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "n_features": X_train.shape[1],
                "n_classes": len(set(y))
            })
            mlflow.log_metrics(train_metrics)
            print("MLflow logging successful.")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

    # Save model with timestamp version 
    os.makedirs('models', exist_ok=True)
    model_filename = f'models/model_{timestamp}_gbc.joblib'
    dump(model, model_filename)
    print(f"Model saved to {model_filename}")