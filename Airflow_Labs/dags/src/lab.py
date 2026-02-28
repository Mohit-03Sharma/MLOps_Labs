
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    """
    Generates a synthetic sales/customer dataset and serializes it.
    In a real project, replace this with pd.read_csv('data/sales.csv').
    """
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'age':             np.random.randint(18, 65, n),
        'annual_spend':    np.random.uniform(100, 10000, n).round(2),
        'num_purchases':   np.random.randint(1, 50, n),
        'avg_order_value': np.random.uniform(10, 500, n).round(2),
        'days_since_last_purchase': np.random.randint(1, 365, n),
        # Target: churn (1 = churned, 0 = retained)
        # Higher days_since_last_purchase → more likely to churn
        'churned': (np.random.rand(n) < (
            0.1 + 0.6 * (np.random.randint(1, 365, n) / 365)
        )).astype(int)
    })

    serialized = pickle.dumps(df)
    print(f"[load_data] Loaded {len(df)} records.")
    return serialized
def data_preprocessing(serialized_data):
    """
    Deserializes data, handles missing values, scales features,
    and returns a serialized dict with X_train, X_test, y_train, y_test.
    """
    df = pickle.loads(serialized_data)

    # Drop nulls (none in synthetic data, but good practice)
    df = df.dropna()

    features = ['age', 'annual_spend', 'num_purchases',
                 'avg_order_value', 'days_since_last_purchase']
    target = 'churned'

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Save scaler alongside splits so evaluation can use same scale
    payload = {
        'X_train': X_train,
        'X_test':  X_test,
        'y_train': y_train,
        'y_test':  y_test,
        'scaler':  scaler,
        'feature_names': features
    }

    print(f"[data_preprocessing] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return pickle.dumps(payload)


def build_save_model(serialized_payload, filename):
    """
    Trains a Logistic Regression model, saves it to disk,
    and returns serialized evaluation data (X_test, y_test).
    """
    payload = pickle.loads(serialized_payload)

    X_train = payload['X_train']
    y_train = payload['y_train']

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Save model to file
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"[build_save_model] Model saved to '{filename}'.")

    # Pass test data downstream for evaluation
    eval_payload = {
        'X_test':  payload['X_test'],
        'y_test':  payload['y_test'],
        'feature_names': payload['feature_names']
    }
    return pickle.dumps(eval_payload)

def evaluate_model(filename, serialized_eval):
    """
    Loads the saved model, runs predictions on test data,
    and prints accuracy + full classification report.
    """
    eval_payload = pickle.loads(serialized_eval)
    X_test  = eval_payload['X_test']
    y_test  = eval_payload['y_test']
    feature_names = eval_payload['feature_names']

    with open(filename, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])

    print(f"\n[evaluate_model] Accuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{report}")

    # Log feature coefficients for interpretability
    coef_df = pd.DataFrame({
        'Feature':     feature_names,
        'Coefficient': model.coef_[0].round(4)
    }).sort_values('Coefficient', ascending=False)

    print(f"\nFeature Coefficients:\n{coef_df.to_string(index=False)}")

    return f"Accuracy: {acc:.4f}"
