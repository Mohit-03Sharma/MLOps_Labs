from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def main():
    # Load the built-in Diabetes regression dataset
    data = load_diabetes()
    X, y = data.data, data.target

    # Split into train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build a simple pipeline: scaling -> random forest regressor
    pipeline = make_pipeline(
        StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Save trained pipeline
    joblib.dump(pipeline, 'diabetes_regressor.pkl')

    print(f"Training complete. MSE: {mse:.4f}, R2: {r2:.4f}")


if __name__ == '__main__':
    main()
