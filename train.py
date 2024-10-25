import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    # Load the California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Get feature names for later use
    feature_names = housing.feature_names

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def train_model(X_train, y_train):
    # Initialize and train the XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=10,
        eval_metric="rmse",
    )

    # Create evaluation set for early stopping
    eval_set = [(X_train, y_train)]

    # Train the model
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    return model


def evaluate_model(model, X_test, y_test, feature_names):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Create Results directory if it doesn't exist
    os.makedirs("Results", exist_ok=True)

    # Save metrics to file
    with open("Results/metrics.txt", "w") as outfile:
        outfile.write(f"Root Mean Squared Error = {rmse}, R-squared Score = {r2}")

    print("\nModel Performance Metrics:")
    print(f"Root Mean Squared Error: ${rmse:.2f}k")
    print(f"R-squared Score: {r2:.3f}")

    # Feature importance plot
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.title("Feature Importances (XGBoost)")
    plt.bar(range(X_test.shape[1]), importances[indices])
    plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()

    # Save plot in Results directory
    plt.savefig("Results/feature_importance.png")

    # Additional XGBoost-specific visualization
    xgb.plot_importance(model, max_num_features=10)
    plt.title("XGBoost Feature Importance Score")
    plt.tight_layout()
    plt.savefig("Results/xgb_feature_importance.png")

    return rmse, r2


def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()

    print("Training XGBoost model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    rmse, r2 = evaluate_model(model, X_test, y_test, feature_names)

    # Save the model and scaler in Results directory
    print("\nSaving model and scaler...")
    joblib.dump(model, "Model/housing_model.joblib")
    joblib.dump(scaler, "Results/scaler.joblib")

    print("\nTraining complete! Files saved in Results directory:")
    print("- metrics.txt")
    print("- feature_importance.png")
    print("- xgb_feature_importance.png")
    print("- housing_model.joblib")
    print("- scaler.joblib")


if __name__ == "__main__":
    main()
