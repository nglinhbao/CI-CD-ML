# test_model.py
import pytest
import numpy as np
import xgboost as xgb
import joblib
from train import load_and_preprocess_data, train_model


def test_data_loading():
    """Test if data loading and preprocessing works correctly"""
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()

    # Test data shapes and relationships
    assert X_train.shape[1] == len(feature_names)
    assert len(X_train) > len(X_test)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_model_training():
    """Test if model training works correctly"""
    X_train, X_test, y_train, y_test, scaler, _ = load_and_preprocess_data()
    model = train_model(X_train, y_train)

    # Test model type and attributes
    assert isinstance(model, xgb.XGBRegressor)
    assert hasattr(model, "feature_importances_")

    # Test predictions
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert np.all(predictions >= 0)  # Housing prices should be non-negative

    # Test prediction shape and basic statistics
    assert predictions.shape == y_test.shape
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))


def test_model_performance():
    """Test if model meets minimum performance requirements"""
    X_train, X_test, y_train, y_test, scaler, _ = load_and_preprocess_data()
    model = train_model(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((y_test - predictions) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum(
        (y_test - np.mean(y_test)) ** 2
    )

    # Assert performance thresholds
    assert r2 >= 0.5, f"Model R-squared score ({r2:.3f}) below minimum threshold of 0.5"
