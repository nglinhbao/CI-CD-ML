# test_model.py
import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from train import load_and_preprocess_data, train_model

def test_data_loading():
    """Test if data loading and preprocessing works correctly"""
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
    
    assert X_train.shape[1] == len(feature_names)
    assert len(X_train) > len(X_test)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

def test_model_training():
    """Test if model training works correctly"""
    X_train, X_test, y_train, y_test, scaler, _ = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    
    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, 'feature_importances_')
    
    # Test predictions
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert np.all(predictions >= 0)  # Housing prices should be non-negative

def test_model_performance():
    """Test if model meets minimum performance requirements"""
    X_train, X_test, y_train, y_test, scaler, _ = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate R-squared score
    r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    # Assert minimum performance threshold
    assert r2 >= 0.5, f"Model R-squared score ({r2:.3f}) below minimum threshold of 0.5"