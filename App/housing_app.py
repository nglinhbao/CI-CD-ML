import gradio as gr
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load the saved model and scaler
model = joblib.load('./Results/housing_model.joblib')
scaler = joblib.load('./Results/scaler.joblib')

# Get feature names from the dataset
feature_names = fetch_california_housing().feature_names

def predict_price(*features):
    """Predict house price based on input features.
    
    Args:
        *features: List of features in the order:
            - MedInc: Median income in block group
            - HouseAge: Median house age in block group
            - AveRooms: Average number of rooms
            - AveBedrms: Average number of bedrooms
            - Population: Block group population
            - AveOccup: Average house occupancy
            - Latitude: House block latitude
            - Longitude: House block longitude
    
    Returns:
        str: Predicted house price in $100,000s
    """
    # Convert features to numpy array and reshape
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Format prediction (convert from $100,000s to actual dollars)
    price_in_dollars = prediction * 100000
    label = f"Predicted House Price: ${price_in_dollars:,.2f}"
    
    return label

# Create input components with appropriate ranges
inputs = [
    gr.Slider(0, 15, step=0.1, label="Median Income ($10,000s)"),
    gr.Slider(0, 100, step=1, label="House Age (years)"),
    gr.Slider(1, 20, step=0.1, label="Average Rooms"),
    gr.Slider(0.5, 10, step=0.1, label="Average Bedrooms"),
    gr.Slider(100, 35000, step=100, label="Population"),
    gr.Slider(1, 20, step=0.1, label="Average Occupancy"),
    gr.Slider(30, 45, step=0.1, label="Latitude"),
    gr.Slider(-125, -110, step=0.1, label="Longitude")
]

outputs = [gr.Label(label="Prediction")]

# Example inputs based on real data points
examples = [
    [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23],
    [3.1250, 21.0, 6.238137, 0.971880, 2401.0, 2.109842, 37.86, -122.22],
    [5.6431, 52.0, 5.817352, 1.073059, 558.0, 2.547945, 37.85, -122.25]
]

title = "California House Price Prediction"
description = "Enter house features to predict its price. The model was trained on the California Housing dataset."
article = """
This app predicts house prices in California based on various features:
- Median Income: Median income in the block group (in tens of thousands of dollars)
- House Age: Median age of houses in the block group
- Average Rooms: Average number of rooms per household
- Average Bedrooms: Average number of bedrooms per household
- Population: Total population in the block group
- Average Occupancy: Average number of household members
- Latitude and Longitude: Geographic coordinates

The predictions are based on a Random Forest model trained on the California Housing dataset.
"""

# Create and launch the interface
gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()