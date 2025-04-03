# Stock Market Prediction using Hybrid Models

## Overview
This project focuses on predicting stock market trends using a hybrid model that integrates multiple machine learning and deep learning techniques. The goal is to improve forecasting accuracy by combining different modeling approaches.

## Features
- Data preprocessing and feature engineering
- Time series analysis using statistical models
- Machine learning-based forecasting
- Deep learning-based forecasting (LSTM, GRU)
- Hybrid model combining multiple techniques
- Model evaluation and visualization

## Installation
```bash
pip install -r requirements.txt
```

## Data Preprocessing
The dataset is first preprocessed by handling missing values, normalizing features, and performing time series transformation.

```python
import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df
```

## Machine Learning Models
### Linear Regression
```python
from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
```

### Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
```

## Deep Learning Model (LSTM)
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

## Hybrid Model
A hybrid approach is implemented by combining predictions from multiple models to improve overall accuracy.

```python
def hybrid_prediction(models, X_test):
    predictions = [model.predict(X_test) for model in models]
    return sum(predictions) / len(predictions)  # Averaging predictions
```

## Model Evaluation
Performance metrics are used to evaluate the models.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"MAE: {mae}, MSE: {mse}")
```

## How to Use
1. Load the dataset and preprocess it.
2. Train individual models (Linear Regression, Random Forest, LSTM).
3. Combine models using the hybrid approach.
4. Evaluate model performance.

## Results
The hybrid model provides better prediction accuracy compared to individual models. Results are visualized using matplotlib.

```python
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.legend()
    plt.show()
```

## Conclusion
This project demonstrates how hybrid models can improve stock market prediction accuracy by leveraging the strengths of both machine learning and deep learning models.
