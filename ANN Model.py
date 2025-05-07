#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ANN Regression Model for Scour Depth Prediction 


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('your_dataset_file_path.csv')  # Replace with your actual file path

# Define input and output features
# Replace 'feature1', 'feature2', ... and 'target_column' with your actual column names
X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']].values
y = data['target_column'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model hyperparameters
n_layers = 3
n_units = 19
learning_rate = 0.0022318676591680585
batch_size = 41
epochs = 229

# Build ANN model
def build_model(n_layers, n_units, learning_rate):
    model = tf.keras.Sequential()
    for _ in range(n_layers):
        model.add(tf.keras.layers.Dense(n_units, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # Output layer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

# Initialize and train the model
model = build_model(n_layers, n_units, learning_rate)
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)

# Model predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluation metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, mape, r2

train_mse, train_rmse, train_mae, train_mape, train_r2 = calculate_metrics(y_train, y_train_pred)
test_mse, test_rmse, test_mae, test_mape, test_r2 = calculate_metrics(y_test, y_test_pred)

# Print evaluation results
print(f"Training Data - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.2f}%, R²: {train_r2:.4f}")
print(f"Testing Data  - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")

# Scatter plot of predicted vs actual values
def plot_scatter(y_true_train, y_pred_train, y_true_test, y_pred_test):
    plt.figure(figsize=(16, 8))

    # Training Data Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true_train, y_pred_train, color='green', edgecolor='k', s=70, alpha=0.8)
    plt.plot([min(y_true_train), max(y_true_train)], [min(y_true_train), max(y_true_train)], 
             'r--', linewidth=2, label='Ideal Fit')
    plt.xlabel('Actual scour depth', fontsize=14)
    plt.ylabel('Predicted scour depth', fontsize=14)
    plt.title('Training Data - Predicted vs Actual', fontsize=16)
    plt.legend(['Ideal Fit'], fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Testing Data Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_test, y_pred_test, color='blue', edgecolor='k', s=70, alpha=0.8)
    plt.plot([min(y_true_test), max(y_true_test)], [min(y_true_test), max(y_true_test)], 
             'r--', linewidth=2, label='Ideal Fit')
    plt.xlabel('Actual scour depth', fontsize=14)
    plt.ylabel('Predicted scour depth', fontsize=14)
    plt.title('Testing Data - Predicted vs Actual', fontsize=16)
    plt.legend(['Ideal Fit'], fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# Plot results
plot_scatter(y_train, y_train_pred, y_test, y_test_pred)

