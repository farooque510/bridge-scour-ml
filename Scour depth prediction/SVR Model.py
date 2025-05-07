#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# svr_model.py
"""
SVR Model for Predicting Relative Scour Depth
-----------------------------------------------------
This script trains and evaluates a Support Vector Regression (SVR) kernel for predicting scour depth around bridge piers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ------------------------- Load Dataset -------------------------
# Path to the cleaned CSV dataset
data = pd.read_csv('your_dataset_file_path.csv')  # Replace with your actual file path

# Define input and output features
# Replace 'feature1', 'feature2', ... and 'target_column' with your actual column names
X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']].values
y = data['target_column'].values

X = data[features].values
y = data[target].values

# ------------------------- Train-Test Split -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------- Feature Scaling -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------- Evaluation Function -------------------------
def evaluate_model(name, y_train, y_train_pred, y_test, y_test_pred):
    """Prints evaluation metrics for both training and test sets."""
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, mape, r2

    rmse_tr, mae_tr, mape_tr, r2_tr = metrics(y_train, y_train_pred)
    rmse_te, mae_te, mape_te, r2_te = metrics(y_test, y_test_pred)

    print(f"{name} - Training:  RMSE = {rmse_tr:.4f}, MAE = {mae_tr:.4f}, MAPE = {mape_tr:.2f}%, R² = {r2_tr:.4f}")
    print(f"{name} - Testing:   RMSE = {rmse_te:.4f}, MAE = {mae_te:.4f}, MAPE = {mape_te:.2f}%, R² = {r2_te:.4f}")
    print("-" * 60)

# ------------------------- Plotting Function -------------------------
def plot_results(y_train, y_train_pred, y_test, y_test_pred, title, save_path):
    """Plots actual vs. predicted values for both training and testing datasets."""
    plt.figure(figsize=(16, 6))

    # Training data plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, color='green', edgecolor='k', s=70, alpha=0.7, label='Training Data')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', linewidth=2, label='Ideal Fit')
    plt.xlabel('Actual scour depth', fontsize=12)
    plt.ylabel('Predicted scour depth', fontsize=12)
    plt.title(f'{title} - Training Set', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Testing data plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, color='blue', edgecolor='k', s=70, alpha=0.7, label='Testing Data')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', linewidth=2, label='Ideal Fit')
    plt.xlabel('Actual scour depth', fontsize=12)
    plt.ylabel('Predicted scour depth', fontsize=12)
    plt.title(f'{title} - Testing Set', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(pad=3)

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------- Train SVR Model -------------------------
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = svr.predict(X_train_scaled)
y_test_pred = svr.predict(X_test_scaled)

# ------------------------- Evaluate & Visualize -------------------------
evaluate_model("Support Vector Regression", y_train, y_train_pred, y_test, y_test_pred)

# Save plot
output_plot_path = #provide path to download the plot
plot_results(y_train, y_train_pred, y_test, y_test_pred, 'SVR', output_plot_path)

