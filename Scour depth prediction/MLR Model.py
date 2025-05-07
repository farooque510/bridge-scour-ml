#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('your_dataset_file_path.csv')  # Replace with your actual file path

# Define input and output features
# Replace 'feature1', 'feature2', ... and 'target_column' with your actual column names
X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']].values
y = data['target_column'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to evaluate the model
def evaluate_model(name, y_train, y_train_pred, y_test, y_test_pred):
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    r2_train = r2_score(y_train, y_train_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    r2_test = r2_score(y_test, y_test_pred)

    print(f"{name} - Training Data: RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}, MAPE = {mape_train:.2f}%, R² = {r2_train:.4f}")
    print(f"{name} - Testing Data: RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}, MAPE = {mape_test:.2f}%, R² = {r2_test:.4f}")
    print('-' * 50)

# Function to plot the results
def plot_results(y_train, y_train_pred, y_test, y_test_pred, title, save_path):
    plt.figure(figsize=(16, 8))

    # Training Data
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, color='green', edgecolor='k', s=70, alpha=0.8, label='Training Data')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', linewidth=2, label='Ideal Fit')
    plt.xlabel('Actual scour depth', fontsize=14)
    plt.ylabel('Predicted scour depth', fontsize=14)
    plt.title(f'{title} - Training Data', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Testing Data
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, color='blue', edgecolor='k', s=70, alpha=0.8, label='Testing Data')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', linewidth=2, label='Ideal Fit')
    plt.xlabel('Actual scour depth', fontsize=14)
    plt.ylabel('Predicted scour depth', fontsize=14)
    plt.title(f'{title} - Testing Data', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(pad=3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Apply Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Predictions
y_train_pred_mlr = mlr.predict(X_train)
y_test_pred_mlr = mlr.predict(X_test)

# Evaluate the MLR model
evaluate_model("Multiple Linear Regression", y_train, y_train_pred_mlr, y_test, y_test_pred_mlr)

# Plot and save the result
save_path = # provide path to download
plot_results(y_train, y_train_pred_mlr, y_test, y_test_pred_mlr, 'MLR', save_path)

