#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('your_dataset_file_path.csv')  # Replace with your actual file path

# Define input and output features
# Replace 'feature1', 'feature2', ... and 'target_column' with your actual column names
X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']].values
y = data['target_column'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define GPR model with the best parameters
kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=20)

# Fit the GPR model on training data
gpr.fit(X_train_scaled, y_train)

# Predict for both training and testing datasets
y_train_pred, sigma_train = gpr.predict(X_train_scaled, return_std=True)
y_test_pred, sigma_test = gpr.predict(X_test_scaled, return_std=True)

# Function to calculate and print evaluation metrics
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

# Evaluate the GPR model
evaluate_model("Gaussian Process Regression", y_train, y_train_pred, y_test, y_test_pred)

# Function to plot scatter plots for training and testing datasets
def plot_results(y_train, y_train_pred, y_test, y_test_pred, title, save_path):
    plt.figure(figsize=(16, 8))

    # Training Data
    plt.subplot(1, 2, 1)
    plt.scatter(
        y_train, y_train_pred, 
        color='green', edgecolor='k', s=70, alpha=0.8, label='Training Data'
    )
    plt.plot(
        [min(y_train), max(y_train)], 
        [min(y_train), max(y_train)], 
        'r--', linewidth=2, label='Ideal Fit'
    )
    plt.xlabel('Actual scour depth', fontsize=14)
    plt.ylabel('Predicted scour depth', fontsize=14)
    plt.title(f'{title} - Training Data', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Testing Data
    plt.subplot(1, 2, 2)
    plt.scatter(
        y_test, y_test_pred, 
        color='blue', edgecolor='k', s=70, alpha=0.8, label='Testing Data'
    )
    plt.plot(
        [min(y_test), max(y_test)], 
        [min(y_test), max(y_test)], 
        'r--', linewidth=2, label='Ideal Fit'
    )
    plt.xlabel('Actual scour depth', fontsize=14)
    plt.ylabel('Predicted scour depth', fontsize=14)
    plt.title(f'{title} - Testing Data', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # General Layout Adjustments
    plt.tight_layout(pad=3)

    # Save the plot to the specified path
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Plot results and save high-resolution images
save_path = # provide path to download plot
plot_results(y_train, y_train_pred, y_test, y_test_pred, 'Gaussian Process Regression', save_path)

