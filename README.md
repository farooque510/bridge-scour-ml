
# Scour Depth Prediction Around Bridge Piers Using Machine Learning Algorithms

This repository contains the source code for the research article:

**Title:** *Prediction of scour depth around bridge piers using Machine Learning Algorithms*  
**Authors:** Farooque Rahman, *Dr. Rutuja Chavan*

The models are developed in Python and utilize various supervised machine learning techniques to predict scour depth around bridge piers based on hydrodynamic and geometric parameters.

---

## 📌 Overview

Scour is one of the leading causes of bridge failure worldwide. This work applies five machine learning algorithms to estimate relative scour depth using input features derived from flow and pier properties.

The following models are implemented:
- **ANN (Artificial Neural Network)**
- **GPR (Gaussian Process Regression)**
- **SVR (Support Vector Regression)**
- **MLR (Multiple Linear Regression)**
- **RF (Random Forest)**

Each script trains, evaluates, and visualizes its model using standard regression metrics.

---

## 📁 Repository Contents

| File Name         | Description                                                   |
|------------------|---------------------------------------------------------------|
| `ANN_Model.py`    | Neural network implementation using TensorFlow/Keras         |
| `GPR_Model.py`    | Gaussian Process Regression using scikit-learn                |
| `SVR_Model.py`    | Support Vector Regression using RBF kernel                    |
| `MLR_Model.py`    | Multiple Linear Regression baseline model                     |
| `RF_Model.py`     | Random Forest implementation                                  |
| `test_data.csv`   | Small sample dataset for quick testing              |
| `README.md`       | This readme file                                              |
---

## 🛠️ How to Use the Scour Depth Prediction Models



### ✅ Prerequisites

- **Python 3.8+**
- Required Python packages:
  ```bash
  pip install numpy pandas scikit-learn matplotlib tensorflow
  ```

### 📊 Input Data Format

Prepare a `.csv` file with the following **dimensionless input parameters**:

| Variable | Description                                   |
|----------|-----------------------------------------------|
| b/y      | Pier width to flow depth ratio               |
| V/Vc     | Approach velocity to critical velocity ratio |
| Fr       | Froude number                                 |
| b/d50    | Pier width to median sediment size           |
| σg       | Standard deviation of sediment size          |
| ys/y     | Normalized scour depth (target/output)       |

Example `test_data.csv`:
```csv
b/y,V/Vc,Fr,b/d50,σg,ys/y
1.2,1.1,0.43,45.6,1.7,0.62
0.9,0.85,0.39,40.3,1.4,0.57
```

### 🚀 Running a Model

1. Replace `'your_dataset_file_path.csv'` with your actual dataset path.
2. Replace the column names 'feature1', 'feature2', ..., 'target_column' with your dataset's actual feature and target names.
3. Run the model (e.g.):
   ```bash
   python models/ANN_Model.py
   ```
4. Each script outputs evaluation metrics and scatter plots.

### 📦 Dependencies

The following Python libraries are required to run the models:
- numpy
- pandas
- scikit-learn
- matplotlib
- tensorflow (for ANN_Model.py only).




