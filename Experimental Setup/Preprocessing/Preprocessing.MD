# Credit Card Fraud Dataset - Preprocessing 

## Overview
Associated Script: “Experimental Setup/Preprocessing.py”

This document outlines the data preparation pipeline used before training our machine learning models. Because fraud data is heavily imbalanced, a major focus of this stage is scaling the features and balancing the classes so our models do not simply predict "not fraud" constantly.

## Dependencies
- `Python 3.x`
- `pandas`

## Input and Output
* **Input:** Dataset output from Feature Engineering `../Datasets/feature_engineered_data.csv`.

* **Output:** Generates 4 new datasets which are:

x_train_final - "../Datasets/x_train_final.csv"  
y_train_final - "../Datasets/y_train_final.csv"  
x_test_final - "../Datasets/x_test_final.csv"  
y_test_final - "../Datasets/y_test_final.csv"  

## Preprocessing
This script adds three new predictive features to the dataset:

1. **Dataset Inspection**
   * **It loads “feature_engineered_data.csv” and runs through basic exploratory checks—printing the head, summary statistics, data types, and checking for missing values to ensure the data is clean before manipulation.

2. **Train/Test Split**
   * **The data is split into 80% training and 20% testing sets. It uses a stratified split (stratify=y) to guarantee that the percentage of fraud cases is distributed evenly between the training and testing sets.

3. **Feature Scaling**
   * **It uses StandardScaler to normalize the feature distributions. The scaler is fit strictly on the training data and then applied to both sets to prevent data leakage. The resulting arrays are converted back into pandas DataFrames to preserve column names. (Note: While RobustScaler is imported in the code, StandardScaler is currently the active scaler as Robust scaler was not effective).


4. **Handling Class Imbalance with SMOTE**
   * **Synthetic Minority Over-sampling Technique (SMOTE) is applied only to the training set. This generates synthetic examples of the minority class (fraud) so the model has a balanced dataset to learn from. The script prints before-and-after counts to verify the classes are balanced.

5. **Exporting the Data**
   * **It saves the model-ready data into four separate CSV files, ready to be fed into the classification algorithms.

