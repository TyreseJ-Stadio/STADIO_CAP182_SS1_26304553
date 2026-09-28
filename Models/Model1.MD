# Model 1: Feedforward Neural Network (FFNN) for Binary Classification

## Overview
This repository contains the implementation and training pipeline for **Model 1**, a Feedforward Neural Network (FFNN) built using TensorFlow/Keras. This model is designed for binary classification tasks, featuring automated validation set creation, data sanitization, early stopping, and performance tracking. 

* Training & Inference File: `..Models/Model1.py`
* Inputs: `..Datasets/x_train_final.csv`, `..Datasets/y_train_final.csv`, `..Datasets/x_test_final.csv`, `..Datasets/y_test_final.csv`
* Output: `..Models/FFNN_model.keras`

## What this Script Does:
1. **Loads & Cleans Data:** Reads `x_train_final.csv` and `y_train_final.csv`, flattens target labels, and fills missing values.  
2. **Validation Split:** Creates a stratified 20% validation split from the training data (`random_state=42`).  
3. **Architecture:** A Sequential neural network with two Dense layers (64 and 32 units, ReLU activation), Dropout regularization (`0.3`), and a Sigmoid output layer.  
4. **Training:** Compiles using the Adam optimizer and binary crossentropy loss, utilizing **Early Stopping** (patience=5) to prevent overfitting.  
5. **Saving:** Exports the trained model to the `Models/` directory.  

## How to Run: 
Ensure dependencies are installed (pip install -r requirements.txt), then execute the main script from the root directory: ..Models/Model1.py