# Model Comparison & Benchmarking

## Overview
Comparative analysis pipeline for the Feedforward Neural Network and Random Forest models[cite: 1].

## File Location
* **Script:** `Experimental Results/Comparison.py`
* **Inputs:** `Models/FFNN_model.keras`, `Models/Random_Forest_model.pkl`, `Datasets/x_test_final.csv`, `Datasets/y_test_final.csv`
* **Output Artifact:** `Experimental Results/Model_Comparison.png`

---

## What the Script Does
1. **Loads Artifacts:** Imports both trained models and the test datasets.
2. **Generates Predictions:** Runs inference on both models to obtain predicted labels and probabilities.
3. **Calculates Comparison Metrics:** Computes comparative F1-Scores, ROC-AUC, and PR-AUC scores, printing a formatted metrics table to the terminal.
4. **Visualizes Results:** Generates a 4-panel dashboard containing individual confusion matrices, an ROC curve overlay, and a Precision-Recall curve overlay, saving it to the `Experimental Results/` directory.

## How to Run: 
Ensure dependencies are installed (pip install -r requirements.txt), then execute the main script from the root directory: Experimental_Results/Comparison.py