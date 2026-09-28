# Model 2: Evaluation & Testing Script

## Overview
Performance evaluation pipeline for the trained Random Forest classifier.

## File Location
* **Script:** `Experimental Results/Model2Performance.py`
* **Inputs:** `Models/Random_Forest_model.pkl`, `Datasets/x_test_final.csv`, `Datasets/y_test_final.csv`
* **Outputs:** `Experimental Results/RandomForest_Confusion_Matrix.png`

---

## What the Script Does
1. **Loads Artifacts:** Imports the trained Random Forest model and test datasets.
2. **Generates Predictions:** Predicts class labels and probability scores on the test set.
3. **Calculates Metrics:** Computes the Confusion Matrix, Classification Report (Precision, Recall, F1-Score), ROC-AUC score, and Precision-Recall AUC (PR-AUC) score.
4. **Visualizes Results:** Generates and saves a styled Seaborn heatmap of the confusion matrix to the `Experimental Results/` directory.
