# Model 1: Evaluation & Testing Script

Performance evaluation pipeline for the trained Feedforward Neural Network (FFNN).

## File Location
* **Script:** `Experimental Results/Model1Performance.py`
* **Input Artifacts:** `Models/FFNN_model.keras`, `Datasets/x_test_final.csv`, `Datasets/y_test_final.csv`
* **Output Artifact:** `Experimental Results/Model1_Confusion_Matrix.png`

---

## What the Script Does
1. **Loads Artifacts:** Imports the trained Keras model and test datasets.
2. **Generates Predictions:** Predicts probability scores and applies a classification threshold (`>= 0.3`) to obtain binary class labels.
3. **Calculates Metrics:** Computes the Confusion Matrix, Classification Report (Precision, Recall, F1-Score), ROC-AUC score, and Precision-Recall AUC (PR-AUC) score.
4. **Visualizes Results:** Generates and saves a styled Seaborn heatmap of the confusion matrix to the `Experimental Results/` directory.
