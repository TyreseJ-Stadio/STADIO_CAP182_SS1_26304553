# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score

# Loading the Test Data
x_test = pd.read_csv('../Datasets/x_test_final.csv')
y_test = pd.read_csv('../Datasets/y_test_final.csv')

if isinstance(y_test, pd.DataFrame):
    y_test = y_test.values.ravel()

# Loading both models
ffnn_model = tf.keras.models.load_model('../Models/FFNN_model.keras')
rf_model = joblib.load('../Models/Random_Forest_model.pkl')

# Generating predictions for both models
# FFNN Predictions
ffnn_prob = ffnn_model.predict(x_test, verbose=0).ravel()
ffnn_pred = (ffnn_prob >= 0.3).astype(int)

# Random Forest Predictions
rf_prob = rf_model.predict_proba(x_test)[:, 1]
rf_pred = rf_model.predict(x_test)

# Calculating Metrics
# Precision-Recall AUC calculations
precision_ffnn, recall_ffnn, _ = precision_recall_curve(y_test, ffnn_prob)
pr_auc_ffnn = auc(recall_ffnn, precision_ffnn)

precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_prob)
pr_auc_rf = auc(recall_rf, precision_rf)

# Creating a summary table
metrics_df = pd.DataFrame({
    'Metric': ['F1-Score', 'ROC-AUC', 'PR-AUC'],
    'FFNN': [f1_score(y_test, ffnn_pred), roc_auc_score(y_test, ffnn_prob), pr_auc_ffnn],
    'Random Forest': [f1_score(y_test, rf_pred), roc_auc_score(y_test, rf_prob), pr_auc_rf]
})

print("\n" + "="*40)
print("--- MODEL COMPARISON SUMMARY ---")
print("="*40)
print(metrics_df.round(4).to_string(index=False))
print("="*40)

# Generating joint visualisations 
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Subplot 1: FFNN Confusion Matrix
sns.heatmap(confusion_matrix(y_test, ffnn_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0])
axes[0, 0].set_title('FFNN Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')

# Subplot 2: RF Confusion Matrix
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[0, 1])
axes[0, 1].set_title('Random Forest Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')

# Subplot 3: ROC Curve Overlay
false_pos_rate_ffnn, true_pos_rate_ffnn, _ = roc_curve(y_test, ffnn_prob)
false_pos_rate_rf, true_pos_rate_rf, _ = roc_curve(y_test, rf_prob)

axes[1, 0].plot(false_pos_rate_ffnn, true_pos_rate_ffnn, label=f'FFNN (AUC = {roc_auc_score(y_test, ffnn_prob):.4f})', color='blue')
axes[1, 0].plot(false_pos_rate_rf, true_pos_rate_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_prob):.4f})', color='green')
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1, 0].set_title('ROC Curve Comparison')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].legend(loc='lower right')

# Subplot 4: Precision-Recall Curve Overlay
axes[1, 1].plot(recall_ffnn, precision_ffnn, label=f'FFNN (PR-AUC = {pr_auc_ffnn:.4f})', color='blue')
axes[1, 1].plot(recall_rf, precision_rf, label=f'Random Forest (PR-AUC = {pr_auc_rf:.4f})', color='green')
axes[1, 1].set_title('Precision-Recall Curve Comparison')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].legend(loc='lower left')

# Saving and rendering the dashboard
plt.tight_layout(pad=3.0)
plt.savefig('../Experimental Results/Model_Comparison.png', dpi=300)
plt.close()

print("\nComparison saved to '../Experimental Results/Model_Comparison.png'")