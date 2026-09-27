#Evaluation of Model1 performance - Feed Forward Neural Network
#Importing the required libraries
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the model
model = tf.keras.models.load_model('../Models/FFNN_model.keras')

# Loading the test data
x_test = pd.read_csv('../Datasets/x_test_final.csv')
y_test = pd.read_csv('../Datasets/y_test_final.csv')

# Generating predictions
y_pred_prob = model.predict(x_test)

# Converting predictions to binary labels
y_pred = (y_pred_prob >= 0.3).astype(int)

# Evaluating the model
print("\n" + "="*40)
print("--- MODEL 1 EVALUATION RESULTS ---")
print("="*40)

print("\n[1] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n[2] Classification Report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"[3] ROC-AUC Score: {roc_auc:.4f}")

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
print(f"[4] Precision-Recall AUC (PR-AUC) Score: {pr_auc:.4f}")
print("="*40)

# --- Generating Confusion Matrix Visual ---
plt.figure(figsize=(8, 6))

# Creating the heatmap
# 'cm' is already defined in your script: cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Legit (0)', 'Fraud (1)'],
            yticklabels=['Legit (0)', 'Fraud (1)'])

# Adding labels and title
plt.title('Confusion Matrix - Model 1 (FFNN)', pad=15)
plt.xlabel('Predicted Fraud', labelpad=10)
plt.ylabel('True Fraud', labelpad=10)

# Saving the image to the Experimental Results folder
plt.tight_layout()
plt.savefig('../Experimental Results/Model1_Confusion_Matrix.png', dpi=300)
plt.close()