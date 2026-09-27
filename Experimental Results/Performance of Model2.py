import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the trained model
rf_model = joblib.load('../Models/Random_Forest_model.pkl')

# Loading the test data
x_test = pd.read_csv('../Datasets/x_test_final.csv')
y_test = pd.read_csv('../Datasets/y_test_final.csv')

# Generating predictions
y_pred = rf_model.predict(x_test)
y_pred_prob = rf_model.predict_proba(x_test)[:, 1]

# Evaluating the model
print("\n" + "="*40)
print("--- MODEL 2 EVALUATION RESULTS ---")
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

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Legit (0)', 'Fraud (1)'],
            yticklabels=['Legit (0)', 'Fraud (1)'])

plt.title('Confusion Matrix - Model 2 (Random Forest)', pad=15)
plt.xlabel('Predicted', labelpad=10)
plt.ylabel('Actual', labelpad=10)

plt.tight_layout()
plt.savefig('../Experimental Results/RandomForest_Confusion_Matrix.png', dpi=300)
plt.close() 

print("\nConfusion matrix visualization saved.")