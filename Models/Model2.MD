# Model 2: Random Forest Training Script
Training pipeline for the optimized Random Forest Model

## File Location
* **Script:** `Models/Model2.py`
* **Output Model:** `Models/Random_Forest_model.pkl`

## What the Script Does:
1. **Loads Data:** Reads training datasets files (`x_train_final.csv`, `y_train_final.csv`).
2. **Target Formatting:** Inspects label dimensions and converts `y_train` into a 1D array if needed.
3. **Model Configuration:** Sets up a `RandomForestClassifier` initialized with tuned hyperparameters (`n_estimators=300`, `max_depth=30`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight='balanced'`).
4. **Training:** Fits the model using the best parameters previously determineed.
5. **Saving:** Exports the trained model using `joblib`.
