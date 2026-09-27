# import required Libraries

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib

# Loading the data
x_train = pd.read_csv("../Datasets/x_train_final.csv")
y_train = pd.read_csv("../Datasets/y_train_final.csv")

# Testing if y train is a 1D array
print(y_train.ndim)

# y_train is a 2D array, so we need to convert it to a 1D array
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.values.ravel()

# Creating the Random Forest Classifier model | Base Model
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Using grid search to find the best hyperparameters for the model
# defining the hyperparameter grid for Random Forest Classifier
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Setting up the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring='roc_auc',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fitting the RandomizedSearchCV to find the best hyperparameters
random_search.fit(x_train, y_train)

# Displaying the best hyperparameters | Best Hyperparameters: {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 30, 'class_weight': 'balanced'} from initial training 
print("Best Hyperparameters:", random_search.best_params_)

#Extracting and saving the best model
best_rf_model = random_search.best_estimator_

joblib.dump(best_rf_model, "../Models/Random_Forest_model.pkl")
print("----MODEL SAVED----")