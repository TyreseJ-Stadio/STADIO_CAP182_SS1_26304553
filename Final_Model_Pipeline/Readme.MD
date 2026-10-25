#Dataset and Feature Engineering 
The dataset contains transaction records with a 98% non-fraud to 2% fraud imbalance. Preprocessing includes sorting transactions by client and timestamp to engineer context-aware features like 'time_since_last_transaction', 'tx_count_prior_24h', 'is_new_city', 'is_new_device', and 'ip_network_freq'. Categorical features are transformed using OneHotEncoder, continuous features are normalized with StandardScaler, and the training data is balanced using SMOTE. Unique identifiers that do not contribute to predictive power are dropped before training.

#Model Training
A Random Forest Classifier is trained and optimized using GridSearchCV with an f1_macro scoring metric. The optimal hyperparameters identified by the grid search are a maximum depth of 12, 150 estimators, a minimum sample split of 15, and a minimum of 5 samples per leaf.

#Validation and Visualization
The model is evaluated using classification reports and ROC-AUC scores to monitor for overfitting. It achieves an AUC of 0.99 on the training set and 0.82 on the unseen testing set. Final visual outputs include a side-by-side comparison of training and testing confusion matrices alongside an overlaid ROC curve plot.
