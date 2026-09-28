# import Libraries
import pandas as pd

# Read the CSV file into a DataFrame
credit_card_data_f = pd.read_csv("../Datasets/feature_engineered_data.csv")

#inspect the first few rows of the DataFrame
print("----HEAD----")
print(credit_card_data_f.head())

#inspect the summary statistics of the DataFrame
print("----DESCRIBE----")
print(credit_card_data_f.describe()) 

#inspect the data types of the DataFrame
print("----DTYPES----")
print(credit_card_data_f.dtypes)

#inspect for missing values in the DataFrame
print("----MISSING VALUES----")
print(credit_card_data_f.isnull().sum())

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Defining features and target variable
x = credit_card_data_f.drop(columns=['Class'])
y = credit_card_data_f['Class']

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Scale all features at once
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Converting the resulting NumPy arrays back to Pandas DataFrames to keep column names
x_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)

#Applying SMOTE to handle class imbalance in the training set
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train) 

#check distribution of the target variable in the resampled training set
print("----DISTRIBUTION OF TARGET VARIABLE BEFORE SMOTE----")
print("Before SMOTE - counts of label '1' (Fraud):", (y_train == 1).sum())
print("Before SMOTE - counts of label '0' (Legit):", (y_train == 0).sum())

print("----DISTRIBUTION OF TARGET VARIABLE AFTER SMOTE----")
print("After SMOTE - counts of label '1' (Fraud):", (y_train_resampled == 1).sum())
print("After SMOTE - counts of label '0' (Legit):", (y_train_resampled == 0).sum())

# Exporting the preprocessed training and testing sets to CSV files
x_train_resampled.to_csv("../Datasets/x_train_final.csv", index=False)
y_train_resampled.to_csv("../Datasets/y_train_final.csv", index=False)

x_test.to_csv("../Datasets/x_test_final.csv", index=False)
y_test.to_csv("../Datasets/y_test_final.csv", index=False)
