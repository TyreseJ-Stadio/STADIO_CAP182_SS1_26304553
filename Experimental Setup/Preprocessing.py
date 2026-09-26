# import Libraries
import pandas as pd

# Read the CSV file into a DataFrame
credit_card_data = pd.read_csv("Datasets/creditcard.csv")

#inspect the first few rows of the DataFrame
print("----HEAD----")
print(credit_card_data.head())

#inspect the summary statistics of the DataFrame
print("----DESCRIBE----")
print(credit_card_data.describe()) 

#inspect the data types of the DataFrame
print("----DTYPES----")
print(credit_card_data.dtypes)

#inspect for missing values in the DataFrame
print("----MISSING VALUES----")
print(credit_card_data.isnull().sum())

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

#Defining features and target variable
x = credit_card_data.drop(columns=['Class'])
y = credit_card_data['Class']

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#scaling the features using RobustScaler
scaler = RobustScaler()

#fit the scaler on the training data - only on "Amount" and "Time" columns
x_train['Scaled_Amount'] = scaler.fit_transform(x_train[['Amount']])
x_test['Scaled_Amount'] = scaler.transform(x_test[['Amount']])

x_train['Scaled_Time'] = scaler.fit_transform(x_train[['Time']])
x_test['Scaled_Time'] = scaler.transform(x_test[['Time']])

#dropping the original "Amount" and "Time" columns from the training and testing sets
x_train.drop(columns=['Amount', 'Time'], inplace=True)
x_test.drop(columns=['Amount', 'Time'], inplace=True)

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
