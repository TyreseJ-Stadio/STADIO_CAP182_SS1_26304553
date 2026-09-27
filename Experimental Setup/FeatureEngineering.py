#importing libraries required

import pandas as pd

# Read the CSV file into a DataFrame
credit_card = pd.read_csv("../Datasets/creditcard.csv")

# Inspect the first few rows of the DataFrame
print("----HEAD----")
print(credit_card.head())

print("----COUNT OF FRAUDULENT TRANSACTIONS----")
print((credit_card["Class"] == 1).sum())
print("----COUNT OF LEGITIMATE TRANSACTIONS----")
print((credit_card["Class"] == 0).sum())

# Creating the new feature - Tiers for client transactions based on the "Amount" column | 0=Low, 1=Medium, 2=High

credit_card["Amount_Tier"] = pd.qcut(
    credit_card["Amount"],
    q=3,
    labels=[0,1,2]
)

#ensuring that the new feature is of integer type
credit_card["Amount_Tier"] = credit_card["Amount_Tier"].astype(int)

# Inspect the first few rows of the DataFrame after Adding Tiers
print("----HEAD AFTER FEATURE ENGINEERING----")
print(credit_card[["Amount", "Amount_Tier"]].head())

# Adding a sliding window feature based on "Time"
credit_card = credit_card.sort_values("Time").reset_index(drop=True)

print("----HEAD AFTER SORTING BY TIME----")
print(credit_card["Time"].head(10))

#Creating temporary datetime column 
base_date = pd.to_datetime('2026-01-01')
credit_card['Temp_Datetime'] = base_date + pd.to_timedelta(credit_card['Time'], unit='s')
credit_card = credit_card.set_index('Temp_Datetime')

# Creating a sliding window feature based on "Time" - Count of transactions in the last 1 hour (3600 seconds)
credit_card['Transactions_Count_Last_1Hour'] = credit_card['Amount'].rolling('1h').count()
credit_card['Transactions_Amount_Sum_Last_1Hour'] = credit_card['Amount'].rolling('1h').sum()

# Reset the index to bring back the original "Time" column
credit_card = credit_card.reset_index(drop=True)

# Viewing the first few rows of the DataFrame after adding the new features
print("----HEAD AFTER FEATURE ENGINEERING----")
print(credit_card.head())

# Saving the feature engineered DataFrame to a new CSV file
credit_card.to_csv("../Datasets/feature_engineered_data.csv", index=False)