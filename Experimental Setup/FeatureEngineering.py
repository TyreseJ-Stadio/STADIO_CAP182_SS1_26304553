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
