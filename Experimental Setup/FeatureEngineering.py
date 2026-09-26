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

