# Credit Card Fraud Feature Engineering

## Overview
This script performs automated feature engineering on a credit card transaction dataset to prepare it for fraud detection machine learning models. It adds useful features which categorize transaction amounts and generate temporal sliding-window metrics.

## Dependencies
- Python 3.x
- `pandas`

## Input and Output
* **Input:** Requires the raw dataset located at `../Datasets/creditcard.csv`.

* **Output:** Generates a new, enriched dataset saved to `../Datasets/feature_engineered_data.csv`.

## Engineered Features
This script adds three new predictive features to the dataset:

1. **`Amount_Tier` (Categorical)**
   * **Description:** Bins the transaction `Amount` into three equal-sized quantiles (terciles).
   * **Values:** `0` (Low), `1` (Medium), and `2` (High).
   * **Note:** This transforms continuous monetary values into categorical risk tiers, which can help tree-based models capture non-linear relationships with transaction size.

2. **`Transactions_Count_Last_1Hour` (Numerical)**
   * **Description:** A rolling count of all transactions that occurred in the 60 minutes (3600 seconds) prior to the current transaction.

3. **`Transactions_Amount_Sum_Last_1Hour` (Numerical)**
   * **Description:** A rolling sum of the total transaction `Amount` (in dollars) processed in the 60 minutes prior to the current transaction.

##  Important Technical Notes
* **Class Imbalance Check:** The script outputs the count of Fraudulent (`Class = 1`) vs. Legitimate (`Class = 0`) transactions upon execution. This is a critical baseline check, as the credit card data is highly imbalanced.
* **Temporal Sorting:** Before calculating the sliding window features, the dataset is explicitly sorted by the `Time` column to ensure chronological accuracy for the rolling calculations.
* **Datetime Conversion:** The original `Time` column (measured in elapsed seconds) is temporarily mapped to a mock datetime index starting at `2026-01-01`. This allows the script to leverage the pandas native `rolling('1h')` frequency function. The temporary index is removed before saving the final CSV.
