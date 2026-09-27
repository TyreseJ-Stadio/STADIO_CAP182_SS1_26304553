# This model requires further data preprocessing as it requires a validation set to be created from the training set.
# The validation set will be used to tune the hyperparameters of the model and prevent overfitting.

#importing libraries required
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# Loading the data and creating a validation set from the training set
x_train_full = pd.read_csv("../Datasets/x_train_final.csv")
y_train_full = pd.read_csv("../Datasets/y_train_final.csv")

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)



