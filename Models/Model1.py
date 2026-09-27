# This model requires further data preprocessing as it requires a validation set to be created from the training set.
# The validation set will be used to tune the hyperparameters of the model and prevent overfitting.

#importing libraries required
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Loading the data and creating a validation set from the training set
x_train_full = pd.read_csv("../Datasets/x_train_final.csv")
y_train_full = pd.read_csv("../Datasets/y_train_final.csv")

# checking the that the data is 1D to avoid errors in the model training
if isinstance(y_train_full, pd.DataFrame):
    y_train_full = y_train_full.values.ravel()

# Check for accidental NaNs or Infinities in features
if x_train_full.isna().sum().sum() > 0 or np.isinf(x_train_full.values).sum() > 0:
    #filling in NaN values with 0
    x_train_full = x_train_full.fillna(0)

#Creating a validation set from the training set
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# Defining the model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(int(x_train.shape[1]),)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy',
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')
                ]
              )
model.summary()

# Inclusion of Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Training the Model

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)    

print("----MODEL TRAINING COMPLETE----")

# Saving the trained model
model.save("../Models/FFNN_model.keras")
print("----MODEL SAVED----")