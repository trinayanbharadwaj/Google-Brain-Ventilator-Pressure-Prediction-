# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Reading the datasets
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
df_sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

n_steps = 80

# dividing the data into features and target variable.
train_x = df_train.drop(columns=["pressure", "id", "breath_id"])
test = df_test.drop(columns=["id", "breath_id"])

# Reshaping the data into a valid format for Bi-LSTM models
train_y = df_train["pressure"].to_numpy().reshape(-1, 80)
train_x = train_x.to_numpy().reshape(-1, 80, train_x.shape[-1])
test = test.to_numpy().reshape(-1, 80, test.shape[-1])

# Creating the model
def get_model():
    act = "selu"
    model = keras.Sequential([
        layers.InputLayer(input_shape=(80,5)),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(1028, return_sequences=True)),
        #layers.Bidirectional(layers.LSTM(2048, return_sequences=True)),
        #layers.Dropout(0.2),
        #layers.Flatten(),
        #layers.Dense(128, activation=act),
        #layers.Dense(256, activation=act),
        #layers.Dense(50, activation=act),
        layers.Dense(1)
    ])
    return model


model = get_model()
model.compile(optimizer="adam", loss="mae")

# training the model
def fit_model(mod):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 5)
    history = mod.fit(train_x, train_y, epochs = 100,
                        validation_split = 0.3, batch_size = 32, 
                        shuffle = False, callbacks = [early_stop]
                       )
    return history

history_bilstm = fit_model(model)

# Saving the model
model.save("Bi-LSTM-1-val_loss: 0.2526.h5")
