# -*- coding: utf-8 -*-
"""
An attempt at using an autoencoder architecture to solve a multitask
problem of having a single network be able to augment incomplete data
whether the fields are regression or classification
"""

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

"""The code for this example was trained on the Medical Cost Personal Datasets available from kaggle.com.
It can perform classification on the categorical variable 'smoker', and can perform regression on any of
 the continuos features 'age','bmi','charges'.   """

df = pd.read_csv('insurance.csv')

def sex_to_int(x):
  if x == 'male':
    return 0
  else:
    return 1

df['sex'] = df['sex'].apply(sex_to_int)

smokingencoding = lambda x: 0 if x == 'yes' else 1

df.head()

df['smoker'] = df['smoker'].apply(smokingencoding)
df.drop(labels=['region','sex','children'],axis=1,inplace=True)

min_max_scaler = preprocessing.MinMaxScaler()
scaled_df = min_max_scaler.fit_transform(df.values)
scaled_df = pd.DataFrame(scaled_df)

df.corr()

scaled_df.corr()

ys = scaled_df[3]

min_max_scaler = preprocessing.MinMaxScaler()
scaled_df = min_max_scaler.fit_transform(df.values)
scaled_df = pd.DataFrame(scaled_df)

train_data, test_data, train_labels, test_labels = train_test_split(
    scaled_df, ys, test_size=0.04, random_state=21)

age_data = train_data[0]
bmi_data = train_data[1]
smoker_data = train_data[2]
charges_data = train_data[3]
smoker_targets = train_data[2]
charges_targets = train_data[3]

train_data.iloc[0].shape

# Merge all available features into a single large vector via concatenation
inputs = keras.Input(shape=(4,), name="inputs")

layer1 = layers.Dense(3, activation='relu',name="layer1")(inputs)
layer2 = layers.Dense(4,activation='relu',name='layer2')(layer1)

# Stick a classifier on top of the features
smoker_pred = layers.Dense(1, name="smoker_pred")(layer2)

# Stick a linear regression on top of the features
charges_pred = layers.Dense(1, activation='linear',name="charges_pred")(layer2)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[inputs],
    outputs=[smoker_pred,charges_pred],
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss={
        "smoker_pred": keras.losses.BinaryCrossentropy(),
        "charges_pred": keras.losses.MeanSquaredError(),
    },
    loss_weights=[1.0, 1.0],
    metrics={
        "smoker_pred": keras.metrics.BinaryAccuracy(),
        "charges_pred": keras.losses.KLDivergence(),
    },
)


model.fit(
    {"inputs": train_data},
    {"smoker_pred": smoker_targets, "charges_pred": charges_targets},
    epochs=5,
    batch_size=4
)

"""Regression on the charges column"""

model.summary()

test_smoker_targets = test_data[2]
test_charges_targets = test_data[3]

init_targets = np.full((len(test_charges_targets)),.5)

test_data_smoker = test_data.copy()
test_data_charges = test_data.copy()

test_data_smoker[2] = init_targets
test_data_charges[3] = init_targets

#predictions on charges data
predictions = model.predict(test_data_charges)
chargespredictions = predictions[1].reshape(-1)
x = np.arange(len(chargespredictions))
plt.bar(x,chargespredictions,label='prediction',alpha=0.5)
plt.bar(x,test_charges_targets,label='targets',alpha=0.5)
plt.legend()

"""Classification on the smoker column"""

predictions = model.predict(test_data_smoker)
smokerpredictions = predictions[0].reshape(-1)
x = np.arange(len(smokerpredictions))
q = []


for i in smokerpredictions :
  if i >= .5:
    q.append(.95)
  else:
    q.append(0.05)
plt.scatter(x,q,label='prediction',alpha=.5)
plt.scatter(x,test_smoker_targets,label='targets',alpha=.5)
plt.legend()
