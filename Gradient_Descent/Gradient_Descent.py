import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("./insurance_data-1.csv")   # C:\Users\User\OneDrive\Desktop\ML Projects\Using_Tensorflow_for_Deep_Learning\Gradient_Descent\insurance_data-1.csv
# print(df.head())

x = df[['age', 'affordibility']]
y = df['bought_insurance']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(x_train)

# Scalling data
x_train_scaled = x_train.copy()
x_train_scaled['age'] = x_train_scaled['age'] / 100
x_test_scaled = x_test.copy()
x_test_scaled['age'] = x_test_scaled['age'] / 100
# print(x_train_scaled)

model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_scaled, y_train, epochs=5000)
loss, accuricy = model.evaluate(x_test_scaled, y_test)
print("Accuricy = ", accuricy*100)
print("Loss = ", loss*100)