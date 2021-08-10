#%%
import tensorflow as tf
from tensorflow.keras.utils import plot_model as plot
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np

# X and Y
X = np.genfromtxt('../inputs/3k/orth.csv', delimiter=',')
Y = np.genfromtxt('../inputs/3k/phon.csv', delimiter=',')

#%%

orth = Input(shape=(X.shape[1], ), name='input')
hidden = Dense(100, activation='sigmoid', name='hidden')(orth)
phon = Dense(Y.shape[1], activation='sigmoid', name='phon')(hidden)
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]

model = Model(inputs=orth, outputs=phon, name='mono-feedforward')

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=metric)
model.summary()
# %%
model.fit(X, Y, batch_size=120, epochs=320, validation_split=0)
# %%
