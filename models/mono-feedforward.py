#%%
import tensorflow as tf
from tensorflow.keras.utils import plot_model as plot
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
from utilities import reshape
import pandas as pd

# X and Y
X = np.genfromtxt('data/mono-feedforward-train-orth.csv', delimiter=',')
Y = np.genfromtxt('data/mono-feedforward-train-phon.csv', delimiter=',')

X_test = np.genfromtxt('data/mono-feedforward-test-orth.csv', delimiter=',')
Y_test = np.genfromtxt('data/mono-feedforward-test-phon.csv', delimiter=',')

#
testwords = pd.read_csv('data/mono-test.csv')

#%%

orth = Input(shape=(X.shape[1], ), name='input')
hidden = Dense(100, activation='sigmoid', name='hidden')(orth)
phon = Dense(Y.shape[1], activation='sigmoid', name='phon')(hidden)
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]

model = Model(inputs=orth, outputs=phon, name='mono-feedforward')

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=metric)
model.summary()
# %%
items = open('../outputs/mono/feedforward/item-data.csv', 'w')
items.write('epoch,word,accuracy,loss\n')

CYCLES = 1
EPOCHS = 80
for cycle in range(1, CYCLES+1):
    model.fit(X, Y, batch_size=120, epochs=10, validation_split=0)

    for i, xy in enumerate(zip(X_test, Y_test)):
        word = testwords.iloc[i]['word']
        epochs = cycle*EPOCHS
        loss, acc = model.evaluate(xy[0].reshape((1, xy[0].shape[0])), xy[1].reshape((1, xy[1].shape[0])))
        items.write('{},{},{},{}\n'.format(epochs, word, acc, loss))
# %%
items.close()