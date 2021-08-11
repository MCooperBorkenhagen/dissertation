#%%
import tensorflow as tf
from tensorflow.keras.utils import plot_model as plot
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
from utilities import reshape
import pandas as pd
import time

# X and Y
X = np.genfromtxt('data/mono-feedforward-train-orth.csv', delimiter=',')
Y = np.genfromtxt('data/mono-feedforward-train-phon.csv', delimiter=',')

X_test = np.genfromtxt('data/mono-feedforward-test-orth.csv', delimiter=',')
Y_test = np.genfromtxt('data/mono-feedforward-test-phon.csv', delimiter=',')

#
testwords = pd.read_csv('data/mono-test.csv')
trainwords = pd.read_csv('data/mono-train.csv')
#%%

orth = Input(shape=(X.shape[1], ), name='input')
hidden = Dense(100, activation='sigmoid', name='hidden')(orth)
phon = Dense(Y.shape[1], activation='sigmoid', name='phon')(hidden)
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]

model = Model(inputs=orth, outputs=phon, name='mono-feedforward')

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=metric)
model.summary()

# %% at four cycles, this takes about 8 minutes
start = time.time()

items = open('../outputs/mono/feedforward/item-data.csv', 'w')
items.write('epoch,word,accuracy,loss,train_test\n')

CYCLES = 4
EPOCHS = 80
for cycle in range(1, CYCLES+1):
    model.fit(X, Y, batch_size=120, epochs=EPOCHS, validation_split=0, sample_weight=np.array(trainwords['freq_scaled']))

    yhat_train = np.empty(Y.shape)
    yhat_test = np.empty(Y_test.shape)

    for i, xy in enumerate(zip(X, Y)):
        word = trainwords.iloc[i]['word']
        epochs = cycle*EPOCHS
        loss, acc = model.evaluate(xy[0].reshape((1, xy[0].shape[0])), xy[1].reshape((1, xy[1].shape[0])))
        items.write('{},{},{},{},{}\n'.format(epochs, word, acc, loss, 'train'))
        yhat = model.predict(xy[0].reshape((1, xy[0].shape[0])))
        yhat_train[i] = yhat

    for i, xy in enumerate(zip(X_test, Y_test)):
        word = testwords.iloc[i]['word']
        epochs = cycle*EPOCHS
        loss, acc = model.evaluate(xy[0].reshape((1, xy[0].shape[0])), xy[1].reshape((1, xy[1].shape[0])))
        items.write('{},{},{},{},{}\n'.format(epochs, word, acc, loss, 'test'))
        yhat = model.predict(xy[0].reshape((1, xy[0].shape[0])))
        yhat_test[i] = yhat

    np.savetxt('../outputs/mono/feedforward/test-outputs-'+str(cycle)+'.csv', yhat_test)
    np.savetxt('../outputs/mono/feedforward/train-outputs-'+str(cycle)+'.csv', yhat_train)

end = time.time()

items.close()
"""Done"""
"""Learntime was {} minutes""".format(round((end-start)/60))
#%%