#!/usr/bin/env python3



"""
Establish and run a phonological autoencoder for words of varying
legths using an LSTM architecture.
"""


# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras import layers
import time
from utilities import pad


tf.debugging.set_log_device_placement(True) # determine where the process is being run

# %% tensorboard
# After cell run
# access tensorboard through: $ tensorboard --logdir logs/train
logs = './logs'
%load_ext tensorboard
%tensorboard --logdir {logs}
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1)

# %%
# on cmu data
from CMUdata import CMUdata
cmu = CMUdata(phonpath='../inputs/')
X, labels = cmu.cmudict_array(return_labels=True, maxphon=5)




# %%
# learner parameters
HIDDEN_SIZE = 500
EPOCHS = 50


# %% Learner
model = Sequential()
model.add(LSTM(HIDDEN_SIZE, input_shape=(X.shape[1], X.shape[2]), activation='tanh', name = 'input'))
model.add(RepeatVector(X.shape[1]))
model.add(LSTM(HIDDEN_SIZE, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(X.shape[2], activation='sigmoid')))
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy(name = 'binary_accuracy', dtype = None, threshold=0.5)])
model.summary()

# set time
start = time.time()
# construct model
model.fit(X, X, epochs=EPOCHS, callbacks=[tb_cb], validation_split=.20)


# time
learntime = round((time.time()-start)/60, 2)
print('Model fit took', learntime, 'minutes')
print(model)
# %%
