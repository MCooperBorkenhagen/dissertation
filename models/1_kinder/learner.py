"""
This is a basic LSTM learner that maps print to speech.
"""
# %%

import numpy as np
import keras
import tensorflow as tf

from utilities import divide
from keras.models import Sequential
from keras import layers
import time
import pandas as pd
import json

# devices:
tf.debugging.set_log_device_placement(True)


t1 = time.time()
seed = 0
np.random.seed(seed)

# load the data
x = np.load("../../inputs/orth_pad_right.npy")
y = np.load("../../inputs/phon_pad_right.npy")
labels = pd.read_csv("../../inputs/syllabics.csv", sep = ",")

# config
with open('./params.json', 'r') as f:
    cfg = json.load(f)


train_frac = 0.9
x_train, y_train, x_test, y_test = divide(x, y, train_frac)


# %%
# build the model
SEQ = layers.LSTM
model = Sequential()

# "Encode" the input sequence using an RNN, producing an output of
# HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).

model.add(SEQ(cfg['hidden_size'], input_shape=x_train[0].shape, name = 'orth'))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat # phoneme times as that's the maximum
# length of output
model.add(layers.RepeatVector(len(y[0])))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(cfg['hidden_layers']):
    nm = 'hidden' + str(_)
    # By setting return_sequences to True, return not only the last output
    # but all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below
    # expects the first dimension to be the timesteps.
    model.add(SEQ(cfg['hidden_size'], return_sequences=True, name = nm))

# Apply a dense layer to the every temporal slice of an input. For each of
# step of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(y[0][0]), activation="sigmoid"), name = 'phon'))
model.compile(loss=cfg['loss'], optimizer="adam",
                metrics=[tf.keras.metrics.BinaryAccuracy(name = cfg['accuracy'], dtype = None, threshold=0.5)])
model.summary()

# train the network
history = model.fit(x_train, y_train, batch_size=cfg['batch_size'], epochs=cfg['epochs'],
            validation_split=cfg['validation_split'])

print("Train loss, acc: ", model.evaluate(x_train, y_train))
print("Test loss, acc: ", model.evaluate(x_test, y_test))
print("Minutes since start: {elapsed}".format(elapsed = round((time.time()-t1)/60, 2)))
# %%
