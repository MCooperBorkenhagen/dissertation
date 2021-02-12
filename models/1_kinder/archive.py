

# %%
import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras.models import Sequential
from keras import layers

# utilities:
from utilities import divide
import time
import json




# %%
class Learner():


    def __init__(self, X, Y, labels=None, hidden_layers=1, hidden=900, batch_size=250, epochs=100, validation=.1, transfer_function='sigmoid', optimizer='adam', loss=None, monitor=True, seed=451):

        tf.debugging.set_log_device_placement(monitor)    
        np.random.seed(seed)

        # data
        if type(X) is np.ndarray:
            self.X = X
        else:
            self.X = np.load(X)

        if type(Y) is np.ndarray:
            self.Y = Y
        else:
            self.Y = np.load(Y)

        self.labels = labels

        # set as attrbutes a number of important input parameters to init:
        self.number_hidden_layers = hidden_layers
        self.number_hidden_units = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.loss_function = loss
        self.seed = seed


        # construct, compile model at class call:
        SEQ = layers.LSTM
        model = Sequential()
        model.add(SEQ(hidden, input_shape=X[0].shape, name = 'orthographic_input'))
        # As the decoder RNN's input, repeatedly provide with the last output of
        # RNN for each time step. Repeat # phoneme times as that's the maximum
        # length of output
        model.add(layers.RepeatVector(Y.shape[1])) # calculated as number of timesteps on output layer
        # The decoder RNN could be multiple layers stacked or a single layer.
        for _ in range(hidden_layers):
            nm = 'hidden_layer' + str(_)
            # By setting return_sequences to True, return not only the last output
            # but all the outputs so far in the form of (num_samples, timesteps,
            # output_dim). This is necessary as TimeDistributed in the below
            # expects the first dimension to be the timesteps.
            model.add(SEQ(hidden, return_sequences=True, name = nm))

        # Apply a dense layer to the every temporal slice of an input. For each of
        # step of the output sequence, decide which character should be chosen.
        model.add(layers.TimeDistributed(layers.Dense(Y.shape[2], activation=transfer_function), name = 'phonological_output'))
        model.compile(loss=loss, optimizer=optimizer, 
            metrics=[tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)])
        
        self.summary = model.summary()
        #self.history = history
        self.model = model
        # fit upon class call:
"""        t1 = time.time()
        history = model.fit(self.X, self.Y, batch_size=self.batch_size, epochs=self.epochs, validation_split=validation)
        t2 = time.time
        history['learntime'] = round((t2-t1)/60, 2)"""


"""
       

    def evaluate(self, X, Y):        
        return(self.model.evaluate(X, Y))
"""

if __name__ == "__main__":
    Learner()


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


# %%

# train the network
history = model.fit(x_train, y_train, batch_size=cfg['batch_size'], epochs=cfg['epochs'],
            validation_data=(x_test, y_test))

print("Train loss, acc: ", model.evaluate(x_train, y_train))
print("Test loss, acc: ", model.evaluate(x_test, y_test))
print("Minutes since start: {elapsed}".format(elapsed = round((time.time()-t1)/60, 2)))
# %%
