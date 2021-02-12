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
class Lstm2lstm():
    def __init__(self, X, Y, labels=None, train_proportion=.9, hidden_layers=1, hidden=900, batch_size=250, epochs=100, transfer_function='sigmoid', optimizer='adam', loss="categorical_crossentropy", monitor=True, seed=451, devices=True):

   
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

        if devices:
            tf.debugging.set_log_device_placement(True)
        else:
            pass
        
        self.labels = labels
        X_train, Y_train, X_test, Y_test = divide(X, Y, train_proportion)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        # set as attrbutes a number of important input parameters to init:
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden
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
        

        t1 = time.time()
        cb = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
        cb.history['learntime'] = round((time.time()-t1)/60, 2)
        self.runtime = time.ctime()
        self.history = cb.history
        self.model = model


    def evaluate(self, X, Y):        
        return(self.model.evaluate(X, Y))


if __name__ == "__main__":
    Lstm2lstm()