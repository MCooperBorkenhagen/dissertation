
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, TimeDistributed
from utilities import changepad
import numpy as np
import time

class Learner():

    def __init__(self, X, Y, labels=None, train_proportion=.9, hidden=300, batch_size=100, transfer_state=False, epochs=20, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", accuracy='binary', seed=886, devices=True, memory_growth=True):


        np.random.seed(seed)
        if type(X) == str:
            X = np.load(X)

        if type(Y) == str:
            Y = np.load(Y)

        if devices:
            tf.debugging.set_log_device_placement(True)
        else:
            pass

        if memory_growth:
            devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(devices[0], enable=True)


        
        self.labels = labels
        self.X = X
        self.Y = Y


        # set as attrbutes a number of important input parameters to init:
        self.hidden_units = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.loss_function = loss
        self.seed = seed



        # learner:

        orth_inputs = Input(shape=(None, X.shape[2]), name='orth_input')
        orth_inputs_masked = Masking(mask_value=9)(orth_inputs)
        orth = LSTM(hidden, return_state=True, return_sequences=False, name = 'orthographic_lstm') # set return_sequences to True if no RepeatVector
        orth_outputs, state_h, state_c = orth(orth_inputs_masked)
        orth_states = [state_h, state_c]

        repeat_outputs = RepeatVector(Y.shape[1])(orth_outputs)

        phon_lstm = LSTM(hidden, return_sequences=True, return_state=True)
        if transfer_state:
            phon_outputs, _, _ = phon_lstm(repeat_outputs, initial_state=orth_states)
        else:
            phon_outputs, _, _ = phon_lstm(repeat_outputs)
        phon_dense = TimeDistributed(Dense(Y.shape[2], activation=transfer_function), name='phon_output')
        #phon_dense = Dense(Y.shape[2], activation=transfer_function, name='phon_output')
        phon_outputs = phon_dense(phon_outputs)

        model = Model(orth_inputs, phon_outputs)


        # train
        if accuracy == 'binary':
            metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]
        else:
            metric = accuracy

        model.compile(optimizer=optimizer, loss=loss, metrics=metric)
        

        self.model = model
        self.summary = model.summary()

        t1 = time.time()
        cb = model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))
        cb.history['learntime'] = round((time.time()-t1)/60, 2)
        self.runtime = time.ctime()
        self.history = cb.history
        self.model = model



    def evaluate(self, X, Y):        
        return(self.model.evaluate(X, Y))



if __name__ == "__main__":
    Learner()