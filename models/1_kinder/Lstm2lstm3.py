
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, TimeDistributed
import numpy as np
import time

class Learner():

    def __init__(self, Xo, Xp, Y, labels=None, train_proportion=.9, hidden=300, batch_size=100, epochs=20, initial_weights=None, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", accuracy='binary', monitor=True, seed=886, devices=True, memory_growth=True):


        np.random.seed(seed)
        if type(Xo) == str:
            Xo = np.load(Xo)

        if type(Xp) == str:
            Xp = np.load(Xp)


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
        self.Xo = Xo
        self.Xp = Xp
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

        orth_inputs = Input(shape=(None, Xo.shape[2]), name='orth_input')
        orth_inputs_masked = Masking(mask_value=9)(orth_inputs)
        orth = LSTM(hidden, return_state=True, return_sequences=True, name = 'orthographic_lstm') # set return_sequences to True if no RepeatVector
        orth_outputs, state_h, state_c = orth(orth_inputs_masked)
        orth_states = [state_h, state_c]

        phon_inputs = Input(shape=(None, Xp.shape[2]), name='phon_input')
        phon_inputs_masked = Masking(mask_value=9)(phon_inputs)

        phon_lstm = LSTM(hidden, return_sequences=True, return_state=True)
        phon_outputs, _, _ = phon_lstm(orth_outputs, initial_state=initial_weights)
        phon_dense = TimeDistributed(Dense(Y.shape[2], activation=transfer_function), name='phon_output')
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
        cb = model.fit([Xo, Xp], Y, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))
        cb.history['learntime'] = round((time.time()-t1)/60, 2)
        self.runtime = time.ctime()
        self.history = cb.history
        self.model = model



    def evaluate(self, X, Y):        
        return(self.model.evaluate(X, Y))


if __name__ == "__main__":
    Learner()