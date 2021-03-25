
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed
import numpy as np
import time

"""
This learner reverses the state passing scheme used by the traditional
encoder-decoder architecture, where here the phonological LSTM
passes its state to the orthographic LSTM. (As of 3/21/21 implementation
this leads to less-than-perfect performance.

"""


class Learner():

    def __init__(self, Xp, Xo, Yp, labels=None, op_names=True, train_proportion=.9, hidden=300, batch_size=100, epochs=20, time_distributed=False, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", accuracy='binary', seed=886, devices=True, memory_growth=True):


        np.random.seed(seed)
        if type(Xp) == str:
            Xp = np.load(Xp)
        if type(Xo) == str:
            Xo = np.load(Xo)
        if type(Yp) == str:
            Yp = np.load(Yp)

        if devices:
            tf.debugging.set_log_device_placement(True)
        else:
            pass

        if memory_growth:
            devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(devices[0], enable=True)


        
        self.labels = labels
        self.Xp = Xp
        self.Xo = Xo
        self.Yp = Yp


        # set as attrbutes a number of important input parameters to init:
        self.hidden_units = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.loss_function = loss
        self.seed = seed



        # learner:
        if op_names:
            input1_name = 'phon_input'
            input2_name = 'orth_input'
            output_name = 'phon_output'
        else:
            input1_name = 'input_1'
            input2_name = 'input_2'
            output_name = 'output'

        encoder_inputs = Input(shape=(None, Xp.shape[2]), name=input1_name)
        encoder_inputs_masked = Masking(mask_value=9)(encoder_inputs)
        encoder = LSTM(hidden, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs_masked)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, Xo.shape[2]), name=input2_name)
        decoder_inputs_masked = Masking(mask_value=9)(decoder_inputs)

        decoder_lstm = LSTM(hidden, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs_masked,
                                            initial_state=encoder_states)
        if time_distributed:
            decoder_dense = TimeDistributed(Dense(Xp.shape[2], activation=transfer_function), name=output_name)
        else:
            decoder_dense = Dense(Xp.shape[2], activation=transfer_function, name=output_name)
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.encoder_inputs = encoder_inputs
        self.encoder_states = encoder_states
        self.decoder_lstm = decoder_lstm
        self.decoder_inputs = decoder_inputs
        self.decoder_dense = decoder_dense 


        # train
        if accuracy == 'binary':
            metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]
        else:
            metric = accuracy

        model.compile(optimizer=optimizer, loss=loss, metrics=metric)
        

        self.model = model
        self.summary = model.summary()

        t1 = time.time()
        cb = model.fit([Xp, Xo], Yp, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))
        cb.history['learntime'] = round((time.time()-t1)/60, 2)
        self.runtime = time.ctime()
        self.history = cb.history
        self.model = model



    def evaluate(self, X, Yp):        
        return(self.model.evaluate(X, Yp))



if __name__ == "__main__":
    Learner()