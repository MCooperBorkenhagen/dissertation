
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed
import numpy as np
import time
from tensorflow.keras.utils import plot_model as plot

class Phon():

    def __init__(self, features, modelname='EncoderDecoder3', verbose=True, op_names=True, hidden=300, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", accuracy='binary', seed=886, devices=True, memory_growth=True):

        np.random.seed(seed)

        if devices:
            tf.debugging.set_log_device_placement(True)

        if memory_growth:
            devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(devices[0], enable=True)

        # set as attrbutes a number of important input parameters to init:
        self.hidden_units = hidden
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.loss_function = loss
        self.seed = seed

        # learner:
        if op_names:
            input1_name = 'orth_input'
            input2_name = 'phon_input'
            output_name = 'phon_output'
        else:
            input1_name = 'input_1'
            input2_name = 'input_2'
            output_name = 'output'

        self.modelname=modelname
        self.input_features = input_features
        self.output_features = output_features

        # input features should be defined as something like Xe.shape[2]
        encoder_inputs = Input(shape=(None, input_features), name=input1_name)
        encoder_inputs_masked = Masking(mask_value=9)(encoder_inputs)
        encoder = LSTM(hidden, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs_masked)
        encoder_states = [state_h, state_c]

        # output features should be defined as something like Xd.shape[2]
        decoder_inputs = Input(shape=(None, output_features), name=input2_name)
        decoder_inputs_masked = Masking(mask_value=9)(decoder_inputs)

        decoder_lstm = LSTM(hidden, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs_masked, initial_state=encoder_states)
        decoder_dense = Dense(output_features, activation=transfer_function, name=output_name)
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.encoder_inputs = encoder_inputs
        self.encoder_states = encoder_states
        self.decoder_lstm = decoder_lstm
        self.decoder_inputs = decoder_inputs
        self.decoder_dense = decoder_dense 

        # specify metrics
        if accuracy == 'binary':
            metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]
        else:
            metric = accuracy

        model.compile(optimizer=optimizer, loss=loss, metrics=metric)
        
        self.model = model
        if verbose:
            self.model.summary()


    def fit(self, X, Y, verbose=True, batch_size=100, epochs=10, train_proportion=.9):
        t1 = time.time()
        # this architecture intends X and Y to be the same, or some near variant of each other (autoencoder)
        cb = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))
        learntime = round((time.time()-t1)/60, 2)
        cb.history['learntime'] = learntime
        if verbose:
            print(learntime, 'minutes elapsed during learning')
        return(cb)

    def summary(self):
        return(self.model.summary())

    def plot(self, PATH=None):
        if PATH is None:
            PATH = self.modelname+'.png'
        else:
            pass
        return(plot(self.model, to_file=PATH))



if __name__ == "__main__":
    Phon()