
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed
import numpy as np
import time
from tensorflow.keras.utils import plot_model as plot
from utilities import printspace

class Learner():

    def __init__(self, input_features, output_features, traindata=None, phon_weights=None, output_weights=None, freeze_phon=False, modelname='EncoderDecoder3', verbose=True, op_names=True, hidden=300, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", accuracy='binary', seed=886, devices=True, memory_growth=True):


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

        if traindata is not None:
            self.traindata = traindata

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
        
        if freeze_phon:
            decoder_lstm.trainable = False
        
        if phon_weights is not None:
            decoder_lstm.set_weights(phon_weights)
        
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs_masked, initial_state=encoder_states)
        decoder_dense = Dense(output_features, activation=transfer_function, name=output_name)
        
        if output_weights is not None:
            decoder_dense.set_weights(output_weights)
        
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


    def fit(self, X, Y, batch_size=100, epochs=10, train_proportion=.9, verbose=True):
        t1 = time.time()
        # remember that X has to be a list of structure [encoder_inputs, decoder_inputs]
        cb = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))
        learntime = round((time.time()-t1)/60, 2)
        cb.history['learntime'] = learntime
        if verbose:
            print(learntime, 'minutes elapsed during learning')
        return(cb)

    def fitcycle(self, traindata=None, return_histories=True, cycles=1, batch_size=25, epochs=1, train_proportion=.9, verbose=True):

        if traindata is None:
            traindata = self.traindata
        
        histories = {}
        cycletime = 0
        for cycle in range(cycles):
            printspace(4)
            print('Training', cycle+1, 'of', cycles, 'cycles')
            printspace(4)
            history_ = {}
            for length, subset in traindata.items():
                printspace(1, symbol='-')
                print('Cycling on phonological length', length)
                printspace(1, symbol='-')
                Xo = traindata[length]['orth']
                Xp = traindata[length]['phonSOS']
                Y = traindata[length]['phonEOS']
                cb = self.fit([Xo, Xp], Y, epochs=epochs, batch_size=batch_size, train_proportion=train_proportion, verbose=verbose)
                cycletime += cb.history['learntime']
                history_[length] = cb.history
            histories[cycle] = history_
        
        self.model.history.history['cycletime'] = cycletime
        self.model.history.history['epochs_done'] = cycletime*epochs

        if verbose:
            print(cycletime, 'minutes elapsed since start of the first cycle')

        if return_histories:
            return(histories)

    def summary(self):
        return(self.model.summary())

    def plot(self, PATH=None):
        if PATH is None:
            PATH = self.modelname+'.png'
        else:
            pass
        return(plot(self.model, to_file=PATH))



if __name__ == "__main__":
    Learner()