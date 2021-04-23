
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed
import numpy as np
import time
from tensorflow.keras.utils import plot_model as plot
from utilities import printspace, reshape



def nearest_phoneme(a, phonreps, round=True, ties=True, return_array=False):
    """This is the updated version of dists() and is slightly faster than
    the previous method.

    Parameters
    ----------
    a : arr
        A numpy array to be compared with each value of phonreps.

    phonreps : dict
        A dictionary where every key is a string specifying symbolically the
        phoneme it represents, and each value is a numpy array to be compared
        with a.

    ties : bool
        Test to see if ties are present. If a tie is present then an 
        exception will be raised. If set to False, the pairwise comparison
        across values of phonreps a random value for the tying distance if
        ties are present. (default is True)

    round : bool
        Specify whether to round the input array or not prior to calculating
        the pairwise distances with values in phonreps. (default is True)

    return_array : bool
        Return an array representing the closest match to a, or return
        the symbolic string representing that array from phonreps.
        (default is True)

    Returns
    -------
    The phonological representation (array) that is nearest the array a, as
    determined by pairwise comparisons across all values in phonreps using 
    the L2 norm for the distance calculation.

    """
    if round:
        a = np.around(a)
    print('nearest phoneme value for a:', a)
    d = {np.linalg.norm(a-np.array(v)):k for k, v in phonreps.items()}
    mindist = min(d.keys())

    if ties:
        u = [k for k, v in d.items() if k == mindist]
        assert len(u) == 1, 'More than one minumum value for pairwise distances. Ties present.'
    
    if return_array:
        return(d[mindist])
        print(d[mindist])
    elif not return_array:
        for k, v in phonreps.items():
            print('k and v', k, v)
            if v == d[mindist]:
                return(k)
                print(k)

class Learner():

    def __init__(self, input_features, output_features, orthreps=None, phonreps=None, traindata=None, phon_weights=None, output_weights=None, freeze_phon=False, modelname='EncoderDecoder3', verbose=True, op_names=True, hidden=300, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", accuracy='binary', seed=886, devices=True, memory_growth=True):


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
        self.orthreps = orthreps
        self.phonreps = phonreps

        # learner:
        if op_names:
            input1_name = 'orth_input'
            input2_name = 'phon_input'
            output_name = 'phon_output'
        else:
            input1_name = 'input_1'
            input2_name = 'input_2'
            output_name = 'output'


        self.words = [word for traindata_ in traindata.values() for word in traindata_['wordlist']]

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


    #def fit(self, X, Y, batch_size=100, epochs=10, train_proportion=.9, verbose=True):
    #    t1 = time.time()
    #    # remember that X has to be a list of structure [encoder_inputs, decoder_inputs]
    #    self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))
    #    learntime = round((time.time()-t1)/60, 2)
        #self.model.history.history['learntime'] = learntime
    #    if verbose:
    #        print(learntime, 'minutes elapsed during learning')


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


    def get_word(self, word, traindata=None, construct=True):

        if traindata is None:
            traindata = self.traindata


        assert word in self.words, 'The word you want is not in your reps. To read this word, set construct=True'

        for length, traindata_ in traindata.items():
            for i, w in enumerate(traindata_['wordlist']):
                return traindata_['orth'][i], traindata_['phonSOS'][i], traindata_['phonEOS'][i]
                    

    def reader(self):
        encoder_inputs = self.encoder_inputs
        encoder_states = self.encoder_states
        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(self.hidden_units,))
        decoder_state_input_c = Input(shape=(self.hidden_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        return encoder_model, decoder_model

    def read(self, word, phonreps=None, verbose=True):
        
        if phonreps is None:
            phonreps = self.phonreps

        Xo, _, Yp = self.get_word(word)


        input_seq = reshape(Xo)

        e, d = self.reader()

        states_value = e.predict(input_seq)
        
        assert Yp.shape[1] == self.output_features, 'You are attempting to construct a phonological output that fails to match your number of output features'
        output_shape = (1, 1, self.output_features)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((output_shape))
        # Populate the first character of target sequence with the start character.
        target_seq[0][0] = phonreps['#']
        print('target seq', target_seq)
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        done = False
        word_produced = ''
        maxlen = Yp.shape[0]

        while not done:
            output_tokens, h, c = d.predict([target_seq] + states_value)

            # produce a phoneme
            print('output tokens', output_tokens)
            print('word produced', word_produced)
            print('max length', maxlen)
            print('output tokens type', type(output_tokens))
            print('shape', output_tokens.shape)
            phoneme_produced = nearest_phoneme(output_tokens[0], phonreps=phonreps, return_array=False)
            print('phoneme produced', phoneme_produced)
            sampled_rep = nearest_phoneme(output_tokens, phonreps, return_array=True)
            #sampled_rep = phonreps[phoneme_produced]
            word_produced += phoneme_produced

            # Stop if you find a word boundary or hit maxlen
            if (phoneme_produced == '%' or len(word_produced) > maxlen):
                done = True
            
            #Update the target sequence.
            target_seq = np.zeros((output_shape))
            # repopulate with the predicted rep
            target_seq[0][0] = sampled_rep

            # Update states
            states_value = [h, c]

        if verbose:
            print(word_produced)
        return(word_produced)
            


if __name__ == "__main__":
    Learner()