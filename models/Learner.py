
import numpy as np
import time
import random
import tensorflow as tf
from tensorflow.keras.utils import plot_model as plot
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed
from utilities import printspace, reshape, L2


log_dir = "logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def nearest_phoneme(a, phonreps, round=True, ties='stop', return_array=False):
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

    ties : str
        Test to see if ties are present. If a tie is present then an 
        exception will be raised. If set to 'stop', an exception is raised
        if the pairwise comparison across representations yields a tie. The
        alternative is the value 'sample' which yields a random representation
        selected from the ties if ties are present. (default is 'stop')

    round : bool
        Specify whether to round the input array or not prior to calculating
        the pairwise distances with values in phonreps. (default is True)

    return_array : bool
        Return an array representing the closest match to a, or return
        the symbolic string representing that array from phonreps.
        (default is True)

    Returns
    -------
    The phonological representation (array) that is nearest the array a,
    determined by pairwise comparisons across all values in phonreps using 
    the L2 norm for the distance calculation.

    """
    if round:
        a = np.around(a)

    #d = {k:np.linalg.norm(a-np.array(v)) for k, v in phonreps.items()}
    d = {k:L2(a, v) for k, v in phonreps.items()}
    mindist = min(d.values())
    
    u = [k for k, v in d.items() if v == mindist]

    if ties == 'stop': # selecting stop is equivalent to assuming that you want only a single phoneme to be selected
        assert len(u) == 1, 'More than one minumum value for pairwise distances for phonemes identified. Ties present.'
        s = u[0] # if the exception isn't raised then the selected phoneme is the single element of u  
    elif ties == 'sample':
        s = random.sample(u, 1)[0]

    if return_array:
        return(phonreps[s])
    elif not return_array:
        return(s)




class Learner():

    def __init__(self, orth_features, phon_features, orthreps=None, phonreps=None, traindata=None, mask_phon=False, phon_weights=None, output_weights=None, freeze_phon=False, modelname='EncoderDecoder3', verbose=True, op_names=True, hidden=300, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", accuracy='binary', seed=886, devices=True, memory_growth=True):
        
        """Initialize your Learner() with some values and methods.

        Parameters
        ----------
        orth_features : int

        """

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



        # input features should be defined as something like Xe.shape[2]
        encoder_inputs = Input(shape=(None, orth_features), name=input1_name)
        encoder_mask = Masking(mask_value=9)(encoder_inputs)
        encoder = LSTM(hidden, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_mask)
        encoder_states = [state_h, state_c]

        # output features should be defined as something like Xd.shape[2]
        decoder_inputs = Input(shape=(None, phon_features), name=input2_name)

        if mask_phon:
            decoder_inputs_masked = Masking(mask_value=9)(decoder_inputs)
        elif not mask_phon:
            print('Note: learner architecture implemented with no phonological mask')


        decoder_lstm = LSTM(hidden, return_sequences=True, return_state=True)
        
        if freeze_phon:
            decoder_lstm.trainable = False
        
        if phon_weights is not None:
            decoder_lstm.set_weights(phon_weights)
        
        if mask_phon:
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs_masked, initial_state=encoder_states)
        elif not mask_phon:
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
        decoder_dense = Dense(phon_features, activation=transfer_function, name=output_name)
        
        if output_weights is not None:
            decoder_dense.set_weights(output_weights)
        
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.encoder_inputs = encoder_inputs
        self.encoder_mask = encoder_mask
        self.encoder = encoder
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

        self.modelname=modelname
        self.orth_features = orth_features
        self.phon_features = phon_features


        if verbose:
            self.model.summary()


    def fitcycle(self, traindata=None, return_histories=False, cycles=1, cycle_id='0', batch_size=25, epochs=1, train_proportion=1, verbose=True, sample_weights=False, evaluate=False, evaluate_when=4):

        if traindata is None:
            traindata = self.traindata
        
        histories = {}
        cycletime = 0
        
        if evaluate:
            # set up file for item data if evaluate set to True
            itemdata = open('item-data'+ '-' + self.modelname + '-' + cycle_id + '.csv', 'w')
            itemdata.write("cycle, word, phonlength, accuracy, loss\n")
            # set up file for model data if evaluate set to True 
            modeldata = open('model-data'+ self.modelname + '-' + cycle_id + '.csv', 'w')
            modeldata.write("cycle, phonlength, accuracy, loss\n")

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
                t1 = time.time()
                if sample_weights:
                    print('need sample weights')
                    #sample_weights = scale(traindata[length]['frequency']) # need to write scale()
                elif not sample_weights:
                    sample_weights=None
                cb = self.model.fit([Xo, Xp], Y, epochs=epochs, batch_size=batch_size, validation_split=1-train_proportion, sample_weight=sample_weights, verbose=verbose, callbacks=[tensorboard_callback])
                cb.history['learntime'] = round((time.time()-t1)/60, 2)
                cycletime += cb.history['learntime']
                history_[length] = cb.history
            histories[cycle] = history_

            if evaluate:
                if cycle % evaluate_when == 0 or cycle+1 == cycles: # calculates periodically based on evaluate_when
                    print('Generating evaluation data at end of cycle. It will take a minute...')
                    for length, subset in traindata.items():
                        Xo = traindata[length]['orth']
                        Xp = traindata[length]['phonSOS']
                        Y = traindata[length]['phonEOS']
                        for i, word in enumerate(traindata[length]['wordlist']):
                            loss, accuracy = self.model.evaluate([reshape(Xo[i]), reshape(Xp[i])], reshape(Y[i]), verbose=False)
                            itemdata.write("{}, {}, {}, {}, {}\n".format(cycle+1, word, length, accuracy, loss))
                        loss, accuracy = self.model.evaluate([Xo, Xp], Y, batch_size=Y.shape[0])
                        modeldata.write("{}, {}, {}, {}\n".format(cycle+1, length, accuracy, loss))

        itemdata.close()
        modeldata.close()

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

        if word in self.words:
            for length, traindata_ in traindata.items():
                for i, w in enumerate(traindata_['wordlist']):
                    if w == word:
                        Xo = traindata_['orth'][i]
                        Xp = traindata_['phonSOS'][i]
                        Yp = traindata_['phonEOS'][i]

        #assert word in self.words, 'The word you want is not in your reps. To read this word, set construct=True'
        if construct:
            if word not in self.words:
                new_word = np.array([self.orthreps[letter] for letter in word])
                return new_word, None, None
            else:
                return Xo, Xp, Yp
                    
        elif not construct:
            if word not in self.words:
                raise Exception('The word is not in your list of words, and you did not tell me to construct it.')
            else:
                return Xo, Xp, Yp

    def reader(self):

        orth_reader = Model(self.encoder_inputs, self.encoder_states)
        phon_state_input_h = Input(shape=(self.hidden_units,))
        phon_state_input_c = Input(shape=(self.hidden_units,))
        phon_states_inputs = [phon_state_input_h, phon_state_input_c]
        phon_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs, initial_state=phon_states_inputs)
        phon_states = [state_h, state_c]
        phon_outputs = self.decoder_dense(phon_outputs)
        phon_reader = Model([self.decoder_inputs] + phon_states_inputs, [phon_outputs] + phon_states)
        return orth_reader, phon_reader

    def read(self, word, phonreps=None, ties='stop', verbose=True, construct=True):
        """Read a word after the learner has learned.

        Parameters
        ----------
        word : str
            A string of letters to be read. The letters need to be present
            in orthreps.keys() in order for the word to have a chance to be
            read. If word is not present in Learner.words, then an array
            will be generated and passed to the method for reading, in which
            case the array is constructed within Learner.get_word().

        phonreps : dict
            A dictionary with symbolic phoneme strings as keys and binary 
            feature vectors as values. This object can be provided and if 
            provided will be used in place of the analagous dictionary of
            representations in Learner.phonreps.

        ties : str
            Specify the behavior of the method in the presence of ties produced
            for phonemes (ie, if the most plausible phoneme produced isn't a
            single phoneme but several different phonemes, none of which can be
            identified as the least distant phonemes over all phonemes). The value
            specified is passed to the nearest_phoneme() method. Possible values 
            are 'stop' and 'sample', where specifying 'stop' halts the execution
            of the method and 'sample' randomly samples from the alternative
            phonemes selected. (Default is 'stop')

        verbose : bool
            If True the resulting sequence of words is printed. (Default is True)
            
        Returns
        -------
        A list whose elements are the phonemes produced for the orthographic
        sequence provided as param word.

        """
        if phonreps is None:
            phonreps = self.phonreps

        Xo, _, Yp = self.get_word(word, construct=construct)

        input_seq = reshape(Xo)

        e, d = self.reader()

        states_value = e.predict(input_seq)
        if Yp is None:
            maxlen = max([v['phonEOS'].shape[1] for k, v in self.traindata.items()])
        else:
            assert Yp.shape[1] == self.phon_features, 'You are attempting to construct a phonological output that fails to match your number of output features'
            maxlen = Yp.shape[0]

        output_shape = (1, 1, self.phon_features)

        target_seq = np.zeros((output_shape))
        target_seq[0][0] = phonreps['#']

        done_reading = False
        word_read = []
        

        while not done_reading:
            output_tokens, h, c = d.predict([target_seq] + states_value)
            phoneme_produced = nearest_phoneme(output_tokens, phonreps=phonreps, ties=ties, return_array=False)
            sampled_rep = nearest_phoneme(output_tokens, phonreps, ties=ties, return_array=True)
            word_read.append(phoneme_produced)
            
            if (phoneme_produced == '%' or len(word_read) == maxlen):
                done_reading = True
            
            target_seq = np.zeros((output_shape))
            target_seq[0][0] = sampled_rep
            states_value = [h, c]

        if verbose:
            print(word_read)

        return(word_read)
            


if __name__ == "__main__":
    Learner()