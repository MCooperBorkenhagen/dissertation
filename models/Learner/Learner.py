
import numpy as np
import time
import random
import tensorflow as tf
from tensorflow.keras.utils import plot_model as plot
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed
from utilities import printspace, reshape, L2, choose, scale, key, mean, get_vowels
import os

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
        exception will be raised or a sample is made from the ties.
        If set to 'stop', an exception is raised if the pairwise comparison 
        across representations yields a tie. The value 'sample' yields a 
        random representation selected from the ties if ties are present.
        The value 'identify' returns 'XX' as the string return (ie, when 
        return_array is False) and an empty representation (default is 'stop')

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

    d = {k:L2(a, v) for k, v in phonreps.items()}
    mindist = min(d.values())
    
    u = [k for k, v in d.items() if v == mindist]

    if ties == 'stop': # selecting stop is equivalent to assuming that you want only a single phoneme to be selected
        assert len(u) == 1, 'More than one minumum value for pairwise distances for phonemes identified. Ties present.'
        s = u[0] # if the exception isn't raised then the selected phoneme is the single element of u  
    elif ties == 'sample':
        if len(u) != 1:
            s = random.sample(u, 1)[0]
        else:
            s = u[0]
    elif ties == 'identify':
        if len(u) != 1:
            s = 'XX'
        else:
            s = u[0]


    if return_array:
        if ties == 'sample' or ties == 'stop':
            return(phonreps[s])
        elif ties == 'identify':
            if s == 'XX':
                e = np.empty(len(phonreps['_']))
                e[:] = np.nan
            else:
                e = phonreps[s]
            return(e) #this option probably won't ever be used, but returns the segment as nans
    elif not return_array:
        return(s)



class Learner():

    def __init__(self, orth_features, phon_features, orthreps=None, phonreps=None, traindata=None, testdata=None, mask_phon=False, phon_weights=None, output_weights=None, freeze_phon=False, modelname='EncoderDecoder3', verbose=True, op_names=True, hidden=400, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", seed=886, devices=True, memory_growth=True):
        
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


        self.trainwords = [word for traindata_ in traindata.values() for word in traindata_['wordlist']]
        self.testwords = [word for testdata_ in testdata.values() for word in testdata_['wordlist']]
        self.words = self.trainwords + self.testwords

        if traindata is not None:
            self.traindata = traindata

        if testdata is not None:
            self.testdata = testdata

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
        self.encoder_outputs = encoder_outputs

        # specify metrics
        metrics = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model

        self.modelname=modelname
        self.orth_features = orth_features
        self.phon_features = phon_features


        if verbose:
            self.model.summary()


    def fitcycle(self, traindata=None, testdata=None, probs=None, return_histories=False, cycles=1, cycle_id='0', batch_size=25, epochs=1, train_proportion=1, verbose=True, K=None, evaluate=False, outpath='.'):

        """Cycle through key, value pairs in traindata and apply fit() at each cycle.

        ----------
        Parameters
        ----------
        
        probs : floats
            Probabilities to be pplied when sampling the sequence of phonological lengths
            contained in the values of traindata. If None, lengths are selected in ascending
            order. This means that if probabilities are not specified, fit() is called over
            examples that start with the (phonetically) shortest words, moving to the longer
            words as you go. (Default is None)


        """
        if traindata is None:
            traindata = self.traindata

        if testdata is None:
            testdata = self.testdata
        
        histories = {}
        cycletime = 0
        
        if evaluate:
            # set up file for item data if evaluate set to True
            itemdata = open(os.path.join(outpath, 'item-data'+ '-' + self.modelname + '-' + cycle_id + '.csv'), 'w')
            itemdata.write("cycle, word, train_test, phonlength, binary_acc, mse, loss\n")
            # set up file for model data if evaluate set to True 
            modeldata = open(os.path.join(outpath, 'model-data'+ '-' + self.modelname + '-' + cycle_id + '.csv'), 'w')
            modeldata.write("cycle, train_test, phonlength, binary_acc, mse, loss\n")

        if probs is None:
            phonlens = traindata.keys()

        for cycle in range(cycles):
            printspace(1)
            print('Training', cycle+1, 'of', cycles, 'cycles')
            printspace(1)
            history_ = {}

            if probs is not None:
                phonlens = choose(list(traindata.keys()), len(traindata.keys()), probs)

            for length in phonlens:
                printspace(1, symbol='-')
                print('Cycling on phonological length', length)
                printspace(1, symbol='-')
                Xo = traindata[length]['orth']
                Xp = traindata[length]['phonSOS']
                Y = traindata[length]['phonEOS']
                t1 = time.time()
                if K is not None:
                    sample_weights = scale(traindata[length]['frequency'], K)
                elif K is None:
                    sample_weights=None
                cb = self.model.fit([Xo, Xp], Y, epochs=epochs, batch_size=batch_size, validation_split=1-train_proportion, sample_weight=sample_weights, verbose=verbose, callbacks=[tensorboard_callback])
                cb.history['learntime'] = round((time.time()-t1)/60, 2)
                cycletime += cb.history['learntime']
                history_[length] = cb.history
            histories[cycle] = history_

            if evaluate:
                if cycle+1 == cycles: # calculates on the last cycle
                    print('Generating evaluation data at end of cycle. It will take a minute...')
                    for length in traindata.keys():
                        Xo = traindata[length]['orth']
                        Xp = traindata[length]['phonSOS']
                        Y = traindata[length]['phonEOS']
                        for i, word in enumerate(traindata[length]['wordlist']):
                            loss, binary_acc, mse = self.model.evaluate([reshape(Xo[i]), reshape(Xp[i])], reshape(Y[i]), verbose=False)
                            itemdata.write("{}, {}, {}, {}, {}, {}, {}\n".format(cycle+1, word, 'train', length, binary_acc, mse, loss))
                        loss, binary_acc, mse = self.model.evaluate([Xo, Xp], Y, batch_size=Y.shape[0])
                        modeldata.write("{}, {}, {}, {}, {}, {}\n".format(cycle+1, 'train', length, binary_acc, mse, loss))
                    for length in testdata.keys():
                        Xo = testdata[length]['orth']
                        Xp = testdata[length]['phonSOS']
                        Y = testdata[length]['phonEOS']
                        for i, word in enumerate(testdata[length]['wordlist']):
                            loss, binary_acc, mse = self.model.evaluate([reshape(Xo[i]), reshape(Xp[i])], reshape(Y[i]), verbose=False)
                            itemdata.write("{}, {}, {}, {}, {}, {}, {}\n".format(cycle+1, word, 'test', length, binary_acc, mse, loss))
                        loss, binary_acc, mse = self.model.evaluate([Xo, Xp], Y, batch_size=Y.shape[0])
                        modeldata.write("{}, {}, {}, {}, {}, {}\n".format(cycle+1, 'test', length, binary_acc, mse, loss))


        if evaluate:
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

    def find(self, word):
        if word in self.trainwords:
            for traindata_ in self.traindata.values():
                for i, w in enumerate(traindata_['wordlist']):
                    if w == word:
                        Xo = traindata_['orth'][i]
                        Xp = traindata_['phonSOS'][i]
                        Yp = traindata_['phonEOS'][i]
                        return Xo, Xp, Yp
        elif word in self.testwords:
            for testdata_ in self.testdata.values():
                for i, w in enumerate(testdata_['wordlist']):
                    if w == word:
                        Xo = testdata_['orth'][i]
                        Xp = testdata_['phonSOS'][i]
                        Yp = testdata_['phonEOS'][i]
                        return Xo, Xp, Yp
        else:
            raise Exception('Word is not present in the training or test pool.')
            
    def get_word(self, word, construct=True):

        if word in self.words:
            Xo, Xp, Yp = self.find(word)

        #assert word in self.trainwords, 'The word you want is not in your reps. To read this word, set construct=True'
        if construct:
            if word not in self.words:
                constructed = np.array([self.orthreps[letter] for letter in word])
                return constructed, None, None
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

    def read(self, word, returns='strings', phonreps=None, ties='stop', verbose=True, construct=True, **kwargs):
        """Read a word after the learner has learned.

        Parameters
        ----------
        word : str
            A string of letters to be read. The letters need to be present
            in orthreps.keys() in order for the word to have a chance to be
            read. If word is not present in Learner.words, then an array
            will be generated and passed to the method for reading, in which
            case the array is constructed within Learner.get_word().

        returns : str
            What type of object do you want returned? By specifying 'strings'
            you will get the list of phonemes specified as strings, and by 
            specifying 'patterns' you will get the vectors that represent 
            those phonemes as distributed patterns. Note that there is an
            asymmetry in this return process: if you specify strings you get
            the nearest phonemes based on the patterns produced, and if you
            specify 'patterns' you get the actual patterns produced. This is
            because specifying 'strings' only makes sense if you want the nearest
            ones to the patterns produced, which is useful if you want the
            output to be human readable. (Default is 'strings')

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
            are 'stop', 'sample', 'identify', where specifying 'stop' halts the execution
            of the method, 'sample' randomly samples from the alternative
            phonemes selected, and 'identify' puts a placeholder for that
            segment to identify it as a tie (placeholder is 'XX'). (Default is 'stop')

        verbose : bool
            If True the resulting sequence of words is printed. (Default is True)

        construct : bool
            If True, create the orthographic form of the word if it isn't in
            the traindata object. In this case, the word cannot be evaluated unless
            you specify a target phonological pattern. (Default is True)
            
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
        word_repd = []
        

        while not done_reading:
            output_tokens, h, c = d.predict([target_seq] + states_value)
            phoneme_produced = nearest_phoneme(output_tokens, phonreps=phonreps, ties=ties, return_array=False)
            sampled_rep = nearest_phoneme(output_tokens, phonreps, ties='sample', return_array=True)
            word_read.append(phoneme_produced)
            word_repd.append(output_tokens.reshape(self.phon_features))
            
            if (phoneme_produced == '%' or len(word_read) == maxlen):
                done_reading = True
            
            target_seq = np.zeros((output_shape))
            target_seq[0][0] = sampled_rep
            states_value = [h, c]

        if verbose:
            print(word_read)

        if returns == 'strings':
            return(word_read)
        elif returns == 'patterns':
            return(np.array(word_repd))


    def test(self, word, target=None, returns='all', return_phonform=True, phonreps=None, ties='stop', construct=True, verbose=False):
        """Test a word after the learner has learned.

        Parameters
        ----------
        word : str
            An orthographic form to be tested.

        return_phonform : bool
            Specify whether or not the phonological form (list of strings) should
            be returned along with test data. If true, it is returned as the
            first element of the return object. (Default is True)

        phonreps : dict
            A dictionary with symbolic phoneme strings as keys and binary 
            feature vectors as values. This object can be provided and if 
            provided will be used in place of the analagous dictionary of
            representations in Learner.phonreps. This parameter is passed 
            to read(). (Default is None)

        ties : str
            Specify the behavior of the method in the presence of ties produced
            for phonemes (ie, if the most plausible phoneme produced isn't a
            single phoneme but several different phonemes, none of which can be
            identified as the least distant phonemes over all phonemes). The value
            specified is passed to the nearest_phoneme() method. Possible values 
            are 'stop', 'sample', 'identify', where specifying 'stop' halts the execution
            of the method, 'sample' randomly samples from the alternative
            phonemes selected, and 'identify' puts a placeholder for that
            segment to identify it as a tie (placeholder is 'XX'). (Default is 'stop')


        construct : bool
            If True, create the orthographic form of the word if it isn't in
            the traindata object. In this case, the word cannot be evaluated unless
            you specify a target phonological pattern. (Default is True)

        returns : str
            Specify the form of the test to be returned. If a distance measure is
            specified as the test, then an L2 norm is used (so that there is no
            penalty for length). Possible values are 'phonemes-right', which specifies 
            how many phonemes were right, 'phonemes-wrong', which specifies how many 
            phonemes were wrong, 'phonemes-proportion', which specifies the proportion 
            of phonemes that were right, 'phonemes-sum', which specifies the sum of all 
            phonemewise distances between the produced phoneme and the right one,
            'phonemes-average', which calculates the average distances between the right
            phoneme and the produced one, 'phoneme-distances', which is a list of all 
            the phonemewise distances, 'stress', which tests the stress pattern only
            returning how much of the stress pattern was correct as a proportion, 
            'length', which returns the discrepancy in length between the word produced
            and the target word, 'word', which calculates the distance between the right 
            word and the produced one (both padded to the longest word in the training data), 
            or 'all' which returns all seven tests as a tuple. (Default is 'word')
            
        Returns
        -------
        Int, float, or tuple depending on the value provided for "returns" above.

        """
        if target is None:
            if word not in self.words:
                raise Exception('Target not specified and the word is not in training or test pool. To test, provide target.')
            else:
                xo, xp, yp = self.find(word)
        else:
            yp = target

        if return_phonform:
            word_read = self.read(word, phonreps=phonreps, returns='strings', ties=ties, verbose=False, construct=construct)

        if phonreps is None:
            phonreps = self.phonreps

        word_cmu = [key(phonreps, list(e)) for e in yp]



        word_repd = self.read(word, phonreps=phonreps, returns='patterns', ties=ties, verbose=False, construct=construct)

        # the produced word can keep '_' at the end, only if it is a different length than the target word
        if word_read[-1] == '%' or word_read[-1] == '_':
            if len(word_read) == len(word_cmu):
                __ = word_read.pop()
            else:
                if word_read[-1] == '%':
                    __ = word_read.pop()

        terminal = word_cmu.pop() # remove the EOS terminal element
        assert terminal == '%', 'The last element of your target string for test is not correct. Check target and check reps.'

        phonemes_right = len([True for e in zip(word_cmu, word_read) if e[0]==e[1]])
        phonemes_wrong = len(word_cmu)-phonemes_right
        phonemes_proportion = phonemes_right/len(word_cmu) # how much of the word it should have produced did it get right
        how_much_longer = len(word_read)-len(word_cmu)
        phoneme_dists = [L2(e[0], e[1]) for e in zip(yp, word_repd)]
        phonemes_sum = sum(phoneme_dists)
        phonemes_average = mean(phoneme_dists)
        target_vowels = get_vowels(word_cmu, index=False)
        read_vowels = get_vowels(word_read, index=False)
        stress_right = [True for e in zip(target_vowels, read_vowels) if e[0][-1] == e[1][-1]]
        stress = len(stress_right)/len(target_vowels)

        if verbose:
            print(word)
            print('word read:', word_read)
            print('word in cmu:', word_cmu)

        if not word_repd.shape[0] == yp.shape[0]:
            maxlen = max(self.traindata.keys())
            pad_target = self.phonreps['_']*(maxlen-yp.shape[0])
            pad_repd = self.phonreps['_']*(maxlen-word_repd.shape[0])
            word_repd_padded = np.append(word_repd.flatten(), pad_repd)
            target_padded = np.append(yp.flatten(), pad_target)
            wordwise_dist = L2(word_repd_padded, target_padded)
        else:
            wordwise_dist = L2(word_repd.flatten(), yp.flatten())

        if returns == 'phonemes-right':
            if return_phonform:
                return word_read, phonemes_right
            else:
                return phonemes_right
        elif returns == 'phonemes-wrong':
            if return_phonform:
                return word_read, phonemes_wrong
            else:
                return phonemes_wrong
        elif returns == 'phonemes-proportion':
            if return_phonform:
                return word_read, phonemes_proportion
            else:
                return phonemes_proportion
        elif returns == 'phonemes-sum':
            if return_phonform:
                return word_read, phonemes_sum
            else:
                return phonemes_sum
        elif returns == 'phonemes-avg':
            if return_phonform:
                return word_read, phonemes_average
            else:
                return phonemes_average
        elif returns == 'phoneme-distances':
            if return_phonform:
                return word_read, phoneme_dists
            else:
                return phoneme_dists
        elif returns == 'stress':
            if return_phonform:
                return word_read, stress
            else:
                return stress
        elif returns == 'length':
            if return_phonform:
                return word_read, how_much_longer
            else:
                return how_much_longer
        elif returns == 'word':
            if return_phonform:
                return word_read, wordwise_dist
            else:
                return wordwise_dist
        elif returns == 'all':
            if return_phonform:
                return word_read, phonemes_right, phonemes_wrong, phonemes_proportion, phonemes_sum, phonemes_average, phoneme_dists, stress, wordwise_dist 
            else:
                return phonemes_right, phonemes_wrong, phonemes_proportion, phonemes_sum, phonemes_average, phoneme_dists, stress, wordwise_dist
        else:
            raise Exception('Provide an appropriate value for returns parameter in test() in order to get data for your test.')



if __name__ == "__main__":
    Learner()