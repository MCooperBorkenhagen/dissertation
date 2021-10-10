import tensorflow as tf
from keras import Model
import numpy as np
from keras.layers import Input, LSTM, Dense, Masking
from utilities import reshape, L2
import random


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

class Reader():

    def __init__(self, orth_weights, phon_weights, output_weights, orthreps=None, phonreps=None, mask_phon=False, freeze_phon=False, modelname='reader', verbose=True, op_names=True, transfer_function='sigmoid', optimizer='rmsprop', loss="binary_crossentropy", seed=886, devices=True, memory_growth=True):
            
        """Initialize your Learner() with some values and methods.

        Parameters
        ----------
        orth_features : int

        """

        self.orth_features = orth_weights[0].shape[0]
        self.phon_features = phon_weights[0].shape[0]
        self.hidden_units = orth_weights[1].shape[0]

        np.random.seed(seed)

        if devices:
            tf.debugging.set_log_device_placement(True)

        if memory_growth:
            devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(devices[0], enable=True)

        # set as attrbutes a number of important input parameters to init:
        self.orthreps = orthreps
        self.phonreps = phonreps

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

        # input features should be defined as something like Xe.shape[2]
        encoder_inputs = Input(shape=(None, self.orth_features), name=input1_name)
        encoder_mask = Masking(mask_value=9)(encoder_inputs)
        encoder = LSTM(self.hidden_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_mask)
        encoder_states = [state_h, state_c]

        # output features should be defined as something like Xd.shape[2]
        decoder_inputs = Input(shape=(None, self.phon_features), name=input2_name)

        if mask_phon:
            decoder_inputs_masked = Masking(mask_value=9)(decoder_inputs)
        elif not mask_phon:
            print('Note: learner architecture implemented with no phonological mask')


        decoder_lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True)
        
        if freeze_phon:
            decoder_lstm.trainable = False

        
        if mask_phon:
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs_masked, initial_state=encoder_states)
        elif not mask_phon:
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
        decoder_dense = Dense(self.phon_features, activation=transfer_function, name=output_name)
        
        decoder_outputs = decoder_dense(decoder_outputs)


        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

                
        encoder.set_weights(orth_weights)
        decoder_lstm.set_weights(phon_weights)
        decoder_dense.set_weights(output_weights)


        self.encoder_inputs = encoder_inputs
        self.encoder_mask = encoder_mask
        self.encoder = encoder
        self.encoder_states = encoder_states
        self.decoder_lstm = decoder_lstm
        self.decoder_inputs = decoder_inputs
        self.decoder_dense = decoder_dense
        self.encoder_outputs = encoder_outputs
        self.orth_reader = Model(self.encoder_inputs, self.encoder_states)

        # specify metrics
        metrics = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model

        self.modelname=modelname

        if verbose:
            self.model.summary()




    def reader(self):

        orth_reader = self.orth_reader
        phon_state_input_h = Input(shape=(self.hidden_units,))
        phon_state_input_c = Input(shape=(self.hidden_units,))
        phon_states_inputs = [phon_state_input_h, phon_state_input_c]
        phon_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs, initial_state=phon_states_inputs)
        phon_states = [state_h, state_c]
        phon_outputs = self.decoder_dense(phon_outputs)
        phon_reader = Model([self.decoder_inputs] + phon_states_inputs, [phon_outputs] + phon_states)
        return orth_reader, phon_reader

    def orth_states(self, xo):

        """Generate the hidden orthographic states for a an orthographic input
        
        Parameters
        ----------
        xo : arr
            An orthographic input pattern, where each timestep
            corresponds to a letter.

        Returns
        -------
        arr
            A hidden state output is returned, representing the
            hidden state resulting from the trained weights passed
            at class initialization.

        """

        return self.orth_reader.predict(xo)


    def make_orth(self, word, orthreps=None):

        if orthreps is None:
            orthreps = self.orthreps
        constructed = np.array([orthreps[letter] for letter in word])
        return constructed



    def make_phon(self, cmu, phonreps=None):
        if phonreps is None:
            phonreps = self.phonreps
        r = np.array([phonreps[phone] for phone in cmu])

        # add terminal elements:
        SOS = np.insert(r, 0, phonreps['#'], axis=0)
        EOS = np.insert(r, r.shape[0], phonreps['%'], axis=0)

        return SOS, EOS


    def read(self, word, cmu, returns='strings', orthreps=None, phonreps=None, ties='stop', verbose=True, construct=True, **kwargs):
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

        if orthreps is None:
            orthreps = self.orthreps

        if phonreps is None:
            phonreps = self.phonreps


        Xo = self.make_orth(word, orthreps=orthreps)
        __, Yp = self.make_phon(cmu, phonreps=phonreps)

        input_seq = reshape(Xo)

        e, d = self.reader()

        states_value = e.predict(input_seq)

        assert Yp.shape[1] == len(phonreps['#']), 'You have a phonological output that fails to match your number of output features'
        maxlen = Yp.shape[0]

        output_shape = (1, 1, Yp.shape[1])

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
            word_repd.append(output_tokens.reshape(Yp.shape[1]))
            
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





if __name__ == "__main__":
    Reader()