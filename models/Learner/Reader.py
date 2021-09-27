import tensorflow as tf
from keras import Model
import numpy as np
from keras.layers import Input, LSTM, Dense, Masking


class Reader():

    def __init__(self, orth_weights, phon_weights, output_weights, mask_phon=False, freeze_phon=False, modelname='learner', verbose=True, op_names=True, transfer_function='sigmoid', optimizer='rmsprop', loss="binary_crossentropy", seed=886, devices=True, memory_growth=True):
            
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

