
from keras import Model
from keras.layers import Input


class Reader():

    def __init__(self, model=None):
        self.model = model
        self.encoder_inputs = model.encoder_inputs
        self.encoder_states = model.encoder_states
        self.hidden_units = model.hidden_units
        self.decoder_lstm = model.decoder_lstm
        self.decoder_inputs = model.decoder_inputs
        self.decoder_dense = model.decoder_dense



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