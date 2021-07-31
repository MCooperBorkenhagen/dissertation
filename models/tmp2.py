#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model as plot
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed

from utilities import load


d = load('../inputs/mono.traindata')
 
#%%
orth_features = d[4]['orth'].shape[2]
phon_features = d[4]['phonSOS'].shape[2]
input1_name = 'orth_input'
input2_name = 'phon_input'
output_name = 'phon_output'
hidden = 400
transfer_function='sigmoid'
optimizer='rmsprop'
loss="categorical_crossentropy"


encoder_inputs = Input(shape=(None, orth_features), name=input1_name)
encoder_mask = Masking(mask_value=9)(encoder_inputs)
encoder = LSTM(hidden, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_mask)
encoder_states = [state_h, state_c]

# output features should be defined as something like Xd.shape[2]
decoder_inputs = Input(shape=(None, phon_features), name=input2_name)



decoder_lstm = LSTM(hidden, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(phon_features, activation=transfer_function, name=output_name)


decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

encoder_inputs = encoder_inputs
encoder_mask = encoder_mask
encoder = encoder
encoder_states = encoder_states
decoder_lstm = decoder_lstm
decoder_inputs = decoder_inputs
decoder_dense = decoder_dense 

# specify metrics
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]


model.compile(optimizer=optimizer, loss=loss, metrics=metric)
#%%