
# %%
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking
import numpy as np


#%%


X = np.load('../../inputs/orth-left.npy')
Xphon = np.load('../../inputs/phon-left.npy')
Y = np.load('../../inputs/phon-left.npy')
Y[0][-1] # should be all nines
#%%
Y[Y==9] = 0 # change the mask value on the outputs because it messes up the accuracy calculation
Y[0][-1] # should be all zeros
#%%
num_encoder_tokens = X.shape[2]
num_decoder_tokens = Xphon.shape[2]
latent_dim = 300
batch_size = 100
epochs = 1
#%%






# %%
# Define an input sequence and process it.

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_inputs_masked = Masking(mask_value=9)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs_masked)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
# we may need to use an embedding layer here in order to mask
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_inputs_masked = Masking(mask_value=9)(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_masked,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='sigmoid')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# train
pointfive = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=pointfive)

model.fit([X, Xphon], Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.20)
#%%