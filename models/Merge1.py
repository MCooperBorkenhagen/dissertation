
#%%
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, TimeDistributed, Concatenate, Bidirectional
from utilities import changepad, loadreps, decode, reshape 
import numpy as np
import time
import pandas as pd

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#%%
from tensorflow.keras.utils import plot_model as plot


#def morepad(a, value=9):
#   for i in 


#%%
Xo = np.load('../../inputs/orth-left.npy')
Xp = np.load('../../inputs/phon-sos-left.npy')
#%%
orthshape = Xo.shape
phonshape = Xp.shape


Xo_dummy = np.zeros(orthshape)
Xp_dummy = np.zeros(phonshape)

Xo_mask = np.where(Xo==1, 0, Xo)
Xp_mask = np.where(Xp==1, 0, Xp)
#%%
Y = np.load('../../inputs/phon-eos-left.npy')
Y = changepad(Y, old=9, new=0)

words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

phonreps = loadreps('../../inputs/phonreps-with-terminals.json', changepad=True)
orthreps = loadreps('../../inputs/raw/orthreps.json', changepad=True)

#%%

# params
hidden = 300
optimizer='rmsprop'
loss="categorical_crossentropy"
transfer_function = 'sigmoid'

# %% learner
orth_inputs = Input(shape=(None, Xo.shape[2]), name='orth_input')
orth_inputs_masked = Masking(mask_value=9, name='orth_mask')(orth_inputs)
orth = LSTM(hidden, return_sequences=True, return_state=True, name = 'orthographic_lstm') # set return_sequences to True if no RepeatVector
orth_outputs, orth_hidden, orth_cell = orth(orth_inputs_masked)
orth_state = [orth_hidden, orth_cell]

orth_dense = Dense(500, activation=transfer_function, name='orth_dense')
orth_decomposed = orth_dense(orth_outputs)
orth_decomposed_masked = Masking(mask_value=9, name='orth_mask2')(orth_decomposed)


phon_inputs = Input(shape=(None, Xp.shape[2]), name='phon_input')
phon_inputs_masked = Masking(mask_value=9, name='phon_mask')(phon_inputs)
phon = LSTM(hidden, return_sequences=True, return_state=True, name='phonological_lstm')
phon_outputs, phon_hidden, phon_cell = phon(phon_inputs_masked, initial_state=orth_state)
phon_state = [phon_hidden, phon_cell]


phon_dense = Dense(500, activation=transfer_function, name='phon_dense')
phon_decomposed = phon_dense(phon_outputs)


#%%
merge = Concatenate(name='merge')
merge_outputs = merge([orth_decomposed_masked, phon_decomposed])
#%%

deep = Dense(250, activation='sigmoid', name='deep_layer')
deep_outputs = deep(merge_outputs)

#%%
phon_output = Dense(Xp.shape[2], activation='sigmoid', name='phon_output')
output_layer = phon_output(deep_outputs)
model = Model([orth_inputs, phon_inputs], output_layer)
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]
model.compile(optimizer=optimizer, loss=loss, metrics=metric)
model.summary()

t1 = time.time()
model.fit([Xo, Xp], Y, batch_size=100, epochs=10, validation_split=(1-.9))

learntime = round((time.time()-t1)/60, 2)
# %%
plot(model, to_file='merge1.png')
#%%
# inspect a prediction

    

pronounce(5987, model, Xo_mask, Xp, Y, labels=words, reps=phonreps)
#%%
model.fit([Xo, Xp_mask], Y, batch_size=100, epochs=30, validation_split=(1-.9))


# %% predicting phon from orth using the "inference" method from machine translation


# encoder_inputs = orth_inputs
# encoder_states = orth_state
# latent_dim = hidden
# decoder_inputs = phon_inputs
# decoder_lstm = phon
# decoder_dense = phon_output

encoder_model = Model(orth_inputs, orth_state)

decoder_state_input_h = Input(shape=(hidden,))
decoder_state_input_c = Input(shape=(hidden,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = phon(phon_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]




decoder_outputs = phon_output(decoder_outputs)
decoder_model = Model(
    [phon_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)





# %%

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence