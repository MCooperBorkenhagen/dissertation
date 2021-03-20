
# %%
from EncoderDecoder import Learner
import numpy as np
from utilities import changepad, key, decode, reshape, loadreps, test_acts, all_equal, cor_acts
import pandas as pd
from tensorflow.keras.utils import plot_model as plot

import keras
from keras import Input, Model
from tensorflow.keras import layers
from scipy.spatial.distance import pdist as dist
from scipy.spatial.distance import squareform as matrix
#%%

# get words for reference
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

############
# PATTERNS #
############
#%% load
Xo_ = np.load('../../inputs/orth-left.npy')
Xp_Sos = np.load('../../inputs/phon-sos-left.npy')
Yp_Eos = np.load('../../inputs/phon-eos-left.npy')
Yp_Eos = changepad(Yp_Eos, old=9, new=0)
phonrepsT = loadreps('../../inputs/phonreps-with-terminals.json', changepad=True)

# %% learn
leftT = Learner(Xo_, Xp_Sos, Yp_Eos, epochs=50, hidden=600, devices=False, monitor=False)

#%% decode
dummy = np.zeros((1, Xp_Sos.shape[1], Xp_Sos.shape[2]))
i = 3
testword = words[i]
xo = reshape(Xo_[i])
xp = reshape(Xp_Sos[i])
out = leftT.model.predict([xo, dummy])


#%%
print(testword)
print('Predicted: ', decode(out, phonrepsT))


print('Actual: ', decode(Yp_Eos[i], phonrepsT))
# %%


#%%
encoder_model = Model(leftT.encoder_inputs, leftT.encoder_states)

decoder_state_input_h = Input(shape=(leftT.hidden_units,))
decoder_state_input_c = Input(shape=(leftT.hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

#%%

decoder_outputs, state_h, state_c = leftT.decoder_lstm(
    leftT.decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = leftT.decoder_dense(decoder_outputs)
#%%
decoder_model = Model([leftT.decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

#%%

max_decoder_seq_length = Yp_Eos.shape[2]
    
    # Encode the input as state vectors.
states_value = encoder_model.predict(reshape(Xo_[i]))


#%%

    # Generate empty target sequence of length 1.
target_seq = np.zeros((1, Yp_Eos.shape[1], Yp_Eos.shape[2]))
#%%
    # Populate the first character of target sequence with the start character.
target_seq[0][0] = phonrepsT['#']

#%%
# Sampling loop for a batch of sequences
# (to simplify, here we assume a batch of size 1).
stop_condition = False
decoded_y = []
#while not stop_condition:
for i in range(1):
    output_tokens, h, c = decoder_model.predict(
        [target_seq] + states_value)



#%%
    # Sample a token
    sampled_phoneme = dists(target_seq, phonrepsT)
    decoded_y.append(sampled_phoneme)

    # Exit condition: either hit max length
    # or find stop character.
    if (sampled_phoneme == '%' or
    len(decoded_y) > max_decoder_seq_length):
        stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1, Y.shape[2]))
    target_seq[0, 0, sampled_token_index] = 1.

    # Update states
    states_value = [h, c]

print(decoded_y)

# %%
