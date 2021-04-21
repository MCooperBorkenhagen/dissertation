

#%%
from EncoderDecoder3 import Learner
import numpy as np
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from utilities import load
#%%
from utilities import load, decode, reshape, loadreps, pronounce2, dists, nearest_phoneme

#%%

# load
left = load('../inputs/left.traindata')
right = load('../inputs/right.traindata')
#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# %%
orth_features = left[3]['orth'].shape[2]
phon_features = left[3]['phonSOS'].shape[2]

learner = Learner(orth_features, phon_features, devices=False)


#%%
E = 100
for e in range(E):
    print(22*'#')
    print('Training', e, 'of', E, 'cycles')
    print(22*'#')
    for length, subset in left.items():
        print(22*'-')
        print('Cycling on length', length)
        print(22*'-')
        Xo = left[length]['orth']
        Xp = left[length]['phonSOS']
        Y = left[length]['phonEOS']

        learner.fit([Xo, Xp], Y, epochs=5, batch_size=40)
# %% "inference"

encoder_model = Model(learner.encoder_inputs, learner.encoder_states)

decoder_state_input_h = Input(shape=(learner.hidden_units,))
decoder_state_input_c = Input(shape=(learner.hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = learner.decoder_lstm(
    learner.decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = learner.decoder_dense(decoder_outputs)

decoder_model = Model(
    [learner.decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
#%%

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

    d = {np.linalg.norm(a-np.array(v)):k for k, v in phonreps.items()}
    mindist = min(d.keys())

    if ties:
        u = [k for k, v in d.items() if k == mindist]
        assert len(u) == 1, 'More than one minumum value for pairwise distances. Ties present.'
    
    if not return_array:
        return(d[mindist])
    elif return_array:
        for k, v in phonreps.items():
            if k == d[mindist]:
                return(v)


#%%
#def decode_sequence(input_seq):
# Encode the input as state vectors.
length = 3
Xo = left[length]['orth']
Xp = left[length]['phonSOS']
Y = left[length]['phonEOS']


i = 1
target_word = left[length]['wordlist'][i]
input_seq = reshape(Xo[i])
states_value = encoder_model.predict(input_seq)
output_shape = (1, Xp[i].shape[0], Xp[i].shape[1])
# Generate empty target sequence of length 1.
target_seq = np.zeros((output_shape))
# Populate the first character of target sequence with the start character.
target_seq[0][0] = phonreps['#']

# Sampling loop for a batch of sequences
# (to simplify, here we assume a batch of size 1).
stop_condition = False
decoded_sentence = ''
max_decoder_seq_length = output_shape[1]

#%%
timestep = 1
while not stop_condition:
    print(timestep)
    output_tokens, h, c = decoder_model.predict(
        [target_seq] + states_value)

    # Sample a token
    #sampled_token_index = np.argmax(output_tokens[0, -1, :])
    #sampled_char = reverse_target_char_index[sampled_token_index]
    sampled_char = nearest_phoneme(output_tokens, phonreps)
    sampled_rep = phonreps[sampled_char]
    decoded_sentence += sampled_char

    # Exit condition: either hit max length
    # or find stop character.
    if (sampled_char == '%' or len(decoded_sentence) == target_seq.shape[1]):
        stop_condition = True
    
    #Update the target sequence.
    target_seq[0][timestep] = sampled_rep

    # Update states
    states_value = [h, c]
    timestep += 1
    #return decoded_sentence


print(decoded_sentence)
# %%
