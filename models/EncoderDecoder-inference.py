
# %%
from EncoderDecoder import Learner
import numpy as np
#%%
from utilities import changepad, key, decode, reshape, loadreps, test_acts, all_equal, cor_acts, pronounce, dists
import pandas as pd
from tensorflow.keras.utils import plot_model as plot
#%%
import keras
#from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input, LSTM, Dense

############
# PATTERNS #
############
#%% load
Xo_ = np.load('../inputs/orth-left.npy')
Xp_ = np.load('../inputs/phon-sos-left.npy')
Yp_ = np.load('../inputs/phon-eos-left.npy')
Yp_ = changepad(Yp_, old=9, new=0)
phonreps = loadreps('../inputs/phonreps-with-terminals.json', changepad=True)
orthreps = loadreps('../inputs/raw/orthreps.json')
words = pd.read_csv('../inputs/encoder-decoder-words.csv', header=None)[0].tolist()
# dummy patterns:
Xo_dummy = np.zeros(Xo_.shape)
Xp_dummy = np.zeros(Xp_.shape)

# %% step one: OP training
hidden = 500
left = Learner(Xo_dummy, Xp_, Yp_, epochs=10, batch_size=100, hidden=hidden, devices=False)
plot(left.model, to_file='encoder-decoder1.png')

# %%
left.model.fit([Xo_, Xp_], Yp_, epochs=10, batch_size=100)
left.model.fit([Xo_, Xp_dummy], Yp_, epochs=50, batch_size=100)

#%% "inference" procedure
encoder_inputs = left.encoder_inputs
encoder_states = left.encoder_states
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(hidden,))
decoder_state_input_c = Input(shape=(hidden,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = left.decoder_lstm(
    left.decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = left.decoder_dense(decoder_outputs)

decoder_model = Model(
    [left.decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

#%%
i = 278
input_seq = Xo_[i]
# Encode the input as state vectors.
states_value = encoder_model.predict(reshape(input_seq))

# Generate empty target sequence of length 1.
target_seq = np.zeros((1, Yp_.shape[1], Yp_.shape[2]))
# Populate the first character of target sequence with the start character.
target_seq[0, 0] = phonreps['#']
# Sampling loop for a batch of sequences
# (to simplify, here we assume a batch of size 1).
word = ''

output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
for timestep in range(1, target_seq.shape[1]):
    predicted_phoneme = dists(output_tokens[0][timestep], phonreps)
    word += predicted_phoneme

print('Predicted:', word)
print('True:', words[i])




# print
#%%
with open('encoder-decoder-items.csv', 'w') as f:
    f.write('word, acc, loss\n')
    for i, word_ in enumerate(words):
        loss_, acc_ = left.model.evaluate([reshape(Xo_[i]), reshape(Xp_[i])], reshape(Yp_[i]))
        f.write('{word:s}, {acc:.8f}, {loss:.8f}\n'.format(
            word = word_,
            acc = acc_,
            loss = loss_))
# %%
