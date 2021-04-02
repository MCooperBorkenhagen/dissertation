
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
Xp_ = np.load('../inputs/phon-for-eos-left.npy')
Yp_ = np.load('../inputs/phon-eos-left.npy')
Yp_ = changepad(Yp_, old=9, new=0)
phonreps = loadreps('../inputs/phonreps-with-sos-only.json', changepad=True)

orthreps = loadreps('../inputs/raw/orthreps.json')
words = pd.read_csv('../inputs/encoder-decoder-words.csv', header=None)[0].tolist()
#%%
pad = np.array(loadreps('../inputs/phonreps-with-terminals.json', changepad=False)['_'])

#%%

# dummy patterns:
Xo_dummy = np.zeros(Xo_.shape)
Xp_dummy = np.zeros(Xp_.shape)

# %% step one: PP training
hidden = 500
left = Learner(Xo_dummy, Xp_, Yp_, epochs=4, batch_size=100, hidden=hidden, devices=False)
plot(left.model, to_file='encoder-decoder1.png')

# %% Op together, then O alone
left.model.fit([Xo_, Xp_], Yp_, epochs=1, batch_size=100)
left.model.fit([Xo_, Xp_dummy], Yp_, epochs=60, batch_size=100)
#%%
i = 6799
print('with phon inputs only:')
pronounce(i, left.model, Xo_dummy, Xp_, Yp_, labels=words, reps=phonreps)
print('#####################')
print('with orth and phon inputs:')
pronounce(i, left.model, Xo_, Xp_, Yp_, labels=words, reps=phonreps)
print('#####################')
print('with no phon inputs:')
pronounce(i, left.model, Xo_, Xp_dummy, Yp_, labels=words, reps=phonreps)


#%% get a single prediction for an input pattern:
words[i]

out = left.model.predict([reshape(Xo_[i]), reshape(Xp_[i])])

#%%


#%% get accuracy over items
with open('encoder-decoder-items.csv', 'w') as f:
    f.write('word, acc, loss\n')
    for i, word_ in enumerate(words):
        loss_, acc_ = left.model.evaluate([reshape(Xo_[i]), reshape(Xp_[i])], reshape(Yp_[i]))
        f.write('{word:s}, {acc:.8f}, {loss:.8f}\n'.format(
            word = word_,
            acc = acc_,
            loss = loss_))
# %%
