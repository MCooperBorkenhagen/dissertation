
# %%
from EncoderDecoder import Learner
import numpy as np
#%%
from utilities import changepad, key, decode, reshape, loadreps, test_acts, all_equal, cor_acts, pronounce2, dists
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


#%%
Xp_ = np.load('../inputs/phon-inputs-for-eos-left.npy')
Yp_ = np.load('../inputs/phon-with-eos-left.npy')

Yp_ = changepad(Yp_, old=9, new=0)
phonreps = loadreps('../inputs/phonreps-with-eos-only.json', changepad=True)

orthreps = loadreps('../inputs/raw/orthreps.json')
words = pd.read_csv('../inputs/encoder-decoder-words.csv', header=None)[0].tolist()
#%%
pad = loadreps('../inputs/phonreps-with-eos-only.json', changepad=True)['_']

#%%
# dummy patterns:
Xo_dummy = np.zeros(Xo_.shape)
Xp_dummy = np.zeros(Xp_.shape)

# %% step one: PP training
hidden = 500
left = Learner(Xo_dummy, Xp_, Yp_, epochs=1, batch_size=400, hidden=hidden, devices=False)

plot(left.model, to_file='encoder-decoder1.png')

# %% Op together, then O alone
left.model.fit([Xo_, Xp_], Yp_, epochs=1, batch_size=1)

#%%
left.model.fit([Xo_, Xp_dummy], Yp_, epochs=1, batch_size=1)


#%%
#%%
for i, word in enumerate(words):
    if word == 'ratio':
        print(word, i)

def build_input(word, reps, maxlen, padvalue=9):
    a = [reps[e] for e in word]
    pad = [padvalue for value in reps['_']]
    for e in range(maxlen-len(word)):
        a.append(pad)
    return np.array(a)


def generalize(xo, xp, model, reps, label=None):
    print('word to predict:', label)
    out = model.predict([reshape(xo), reshape(xp)])
    print('Predicted:', decode(out, reps))
    
diction = build_input('diction', orthreps, max([len(word) for word in words]))
hawking = build_input('hawking', orthreps, max([len(word) for word in words]))
#%%
generalize(diction, Xp_dummy[0], left.model, phonreps, label='diction')

#%%
generalize(hawking, Xp_dummy[0], left.model, phonreps, label='hawking')


#%%
def get_index(word, words):
    return([i for i, w in enumerate(words) if w == word][0])

get_index('mit', words)

#%%
i = get_index('crabtree', words)
print('with phon inputs only:')
pronounce2(i, left.model, Xo_dummy, Xp_, Yp_, labels=words, reps=phonreps)
print('#####################')
print('with orth and phon inputs:')
pronounce2(i, left.model, Xo_, Xp_, Yp_, labels=words, reps=phonreps)
#%%


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


with open('encoder-decoder-items.csv', 'w') as f:
    f.write('word, acc, loss\n')
    for i, word_ in enumerate(words):
        loss_, acc_ = left.model.evaluate([reshape(Xo_[i]), reshape(Xp_[i])], reshape(Yp_[i]))
        f.write('{word:s}, {acc:.8f}, {loss:.8f}\n'.format(
            word = word_,
            acc = acc_,
            loss = loss_))