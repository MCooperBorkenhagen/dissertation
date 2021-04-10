"""
This version of the lstm2lstm differs from the other because
it models the output lstm off the sequence of returned values
from the input lstm.

It still doesn't quite make all the right predictions.

"""


# %%
import tensorflow as tf
import numpy

from Lstm2lstm2b import Learner as l
from Lstm2lstm2b import PreLearner as p

import numpy as np
import pandas as pd
import tensorflow as tf

import json
import copy
from tensorflow.keras.utils import plot_model as plot
from utilities import changepad, key, decode, reshape, loadreps






# get words for reference
words = pd.read_csv('../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

############
# PATTERNS #
############
# left justified
#%% load
X_ = np.load('../inputs/orth-left.npy')
Y_ = np.load('../inputs/phon-for-eos-left.npy')
Y_ = changepad(Y_, old=9, new=0)
phonreps = loadreps('../inputs/phonreps-with-eos-only.json', changepad=True)
words = pd.read_csv('../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

#%%

#lp = p(Xp_, labels=words, batch_size=40, hidden=300, epochs=2, devices=False, loss='categorical_crossentropy', train_proportion=.9)

#plot(lp.model, to_file='lstm2lstm2.png')
#%%
#i = 209
#print('word to predict:', words[i])
#out = lp.model.predict(reshape(Xp_[i]))
#print('Predicted:', decode(out, phonreps))
#print('True:', decode(Xp_[i], phonreps))


#%%
l1 = l(X_, Y_, labels=words, batch_size=100, hidden=300, epochs=2, devices=False, loss='categorical_crossentropy', train_proportion=.8)


#%%
# predict one and see:
i = 203
print('word to predict:', words[i])
out = l1.model.predict(reshape(X_[i]))
print('Predicted:', decode(out, phonreps))
print('True:', decode(Y_[i], phonreps))

# %%
