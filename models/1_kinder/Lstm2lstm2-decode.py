"""
This version of the lstm2lstm differs from the other because
it models the output lstm off the sequence of returned values
from the input lstm.

It still doesn't quite make all the right predictions.

"""


# %%
import tensorflow as tf
import numpy

from Lstm2lstm2 import Learner as l

import numpy as np
import pandas as pd
import tensorflow as tf

import json

from tensorflow.keras.utils import plot_model as plot
from utilities import changepad, key, decode, reshape, loadreps






# get words for reference
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

############
# PATTERNS #
############
# left justified
X_ = np.load('../../inputs/orth-left.npy')
Y_ = np.load('../../inputs/phon-eos-left.npy')
Y_ = changepad(Y_, old=9, new=0) # take the masked version and turn the mask to zero (ie, add a pad)
# %%
pre = [i for i, word in enumerate(words) if len(word) < 5]

#%%


phonreps = loadreps('../../inputs/phonreps.json', changepad=True)
l1 = l(X_, Y_, words, batch_size=100, hidden=400, epochs=20, devices=False, loss='categorical_crossentropy', train_proportion=.8)

# %%
print('word to inspect:', words[206])
out = l1.model.predict(reshape(X_[206]))
decode(out, phonreps)

#%%
decode(Y_[206], phonreps)
# %%
