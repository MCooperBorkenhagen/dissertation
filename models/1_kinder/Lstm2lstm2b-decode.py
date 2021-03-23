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
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

############
# PATTERNS #
############
# left justified
X_ = np.load('../../inputs/orth-left.npy')
Y_ = np.load('../../inputs/phon-left.npy')
Xp_ = np.load('../../inputs/phon-left.npy')
Y_ = changepad(Y_, old=9, new=0) # take the masked version and turn the mask to zero (ie, add a pad)

# %%
simpler_words = [i for i, word in enumerate(words) if len(word) < 5]

#%%
X_ = X_[simpler_words]
Y_ = Y_[simpler_words]
Xp_ = Xp_[simpler_words]
words = [word for i, word in enumerate(words) if i in simpler_words]

#%%

phonreps = loadreps('../../inputs/phonreps.json', changepad=True)
lp = p(Xp_, labels=words, batch_size=40, hidden=300, epochs=20, devices=False, loss='categorical_crossentropy', train_proportion=.9)
#%%
i = 209
print('word to predict:', words[i])
out = lp.model.predict(reshape(Xp_[i]))
print('Predicted:', decode(out, phonreps))
print('True:', decode(Xp_[i], phonreps))


#%%
prek = copy.copy(lp.phon_lstm)
for layer in lp.model.layers:
    if 'LSTM' in str(layer):
        weights = layer.get_weights()



#%%
l1 = l(X_, Y_, labels=words, batch_size=100, hidden=300, epochs=2, devices=False, loss='categorical_crossentropy', train_proportion=.8)
# predict one and see:
i = 203
print('word to predict:', words[i])
out = l1.model.predict(reshape(X_[i]))
print('Predicted:', decode(out, phonreps))
print('True:', decode(Y_[i], phonreps))

# %%
