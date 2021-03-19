# %%
import tensorflow as tf
import numpy

from Lstm2lstm2 import Learner as l

import numpy as np
import pandas as pd
import tensorflow as tf

import json

from tensorflow.keras.utils import plot_model as plot
from utilities import changepad, key, decode, test_acts, all_equal, cor_acts

from keras import predict
# get words for reference
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None) 

############
# PATTERNS #
############
# left justified
X_ = np.load('../../inputs/orth-left.npy')
Y_ = np.load('../../inputs/phon-left.npy')
Y_ = changepad(Y_, old=9, new=0) # take the masked version and turn the mask to zero (ie, add a pad)
# %%

with open('../../inputs/phonreps.json', 'r') as p:
    phonreps = json.load(p)

#%%

l1 = l(X_, Y_, words, batch_size=200, hidden=300, epochs=200)

# %%
def input(a):
    shape = (1, a.shape[0], a.shape[1])
    return(numpy.reshape(a, shape))

out = l1.model.predict(input(X_[0]))
# %%
def dists(a, reps):
    d = {numpy.linalg.norm(a-numpy.array(v)):k for k, v in reps.items()}
    min_ = min(d.keys())
    return(d[min_])


#%%
def decode(a, reps, round=True):
    if a.ndim == 3:
        a = a[0]
    a = numpy.around(a)
    word = []
    for phoneme in a:
        print(list(phoneme))
        word.append(dists(phoneme, reps))
    return(word)

decode(out, phonreps)
# %%
