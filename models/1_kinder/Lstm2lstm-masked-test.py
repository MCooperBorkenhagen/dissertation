
# %%
from Lstm2lstm import Lstm2lstm
import numpy as np
import pandas as pd
import tensorflow as tf

import time
import json
import pickle
import os

import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model as plot

from scipy.spatial.distance import pdist as dist
from scipy.spatial.distance import squareform as matrix

from utilities import changepad, key, decode, test_acts, all_equal, cor_acts

# get words for reference
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None) 

############
# PATTERNS #
############
# left justified
X_ = np.load('../../inputs/orth-left.npy')
Y_ = np.load('../../inputs/phon-left.npy')
Y_ = changepad(Y_, old=9, new=0) # take the masked version and turn the mask to zero (ie, add a pad)

# right justified
_X = np.load('../../inputs/orth-right.npy')
assert not np.array_equal(X_, _X), 'Your X patterns are the same for right and left and should not be'
#%%



left = Lstm2lstm(X_, Y_, labels=words, epochs=100)


right = Lstm2lstm(_X, Y_, labels=words, epochs=100)
# %%
model = m1.model
# %%
