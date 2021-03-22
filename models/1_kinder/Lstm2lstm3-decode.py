"""
Here is an Lstm2lstm that incorporates a second input for phon.

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
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None) 

#%%
############
# PATTERNS #
############
# left justified
X_ = np.load('../../inputs/orth-left.npy')
Y_ = np.load('../../inputs/phon-left.npy')
Y_ = changepad(Y_, old=9, new=0) # take the masked version and turn the mask to zero (ie, add a pad)
# %%


phonreps = loadreps('../../inputs/phonreps.json', changepad=True)
l1 = l(X_, Y_, words, batch_size=350, hidden=800, epochs=25, devices=False)
