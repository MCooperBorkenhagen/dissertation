#%%
import numpy as np
import pandas as pd
import keras

from utilities import changepad, reshape, loadreps, dists, pronounce2, addpad
from Merge1 import Learner
from tensorflow.keras.utils import plot_model as plot





############
# PATTERNS #
############
#%% load
orthpad = loadreps('../inputs/raw/orthreps.json', changepad=True, newpad=9)['_']
Xo = addpad(np.load('../inputs/orth-left.npy'), orthpad)

Xp = np.load('../inputs/phon-for-eos-left.npy')
Yp = np.load('../inputs/phon-eos-left.npy')


Yp = changepad(Yp, old=9, new=0)
phonreps = loadreps('../inputs/phonreps-with-eos-only.json', changepad=True)
words = pd.read_csv('../inputs/encoder-decoder-words.csv', header=None)[0].tolist()


# dummy patterns:
Xo_dummy = np.zeros(Xo.shape)
Xp_dummy = np.zeros(Xp.shape)

#%%


left = Learner(Xo, Xp, Yp, epochs=25, devices=False)
# %%
pronounce2(1, left.model, Xo, Xp, Yp, labels=words, reps=phonreps)
# %%
