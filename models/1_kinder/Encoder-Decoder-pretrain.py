"""
Here we attempt to pretrain on the encoder-decoder architecture
with passing phonological patterns just to the decoder to start.

"""


# %%
from EncoderDecoder import Learner
import numpy as np
from utilities import changepad, key, decode, reshape, loadreps, test_acts, all_equal, cor_acts
import pandas as pd
from tensorflow.keras.utils import plot_model as plot

import keras
from tensorflow.keras import layers
from scipy.spatial.distance import pdist as dist
from scipy.spatial.distance import squareform as matrix
#%%

# get words for reference
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

############
# PATTERNS #
############
#%% load
Xo_ = np.load('../../inputs/orth-left.npy')
Xo_dummy = np.zeros(Xo_.shape)
Xp_ = np.load('../../inputs/phon-left.npy')
Yp_ = np.load('../../inputs/phon-left.npy')
Xp_dummy = np.zeros(Xp_.shape)
Yp_ = changepad(Yp_, old=9, new=0)
phonreps = loadreps('../../inputs/phonreps.json', changepad=True)
orthreps = loadreps('../../inputs/raw/orthreps.json')

# %% pretrain
leftT = Learner(Xo_dummy, Xp_, Yp_, epochs=10, devices=False, monitor=False)


# %% test one
phonshape = (1, Xp_.shape[1], Xp_.shape[2])
orthshape = (1, Xo_.shape[1], Xo_.shape[2])
dummyP = np.zeros(phonshape)
dummyO = np.zeros(orthshape)


i = 3296
testword = words[i]
xo = reshape(Xo_[i])
xp = reshape(Xp_[i])
out = leftT.model.predict([dummyO, xp])

print(testword)
print('Predicted: ', decode(out, phonreps))

print('Actual: ', decode(Yp_[i], phonreps))
# %% learn
leftT = Learner(Xo_, Xp_, Yp_, epochs=20, devices=False, monitor=False)




#%% decode

#%%


#%%

# %%


