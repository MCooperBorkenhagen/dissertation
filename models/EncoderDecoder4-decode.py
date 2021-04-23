

#%%
from EncoderDecoder4 import Learner
import numpy as np
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from utilities import load
#%%
from utilities import load, decode, reshape, loadreps, pronounce2, dists, nearest_phoneme

#%%

# load
left = load('../inputs/left.traindata')
right = load('../inputs/right.traindata')
#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# %%
orth_features = left[3]['orth'].shape[2]
phon_features = left[3]['phonSOS'].shape[2]
#%%
learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=left, devices=False)

# %%
#cb = learner.fitcycle(epochs=1)
xo = reshape(left[3]['orth'][0])
xp = reshape(left[3]['phonSOS'][0])
yp = reshape(left[3]['phonEOS'][0])

learner.model.fit([xo, xp], yp, epochs=1)
#%%
learner.read('the', phonreps=phonreps)
# %%

Xo, Xp, Yp = learner.get_word('the')
# %%