

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
learner = Learner(orth_features, phon_features, traindata=left, devices=False)

# %%
cb = learner.fitcycle(batch_size=1)
# %%
