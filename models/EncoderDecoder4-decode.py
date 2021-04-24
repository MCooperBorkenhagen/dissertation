

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
learner.fitcycle(batch_size=50, epochs=2, cycles=12)
# %%
#cb = learner.fitcycle(batch_size=50, epochs=1, cycles=30)
#learner.fitcycle(batch_size=5, epochs=1, cycles=20)

#%%
out = []
for word in learner.words[526:529]:
    print(word)
    r = learner.read(word, phonreps=phonreps, ties='sample')
    out.append(r)
    print('word read:', r)
    

# %%
traindata = left


# %%

def environment(n = 10, ):
