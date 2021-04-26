

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

t = {k:v for k, v in left.items() if k in [4, 5]}

#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# %%
orth_features = left[3]['orth'].shape[2]
phon_features = left[3]['phonSOS'].shape[2]
#%%
learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=t, devices=False)
# bigger batch
learner.fitcycle(batch_size=50, epochs=2, cycles=20)
# smaller batch
learner.fitcycle(batch_size=5, epochs=2, cycles=10)
#%%
# even smaller
learner.fitcycle(batch_size=1, epochs=1, cycles=5)
# %%
#cb = learner.fitcycle(batch_size=50, epochs=1, cycles=30)
#learner.fitcycle(batch_size=5, epochs=1, cycles=20)

#%%
out = []
for word in learner.words[506:520]:
    print(word)
    r = learner.read(word, phonreps=phonreps, ties='sample')
    out.append(r)
    print('word read:', r)
    

# %%
word == 'mad'
for length, traindata_ in t.items():
    for i, w in enumerate(traindata_['wordlist']):
        if w == word:
            out = traindata_['orth'][i], traindata_['phonSOS'][i], traindata_['phonEOS'][i]

#%%
traindata = left


# %%

#def environment(n = 10, ):
