
#%%
from Learner import Learner
import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from utilities import *

from scipy.spatial.distance import pdist, squareform

import time

test = load('data/taraban-test.traindata')
train = load('data/taraban-train.traindata')


# phonreps and orthreps
phonreps = loadreps('../inputs/taraban/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# get frequencies for all words across train and test:
frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
# add the test frequencies
for k, v in test.items():
    for i, word in enumerate(v['wordlist']):
        frequencies[word] = v['frequency'][i]

#%%

orth_features = train[4]['orth'].shape[2]
phon_features = train[4]['phonSOS'].shape[2]
# probabilities for sampling phonological lengths during fitcycle() for training
probs = [.3, .4, .2, .075, .025]
assert len(probs) == len(train.keys()), 'Pick probabilities that match the phonological lengths in train.traindata'

K = pd.read_csv('data/taraban-K.txt', header=None)[0].tolist()[0]
#%%
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, modelname='taraban', hidden=400, devices=False)

# %%

learner.fitcycle(probs=probs, K=K, outpath='../outputs/taraban')
# %%
