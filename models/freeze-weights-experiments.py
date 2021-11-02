"""Here we run training procedures that incorporate null inputs 
into the process in order to determine if learning is possible just 
from orthography, and the relative importance to phonology"""


# lets just use the simple (mono) corpus for this, to accelerate training time

#%%
from Learner import Learner
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from utilities import load, loadreps

test = load('data/mono-lstm-test.traindata')
train = load('data/mono-lstm-train.traindata')


# phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
# get frequencies for all words:
frequencies = {word: v['frequency'][i] for v in train.values() for i, word in enumerate(v['wordlist'])}
# add the test frequencies
for k, v in test.items():
    for i, word in enumerate(v['wordlist']):
        frequencies[word] = v['frequency'][i]

# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
#assert test.keys() == train.keys(), 'Phonological lengths represented in test and train are not the same. Resample'

orth_features = train[4]['orth'].shape[2]
phon_features = train[4]['phonSOS'].shape[2]
# probabilities for sampling phonological lengths during fitcycle() for training
probs = [.3, .4, .2, .075, .025]
assert len(probs) == len(train.keys()), 'Pick probabilities that match the phonological lengths in train.traindata'
#
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='', hidden=400, devices=False)
train_frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(train_frequencies.values())
K = p/np.log(maxf)
#%%




# first condition: freeze weights on epoch 18
C = 7
CYCLES = 9
freezetime = 18
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    if cycle == freezetime:
        learner.model.layers[4].trainable = False
    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon/a') 
    
keras.backend.clear_session()

# second condition: freeze weights on epoch 36
# reimplement learner
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='', hidden=400, devices=False)

freezetime = 36
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    if cycle == freezetime:
        learner.model.layers[4].trainable = False
    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon/b') 

keras.backend.clear_session()

# third condition: freeze weights on epoch 54
# reimplement learner
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='', hidden=400, devices=False)

freezetime = 54
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    if cycle == freezetime:
        learner.model.layers[4].trainable = False
    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon/c') 
keras.backend.clear_session()


# fourth condition: never freeze weights
# reimplement learner
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='', hidden=400, devices=False)
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon/d') 
    