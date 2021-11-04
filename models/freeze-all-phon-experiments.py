"""Here we run training procedures that incorporate frozen weights 
into the process in order to determine if learning is possible in
such conditions. These trials freeze weights on the phonological
LSTM and the dense output layer"""


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

orth_features = train[4]['orth'].shape[2]
phon_features = train[4]['phonSOS'].shape[2]
# probabilities for sampling phonological lengths during fitcycle() for training
probs = [.3, .4, .2, .075, .025]
assert len(probs) == len(train.keys()), 'Pick probabilities that match the phonological lengths in train.traindata'
#
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, modelname='freezephon-all', loss='binary_crossentropy', hidden=400, freeze_phon=True, devices=False)
train_frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(train_frequencies.values())
K = p/np.log(maxf)


#%%
C = 7
CYCLES = 9
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    learner.model.layers[4].trainable = False
    learner.model.layers[5].trainable = False

    optimizer='rmsprop'
    loss='binary_crossentropy'
    metrics = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]
    learner.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon-all/0', verbose=0) 
keras.backend.clear_session()


# 18 condition: freeze weights on epoch 18
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='freezephon-all', hidden=400, devices=False)

C = 7
CYCLES = 9
freezetime = 18
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    if cycle == freezetime:
        learner.model.layers[4].trainable = False
        learner.model.layers[5].trainable = False
        optimizer='rmsprop'
        loss='binary_crossentropy'
        metrics = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]
        learner.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon-all/18', verbose=0) 
    
keras.backend.clear_session()

# 36 condition: freeze weights on epoch 36
# reimplement learner
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='freezephon-all', hidden=400, devices=False)

freezetime = 36
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    if cycle == freezetime:
        learner.model.layers[4].trainable = False
        learner.model.layers[5].trainable = False
        optimizer='rmsprop'
        loss='binary_crossentropy'
        metrics = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]
        learner.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon-all/36', verbose=0) 

keras.backend.clear_session()

# 54 condition: freeze weights on epoch 54
# reimplement learner
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='freezephon-all', hidden=400, devices=False)

freezetime = 54
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    if cycle == freezetime:
        learner.model.layers[4].trainable = False
        learner.model.layers[5].trainable = False
        optimizer='rmsprop'
        loss='binary_crossentropy'
        metrics = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]
        learner.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon-all/54', verbose=0) 

keras.backend.clear_session()


# never condition: never freeze weights
# reimplement learner
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='freezephon-all', hidden=400, devices=False)
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='../outputs/freezephon-all/never', verbose=0) 
