"""Here we run training procedures that incorporate null inputs 
into the process in order to determine if learning is possible just 
from orthography, and the relative importance to phonology"""


# lets just use the simple (mono) corpus for this, to accelerate training time

#%%
from Learner import Learner
import numpy as np
import keras
import tensorflow as tf
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
train_frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(train_frequencies.values())
K = p/np.log(maxf)


#%%
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, freeze_phon=True, loss='binary_crossentropy', modelname='', hidden=200, devices=False)



C = 3
CYCLES = 5
for i in range(1, C+1): 
    cycle_id = 'precompile-test-freezeallphon'+str(i)
    learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='tmp') 


#%%
keras.backend.clear_session()


#%%
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, freeze_phon=False, loss='binary_crossentropy', modelname='', hidden=200, devices=False)


with open('tmp/postcompile-test-freezeallphon.csv', 'w') as f:
    C = 3
    CYCLES = 5
    for i in range(1, C+1):
        cycle_id = 'postcompile-test-freezeallphon'+str(i)    
        
        if i == 2:
            learner.model.layers[4].trainable = False
            learner.model.layers[5].trainable = False

            optimizer='rmsprop'
            loss='binary_crossentropy'
            metrics = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]
            learner.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        learner.fitcycle(batch_size=32, cycles=CYCLES, probs=probs, K=K, evaluate=True, cycle_id=cycle_id, outpath='tmp', verbose=0)
        f.write('{},{},{}\n'.format(i, str(learner.decoder_lstm.trainable), str(learner.decoder_dense.trainable)))
f.close()
# %%
