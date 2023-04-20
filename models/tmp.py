


# lets just use the simple (mono) corpus for this, to accelerate training time

#%%
import LearnerRS
import numpy as np
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
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='tmp', hidden=400, devices=False)
train_frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(train_frequencies.values())
K = p/np.log(maxf)


#%%

# 0 condition: null phon on epoch 0
C = 1
CYCLES = 1
for i in range(1, C+1):
    cycle = i*CYCLES
    cycle_id = str(cycle)    
    learner.fitcycle(batch_size=32, null_phon=False, cycles=CYCLES, probs=probs, K=K, evaluate=False, cycle_id=cycle_id, outpath='../outputs/tmp/') 

