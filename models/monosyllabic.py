#%%
from Learner import Learner
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from utilities import load, loadreps, reshape, choose, split, collapse, flatten, shelve, flad, scale, subset

import time

mono = load('../inputs/mono.traindata')

# phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
# get frequencies for all words:
frequencies = {word: v['frequency'][i] for k, v in mono.items() for i, word in enumerate(v['wordlist'])}


#%%
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
test, train = split(mono, .07)
assert test.keys() == train.keys(), 'Phonological lengths represented in test and train are not the same. Resample'

orth_features = train[4]['orth'].shape[2]
phon_features = train[4]['phonSOS'].shape[2]
#%% probabilities for sampling phonological lengths during fitcycle() for training
probs = [.3, .4, .2, .075, .025]

#%%
learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, modelname='monosyllabic', hidden=400, devices=False)
#%%


#%% using same function for sampling probabilities from Seidenberg & McClelland (1989)
train_frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(train_frequencies.values())
K = p/np.log(maxf)

# low train for strong manipulation of frequency:
C = 3
for i in range(1, C+1):
    cycle_id = 'pre-'+str(i)
    learner.fitcycle(batch_size=70, cycles=3, probs=probs, K=K, evaluate=True, cycle_id=cycle_id) 
#%%
learner.model.save('monosyllabic-model-pre')

#%%
C = 2 # how many runs of fitcycle do you want to run. # might be too many to get variability on accuracy, especially with respect to frequency
for i in range(1, C+1):
    cycle_id = 'advanced-'+str(i)
    learner.fitcycle(batch_size=70, cycles=18, probs=probs, K=K, evaluate=True, cycle_id=cycle_id) 

print('Done with advanced training')
#%%
learner.model.save('monosyllabic-model-advanced')
# %% save the train and test words for the future. We will also save its scaled frquency along with it
#%%
with open('train-test-items.csv', 'w') as tt:
    tt.write('word,freq_scaled,train-test\n')
    for word in learner.words:
        sf = str(scale(frequencies[word], K))
        tt.write(word+','+sf+','+'train'+'\n')
    
    for v in test.values():
        for word in v['wordlist']:
            tt.write(word+','+''+','+'test'+'\n')

tt.close()
#%%
# if you want to load the previously saved model:
#m = keras.models.load_model('...')

#%%
# assessments for the holdout items:
colnames = 'word,freq,phon_read,phonemes_right,phonemes_wrong,phonemes_proportion,phonemes_sum,phonemes_average,phoneme_dists,stress,wordwise_dist\n'

steps = len([word for data in test.values() for word in data['wordlist']])
step = 1

with open('posttest-holdout-words.csv', 'w') as ht:
    ht.write(colnames)

    for length, data in test.items():
        for i, word in enumerate(data['wordlist']):
            print('on word', step, 'of', steps, 'total words')
            wd = learner.test(word, target=data['phonEOS'][i], return_phonform=True, returns='all', ties='identify')
            ht.write(word+','+str(frequencies[word])+','+flatten(wd))
            step += 1
ht.close()


# %% calculate and write item performance data at end of training for the training items
# should take about 45 minutes per 1000 words
steps = len([word for data in train.values() for word in data['wordlist']])
step = 1

with open('posttest-trainwords.csv', 'w') as at:
    at.write(colnames)
    for word in learner.words:
        print('on word', step, 'of', steps, 'total words')
        wd = learner.test(word, return_phonform=True, returns='all', ties='identify')
        at.write(word+','+str(frequencies[word])+','+flatten(wd))
        step += 1
at.close()


#
# calculate true phonological outputs for all training and test items: see monosyllabic-supplemet.py
#
