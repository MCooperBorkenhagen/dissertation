#%%
from Learner import Learner
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from utilities import load, loadreps, reshape, choose, split, collapse, flatten, shelve, flad, scale, subset
from scipy.spatial.distance import pdist, cdist, squareform

import time

test = load('data/mono-lstm-test.traindata')
train = load('data/mono-lstm-train.traindata')


# phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
# get frequencies for all words:
frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
# add the test frequencies
for k, v in test.items():
    for i, word in enumerate(v['wordlist']):
        frequencies[word] = v['frequency'][i]

#%%
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
#assert test.keys() == train.keys(), 'Phonological lengths represented in test and train are not the same. Resample'

orth_features = train[4]['orth'].shape[2]
phon_features = train[4]['phonSOS'].shape[2]
#%% probabilities for sampling phonological lengths during fitcycle() for training
probs = [.3, .4, .2, .075, .025]
assert len(probs) == len(train.keys()), 'Pick probabilities that match the phonological lengths in train.traindata'
#%%
mse = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, modelname='monosyllabic', hidden=400, devices=False, accuracy=mse)
#%% using same function for sampling probabilities from Seidenberg & McClelland (1989)
train_frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(train_frequencies.values())
K = p/np.log(maxf)
# write K to file
with open('../outputs/mono/monosyllabic_k.csv', 'w') as f:
    f.write('{}\n'.format(K))
f.close()
# low train for strong manipulation of frequency:
C = 3
for i in range(1, C+1):
    cycle_id = 'early-'+str(i)
    learner.fitcycle(batch_size=32, cycles=9, probs=probs, K=K, evaluate=True, cycle_id=cycle_id) 

learner.model.save('../outputs/mono/lstm/monosyllabic-model-early')


#%%
C = 2 # how many runs of fitcycle do you want to run. # might be too many to get variability on accuracy, especially with respect to frequency
for i in range(1, C+1):
    cycle_id = 'late-'+str(i)
    learner.fitcycle(batch_size=32, cycles=18, probs=probs, K=K, evaluate=True, cycle_id=cycle_id) 

"""Done with late training"""
#%%
learner.model.save('../outputs/mono/lstm/monosyllabic-model-late')
# %% save the train and test words for the future. We will also save its scaled frquency along with it
#%%
with open('../outputs/mono/lstm/train-test-items.csv', 'w') as tt:
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

# assessments for the holdout items
colnames = 'word,freq,phon_read,phonemes_right,phonemes_wrong,phonemes_proportion,phonemes_sum,phonemes_average,phoneme_dists,stress,wordwise_dist\n'

steps = len([word for data in test.values() for word in data['wordlist']])
step = 1

with open('../outputs/mono/lstm/posttest-holdout-words.csv', 'w') as ht:
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
steps = len(learner.words)
step = 1
#%%
with open('../outputs/mono/lstm/posttest-trainwords.csv', 'w') as at:
    at.write(colnames)
    for word in learner.words:
        print('on word', step, 'of', steps, 'total words')
        wd = learner.test(word, return_phonform=True, returns='all', ties='identify')
        at.write(word+','+str(frequencies[word])+','+flatten(wd))
        step += 1
at.close()

# calculate true phonological outputs for all training and test items: see monosyllabic-outputs.py
# %%
train_test_words = pd.read_csv('../outputs/mono/lstm/train-test-items.csv')
trainwords = train_test_words['word'][train_test_words['train-test'] == 'train'].tolist()
# test
testwords = train_test_words['word'][train_test_words['train-test'] == 'test'].tolist()
#%%
dim1 = len(trainwords)
dim2 = len(phonreps['_'])*max(train.keys())
train_outputs = np.zeros((dim1, dim2))
train_targets = np.zeros((dim1, dim2))
dim1 = len(testwords)
test_outputs = np.zeros((dim1, dim2))
test_targets = np.zeros((dim1, dim2))
assert train_outputs.shape[0]+test_outputs.shape[0] == train_test_words.shape[0], 'Dims of your arrays are not right, probably'

#%% takes a few minutes

t1 = time.time()
row = 0
l1 = []
for data in train.values():
    for i, word in enumerate(data['wordlist']):
        y = data['phonEOS'][i].flatten()
        y_hat = learner.model.predict([reshape(data['orth'][i]), reshape(data['phonSOS'][i])]).flatten()
        train_outputs[row][0:y_hat.shape[0]] = y_hat
        train_targets[row][0:y.shape[0]] = y
        l1.append(word)
        print(word, 'done', '...the', str(row)+'th word')
        row += 1
t2 = time.time()
print('all', row, 'words took', (t2-t1)/60, 'minutes')

# %%

t1 = time.time()
row = 0
l2 = []
for data in test.values():
    for i, word in enumerate(data['wordlist']):
        y = data['phonEOS'][i].flatten()
        y_hat = learner.model.predict([reshape(data['orth'][i]), reshape(data['phonSOS'][i])]).flatten()
        test_outputs[row][0:y_hat.shape[0]] = y_hat
        test_targets[row][0:y.shape[0]] = y
        l2.append(word)
        print(word, 'done', '...the', str(row)+'th word')
        row += 1
t2 = time.time()
print('all', row, 'words took', (t2-t1)/60, 'minutes')

assert l1+l2 == trainwords + testwords == train_test_words['word'].tolist()

#%%
np.savetxt('../outputs/mono/lstm/posttest-test-outputs.csv', test_outputs)
np.savetxt('../outputs/mono/lstm/posttest-train-outputs.csv', train_outputs)

#%%
all_outputs = np.concatenate((train_outputs, test_outputs))
all_targets = np.concatenate((train_targets, test_targets))
# %%

d_hat = squareform(pdist(all_outputs))
d_true = squareform(pdist(all_targets))
#%%
d_comp = squareform(pdist(np.concatenate((all_outputs, all_targets))))
# %%
# if you want to save:
#np.savetxt('posttest-outputs-distance-matrix.csv', d_hat)
#np.savetxt('targets-distance-matrix.csv', d_true)

# %%
d_targets_by_outputs = np.zeros((d_hat.shape))
d_targets_by_outputs[:] = np.nan
#%%

for row in range(d_targets_by_outputs.shape[0]):
    d_targets_by_outputs[row] = d_comp[row][d_hat.shape[0]:d_comp.shape[0]]
    
#%%
np.savetxt('../outputs/mono/lstm/posttest-outputs-targets-distance-matrix-late.csv', d_targets_by_outputs)
# %% you can create the same for the less advanced lstm model, go to: mono-lstm-postprocess.py


