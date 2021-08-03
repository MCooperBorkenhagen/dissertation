
#%%
"""
Note that this script was written after the original fitcycles (in monosyllabic.py) 
were run, and if executed again, may run into (solvable) problems. This portion of 
the script was executed with the pretrained model (and word data) read back into the
file so that the outputs could be calculated using native numpy methods, which
are much faster. This is why the 'train-test-items.csv' file is read back in and
the train and test data parsed from that written file rather than using the
train and test traindata dictionary objects constructed above.

"""


import numpy as np
import pandas as pd
import keras
from utilities import load, loadreps, reshape, choose, split, collapse, flatten, shelve, flad, scale
import tensorflow as tf
from utilities import reshape
from utilities import subset

import time

mono = load('../inputs/mono.traindata')

# phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# %%
from memory_growth import memory_growth as grow
grow()


m = keras.models.load_model('monosyllabic-model-advanced')
#%%


train_test_words = pd.read_csv('train-test-items.csv')
trainwords = train_test_words['word'][train_test_words['train-test'] == 'train'].tolist()
# test
testwords = train_test_words['word'][train_test_words['train-test'] == 'test'].tolist()

dim1 = len(trainwords)
dim2 = len(phonreps['_'])*max(mono.keys())
train_outputs = np.zeros((dim1, dim2))
train_targets = np.zeros((dim1, dim2))
dim1 = len(testwords)
test_outputs = np.zeros((dim1, dim2))
test_targets = np.zeros((dim1, dim2))
assert train_outputs.shape[0]+test_outputs.shape[0] == train_test_words.shape[0], 'Dims of your arrays are not right, probably'


trd = subset(mono, trainwords)
#%% takes a few minutes

t1 = time.time()
row = 0
l1 = []
for data in trd.values():
    for i, word in enumerate(data['wordlist']):
        y = data['phonEOS'][i].flatten()
        y_hat = m.predict([reshape(data['orth'][i]), reshape(data['phonSOS'][i])]).flatten()
        train_outputs[row][0:y_hat.shape[0]] = y_hat
        train_targets[row][0:y.shape[0]] = y
        l1.append(word)
        print(word, 'done', '...the', str(row)+'th word')
        row += 1
t2 = time.time()
print('all', row, 'words took', (t2-t1)/60, 'minutes')

# %%
ted = subset(mono, testwords)

t1 = time.time()
row = 0
l2 = []
for data in ted.values():
    for i, word in enumerate(data['wordlist']):
        y = data['phonEOS'][i].flatten()
        y_hat = m.predict([reshape(data['orth'][i]), reshape(data['phonSOS'][i])]).flatten()
        test_outputs[row][0:y_hat.shape[0]] = y_hat
        test_targets[row][0:y.shape[0]] = y
        l2.append(word)
        print(word, 'done', '...the', str(row)+'th word')
        row += 1
t2 = time.time()
print('all', row, 'words took', (t2-t1)/60, 'minutes')

assert l1+l2 == trainwords + testwords == train_test_words['word'].tolist()

#%%
np.savetxt('posttest-test-outputs.csv', test_outputs)
np.savetxt('posttest-train-outputs.csv', train_outputs)

#%%
all_outputs = np.concatenate((train_outputs, test_outputs))
all_targets = np.concatenate((train_targets, test_targets))
# %%
from scipy.spatial.distance import pdist, cdist, squareform

d_hat = squareform(pdist(all_outputs))
d_true = squareform(pdist(all_targets))
#%%
d_comp = squareform(pdist(np.concatenate((all_outputs, all_targets))))
# %%

np.savetxt('posttest-outputs-distance-matrix.csv', d_hat)
np.savetxt('targets_distance-matrix.csv', d_true)

# %%
d_targets_by_outputs = np.zeros((d_hat.shape))
d_targets_by_outputs[:] = np.nan
#%%

for row in range(d_targets_by_outputs.shape[0]):
    d_targets_by_outputs[row] = d_comp[row][d_hat.shape[0]:d_comp.shape[0]]
    
#%%
np.savetxt('posttest-outputs-targets-distance-matrix.csv', d_targets_by_outputs)