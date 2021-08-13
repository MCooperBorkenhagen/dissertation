#%%
import keras
import numpy as np
import time
import pandas as pd

from utilities import loadreps, load, reshape
from utilities import memory_growth as grow
from scipy.spatial.distance import pdist, cdist, squareform

grow()

test = load('data/mono-lstm-test.traindata')
train = load('data/mono-lstm-train.traindata')
phonreps = loadreps('../inputs/phonreps-with-terminals.json')

model_pre = keras.models.load_model('../outputs/mono/lstm/monosyllabic-model-pre')
#%%
train_test_words = pd.read_csv('../outputs/mono/lstm/train-test-items.csv')


trainwords = train_test_words['word'][train_test_words['train-test'] == 'train'].tolist()
# test
testwords = train_test_words['word'][train_test_words['train-test'] == 'test'].tolist()

# %%
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
        y_hat = model_pre.predict([reshape(data['orth'][i]), reshape(data['phonSOS'][i])]).flatten()
        train_outputs[row][0:y_hat.shape[0]] = y_hat
        train_targets[row][0:y.shape[0]] = y
        l1.append(word)
        print(word, 'done', '...the', str(row)+'th word')
        row += 1
t2 = time.time()
print('all', row, 'words took', (t2-t1)/60, 'minutes')

# %%
row = 0
l2 = []
for data in test.values():
    for i, word in enumerate(data['wordlist']):
        y = data['phonEOS'][i].flatten()
        y_hat = model_pre.predict([reshape(data['orth'][i]), reshape(data['phonSOS'][i])]).flatten()
        test_outputs[row][0:y_hat.shape[0]] = y_hat
        test_targets[row][0:y.shape[0]] = y
        l2.append(word)
        print(word, 'done', '...the', str(row)+'th word')
        row += 1


assert l1+l2 == trainwords + testwords == train_test_words['word'].tolist()

#%%
np.savetxt('../outputs/mono/lstm/posttest-test-outputs-premodel.csv', test_outputs)
np.savetxt('../outputs/mono/lstm/posttest-train-outputs-premodel.csv', train_outputs)

#%%
all_outputs = np.concatenate((train_outputs, test_outputs))
all_targets = np.concatenate((train_targets, test_targets))
d_hat = squareform(pdist(all_outputs))
d_true = squareform(pdist(all_targets))
d_comp = squareform(pdist(np.concatenate((all_outputs, all_targets))))
np.savetxt('../outputs/mono/lstm/posttest-outputs-distance-matrix-premodel.csv', d_hat)
np.savetxt('../outputs/mono/lstm/targets-distance-matrix-premodel.csv', d_true)

d_targets_by_outputs = np.zeros((d_hat.shape))
d_targets_by_outputs[:] = np.nan
#%%

for row in range(d_targets_by_outputs.shape[0]):
    d_targets_by_outputs[row] = d_comp[row][d_hat.shape[0]:d_comp.shape[0]]
    
#%%
np.savetxt('../outputs/mono/lstm/posttest-outputs-targets-distance-matrix-premodel.csv', d_targets_by_outputs)
# %%

model_advanced = keras.models.load_model('../outputs/mono/lstm/monosyllabic-model-advanced')

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
        y_hat = model_advanced.predict([reshape(data['orth'][i]), reshape(data['phonSOS'][i])]).flatten()
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
        y_hat = model_advanced.predict([reshape(data['orth'][i]), reshape(data['phonSOS'][i])]).flatten()
        test_outputs[row][0:y_hat.shape[0]] = y_hat
        test_targets[row][0:y.shape[0]] = y
        l2.append(word)
        print(word, 'done', '...the', str(row)+'th word')
        row += 1
t2 = time.time()
print('all', row, 'words took', (t2-t1)/60, 'minutes')

assert l1+l2 == trainwords + testwords == train_test_words['word'].tolist()

#%%
np.savetxt('../outputs/mono/lstm/posttest-test-outputs-advanced.csv', test_outputs)
np.savetxt('../outputs/mono/lstm/posttest-train-outputs-advanced.csv', train_outputs)

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
np.savetxt('../outputs/mono/lstm/posttest-outputs-distance-matrix-advanced.csv', d_hat)
np.savetxt('../outputs/mono/lstm/targets-distance-matrix-advanced.csv', d_true)

# %%
d_targets_by_outputs = np.zeros((d_hat.shape))
d_targets_by_outputs[:] = np.nan
#%%

for row in range(d_targets_by_outputs.shape[0]):
    d_targets_by_outputs[row] = d_comp[row][d_hat.shape[0]:d_comp.shape[0]]
    
#%%
np.savetxt('../outputs/mono/lstm/posttest-outputs-targets-distance-matrix-advanced.csv', d_targets_by_outputs)



# %%
