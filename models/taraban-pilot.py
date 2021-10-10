
#%%
from Learner import Learner
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform

from utilities import *
import time

#from scipy.spatial.distance import pdist, squareform

#import time

test = load('data/taraban_pilot/taraban-test.traindata')
train = load('data/taraban_pilot/taraban-train.traindata')


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

K = pd.read_csv('data/taraban_pilot/taraban-K.txt', header=None)[0].tolist()[0]
#%%
learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='taraban', hidden=400, devices=False)
# %%

# %%
gentests = 72
cycles = 9
for j in range(1, 9): # stop on the 4th cycle
    cycle = j*cycles
    learner.fitcycle(cycles=cycles, epochs=1, probs=probs, K=K, outpath='../outputs/taraban_pilot', evaluate=True, cycle_id=str(cycle))
    learner.model.save('../outputs/taraban_pilot/taraban-model-epoch{}'.format(cycle))
    model_json = learner.model.to_json()
    with open('../outputs/taraban_pilot/model-epoch{}.json'.format(cycle), 'w') as json_f:
        json_f.write(model_json)

    learner.model.save_weights('../outputs/taraban_pilot/model-epoch{}-weights.h5'.format(cycle))
    print(cycle, 'done')
    if cycle == gentests:
        # calculate and write item performance data at end of training for the training items
        # should take about 45 minutes per 1000 words
        colnames = 'word,train_test,freq,phon_read,phonemes_right,phonemes_wrong,phonemes_proportion,phonemes_sum,phonemes_average,phoneme_dists,stress,wordwise_dist\n'

        steps = len([word for data in test.values() for word in data['wordlist']]) + len(learner.trainwords)
        step = 1

        with open('../outputs/taraban_pilot/taraban-generalization-epoch{}.csv'.format(cycle), 'w') as gt:
            gt.write(colnames)

            for length, data in test.items():
                for i, word in enumerate(data['wordlist']):
                    print('on word', step, 'of', steps, 'total words')
                    wd = learner.test(word, return_phonform=True, returns='all', ties='identify')
                    gt.write('{},{},{},{}'.format(word,'test',frequencies[word],flatten(wd)))
                    step += 1

            for word in learner.trainwords:
                print('on word', step, 'of', steps, 'total words')
                wd = learner.test(word, return_phonform=True, returns='all', ties='identify')
                gt.write('{},{},{},{}'.format(word,'train',frequencies[word],flatten(wd)))
                step += 1

        gt.close()

# %%


# %%

trainwords = learner.trainwords
testwords = learner.testwords

dim1 = len(trainwords)
dim2 = len(phonreps['_'])*max(train.keys())
train_outputs = np.zeros((dim1, dim2))
train_targets = np.zeros((dim1, dim2))
dim1 = len(testwords)
test_outputs = np.zeros((dim1, dim2))
test_targets = np.zeros((dim1, dim2))
assert train_outputs.shape[0]+test_outputs.shape[0] == len(learner.words), 'Dims of your arrays are not right, probably'

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

assert l1+l2 == trainwords + testwords == learner.words, 'train and test words do not match the order of words in learner'

#%%
np.savetxt('../outputs/taraban_pilot/test-outputs.csv', test_outputs)
np.savetxt('../outputs/taraban_pilot/train-outputs.csv', train_outputs)

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
np.savetxt('../outputs/taraban_pilot/outputs-targets-distance-matrix-epoch{}.csv'.format(cycle), d_targets_by_outputs)
# %% you can create the same for the less advanced lstm model, go to: mono-lstm-postprocess.py



#%%

from utilities import load, get

# write train and test words to file for pairing with pairwise distance measures later
trainwords = [word for traindata in train.values() for word in traindata['wordlist']]
testwords = [word for testdata in test.values() for word in testdata['wordlist']]

with open('../outputs/taraban_pilot/order.csv', 'w') as f:
    for i, word in enumerate(trainwords+testwords): # it has to be in this order to match the order of the distance matrices
        if word in trainwords:
            condition = 'train'
        elif word in testwords:
            condition = 'test'
        f.write('{},{},{}\n'.format(i, word, condition))
f.close()


# %%
