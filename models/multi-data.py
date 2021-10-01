"""Putting together data for the multisyllabic learner in ./multi.py"""


#%%
from utilities import load, allocate, loadreps, save, scale
import pandas as pd
import numpy as np

d = load('../inputs/left.traindata')
words = [word for v in d.values() for word in v['wordlist']]
print(len(words), 'words present in full dataset')
# %%
test, train = allocate(d, .07, for_test=[], for_train=[], drop=True)
# %%

trainfreqs = {word: freq for v in train.values() for word, freq in zip(v['wordlist'], v['frequency'])}
testwords = [word for v in test.values() for word in v['wordlist']]

p = .93
maxf = max(trainfreqs.values())
K = p/np.log(maxf)
# %%


trainfile = open('data/multi-train.csv', 'w')
trainfile.write('{},{},{}\n'.format('word','freq','freq_scaled'))
testfile = open('data/multi-test.csv', 'w')

for i, word in enumerate(words):
    if word in trainfreqs.keys():    
        freq = trainfreqs[word]
        freq_scaled = scale(freq, K)
        trainfile.write('{},{},{}\n'.format(word, freq, freq_scaled))
    else:
        testfile.write('{}\n'.format(word))
trainfile.close()
testfile.close()


with open('data/multi-K.txt', 'w') as f:
    f.write('{}\n'.format(K))
f.close()


# save traindata and testdata
# saves:
save(test, 'data/multi-test.traindata')
save(train, 'data/multi-train.traindata')


print('Done.')
# %%
