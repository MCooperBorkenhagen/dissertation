

"""Data for the multisyllabic learner that also includes the Cahteau & Jared (2003) words in ./chateau.py"""


#%%
from utilities import load, allocate, loadreps, save, scale, get_words
import pandas as pd
import numpy as np

d = load('../inputs/chateau/chateau.traindata')


# phonreps and orthreps
phonreps = loadreps('../inputs/chateau/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
words = get_words(d)

ca = pd.read_csv('../inputs/raw/chateau_etal_2003_a.csv').word.tolist()
cb = pd.read_csv('../inputs/raw/chateau_etal_2003_b.csv').word.tolist()
cc = pd.read_csv('../inputs/raw/chateau_etal_2003_c.csv').word.tolist()


#%%
# we want to make sure these words are in the test set (nonwords):
chateau_words = list(set(ca + cb + cc))


test, train = allocate(d, .07, for_train=chateau_words, for_test=[], drop=True)
# %%
trainfreqs = {word: freq for k, v in train.items() for word, freq in zip(v['wordlist'], v['frequency'])}

p = .93
maxf = max(trainfreqs.values())
K = p/np.log(maxf)
# %%
save(test, 'data/chateau/test.traindata')
save(train, 'data/chateau/train.traindata')

#%%
with open('data/chateau/train.csv', 'w') as f:
    f.write('word,freq,freq_scaled\n')
    for word in get_words(train, verbose=False):
        freq = trainfreqs[word]
        freq_scaled = scale(freq, K)
        f.write('{},{},{}\n'.format(word, freq, freq_scaled))
f.close()
# %%
with open('data/chateau/test.csv', 'w') as f:
    for word in get_words(test, verbose=False):
        f.write('{}\n'.format(word))
f.close()
# %%
with open('data/chateau/K.txt', 'w') as f:
    f.write('{}\n'.format(K))
f.close()
# %%
