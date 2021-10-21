

"""Data for the multisyllabic learner that also includes the Jared et al (1990) and Chateau & Jared (2003) words in ./js1990.py"""


#%%
from utilities import load, allocate, loadreps, save, scale, get_words
import pandas as pd
import numpy as np

d = load('../inputs/js1990/js.traindata')


# phonreps and orthreps
phonreps = loadreps('../inputs/js1990/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
words = get_words(d, verbose=False)

js_words = pd.read_csv('../inputs/raw/jared_etal_1990_e1.csv').word.tolist()
ca = pd.read_csv('../inputs/raw/chateau_etal_2003_a.csv').word.tolist()
cb = pd.read_csv('../inputs/raw/chateau_etal_2003_b.csv').word.tolist()
cc = pd.read_csv('../inputs/raw/chateau_etal_2003_c.csv').word.tolist()

# we want to make sure these words are in the test set (nonwords):
jsc_words = list(set(js_words + ca + cb + cc))


test, train = allocate(d, .07, for_train=jsc_words, for_test=[], drop=True)

trainfreqs = {word: freq for k, v in train.items() for word, freq in zip(v['wordlist'], v['frequency'])}

p = .93
maxf = max(trainfreqs.values())
K = p/np.log(maxf)

save(test, 'data/js1990/test.traindata')
save(train, 'data/js1990/train.traindata')


with open('data/js1990/train.csv', 'w') as f:
    f.write('word,freq,freq_scaled\n')
    for word in get_words(train, verbose=False):
        freq = trainfreqs[word]
        freq_scaled = scale(freq, K)
        f.write('{},{},{}\n'.format(word, freq, freq_scaled))
f.close()

with open('data/js1990/test.csv', 'w') as f:
    for word in get_words(test, verbose=False):
        f.write('{}\n'.format(word))
f.close()

with open('data/js1990/K.txt', 'w') as f:
    f.write('{}\n'.format(K))
f.close()
# %%
