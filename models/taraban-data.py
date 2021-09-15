
from utilities import load, split_but_skip, loadreps, save, scale
import pandas as pd
import numpy as np

tk = load('../inputs/taraban/taraban.traindata')
#words = pd.read_csv('../inputs/taraban/words.csv', header=None)
# phonreps and orthreps
phonreps = loadreps('../inputs/taraban/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
words = pd.read_csv('../inputs/taraban/words.csv', header=None)[0].tolist()

taraban = pd.read_csv('../inputs/raw/taraban_etal_1987_words.csv').word.tolist()
jaredA = pd.read_csv('../inputs/raw/jared_1997_appendixA.csv').word.tolist()
jaredC = pd.read_csv('../inputs/raw/jared_1997_appendixC.csv').word.tolist()

skip = [word for word in list(set(taraban+jaredA+jaredC)) if word in words]


test, train = split_but_skip(tk, .07, skip=skip, drop=True, keys=[4, 5, 6])

trainfreqs = {word: freq for k, v in train.items() for word, freq in zip(v['wordlist'], v['frequency'])}
testwords = [word for k, v in test.items() for word in v['wordlist']]

p = .93
maxf = max(trainfreqs.values())
K = p/np.log(maxf)






trainfile = open('data/taraban-train.csv', 'w')
trainfile.write('{},{},{}\n'.format('word','freq','freq_scaled'))

testfile = open('data/taraban-test.csv', 'w')

for i, word in enumerate(words):
    if word in trainfreqs.keys():    
        freq = trainfreqs[word]
        freq_scaled = scale(freq, K)
        trainfile.write('{},{},{}\n'.format(word, freq, freq_scaled))
    else:
        testfile.write('{}\n'.format(word))
trainfile.close()
testfile.close()


with open('data/taraban-K.txt', 'w') as f:
    f.write('{}\n'.format(K))
f.close()


# save traindata and testdata
# saves:
save(test, 'data/taraban-test.traindata')
save(train, 'data/taraban-train.traindata')

