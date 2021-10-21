#%%
from utilities import load, get_words, scale
import numpy as np
train = load('data/chateau/train.traindata')

trainfreqs = {word: freq for k, v in train.items() for word, freq in zip(v['wordlist'], v['frequency'])}

p = .93
maxf = max(trainfreqs.values())
K = p/np.log(maxf)
#%%
with open('data/chateau/train.csv', 'w') as f:
    f.write('word,freq,freq_scaled\n')
    for word in get_words(train, verbose=False):
        freq = trainfreqs[word]
        freq_scaled = scale(freq, K)
        f.write('{},{},{}\n'.format(word, freq, freq_scaled))
f.close()