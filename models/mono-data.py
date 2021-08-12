
#%%
from utilities import load, split, loadreps, save, scale
import pandas as pd
import numpy as np

tk = load('../inputs/3k/3k.traindata')
words = pd.read_csv('../inputs/3k/words.csv', header=None)

#%%
# phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
# get frequencies for all words:
#%%

#%%


#%%
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
mono_lstm_test, mono_lstm_train = split(tk, .07, drop=True, keys=[4, 5, 6])
#assert mono_lstm_test.keys() == mono_lstm_train.keys(), 'Phonological lengths represented in test and train are not the same. Resample'



#%% get train and test words to match for the feedforward implementation
trainwords = {word: freq for k, v in mono_lstm_train.items() for word, freq in zip(v['wordlist'], v['frequency'])}
testwords = [word for k, v in mono_lstm_test.items() for word in v['wordlist']]
p = .93
maxf = max(trainwords.values())
K = p/np.log(maxf)

# %%
train_indices = []
test_indices = []

trainfile = open('data/mono-train.csv', 'w')
trainfile.write('{},{},{},{}\n'.format('index','word','freq','freq_scaled'))

testfile = open('data/mono-test.csv', 'w')
testfile.write('index,word\n')

for i, row in words.iterrows():
    word = row[0]
    if row[0] in trainwords.keys():    
        freq = trainwords[word]
        freq_scaled = scale(freq, K)
        trainfile.write('{},{},{},{}\n'.format(i, word, freq, freq_scaled))
        train_indices.append(i)
    else:
        testfile.write('{},{}\n'.format(i, word))
        test_indices.append(i)
trainfile.close()
testfile.close()

# now for the feedforward data
X = np.genfromtxt('../inputs/3k/orth.csv', delimiter=',')
Y = np.genfromtxt('../inputs/3k/phon.csv', delimiter=',')

#%%
X_train = X[train_indices]
X_test = X[test_indices]
Y_train = Y[train_indices]
Y_test = Y[test_indices]

#%%
# saves:
save(mono_lstm_test, 'data/mono-lstm-test.traindata')
save(mono_lstm_train, 'data/mono-lstm-train.traindata')


np.savetxt('data/mono-feedforward-train-orth.csv', X_train, delimiter=',')
np.savetxt('data/mono-feedforward-train-phon.csv', Y_train, delimiter=',')
np.savetxt('data/mono-feedforward-test-orth.csv', X_test, delimiter=',')
np.savetxt('data/mono-feedforward-test-phon.csv', Y_test, delimiter=',')

#%%