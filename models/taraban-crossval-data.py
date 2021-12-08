#%%
from utilities import load, allocate, loadreps, save, scale
import pandas as pd
import numpy as np

tk = load('../inputs/taraban/taraban.traindata')

#%%

# phonreps and orthreps
phonreps = loadreps('../inputs/taraban/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
words = pd.read_csv('../inputs/taraban/words.csv', header=None)[0].tolist()

taraban = pd.read_csv('../inputs/raw/taraban_etal_1987_words.csv')#.word.tolist()
jaredA = pd.read_csv('../inputs/raw/jared_1997_appendixA.csv')#.word.tolist()
jaredC = pd.read_csv('../inputs/raw/jared_1997_appendixC.csv')#.word.tolist()

# we want to make sure these words are in the test set (nonwords):
nonwords = taraban.word[taraban.condition == 'nonword'].tolist()

#%%


# these are the words we want make sure are in the train set
# first, all the taraban and jared words
tjwords = [word for word in list(set(taraban.word.tolist()+jaredA.word.tolist()+jaredC.word.tolist())) if word in words]
# and filter out the nonwords and call the list "for_train"
for_train = [word for word in tjwords if word not in nonwords] 


#%%



SETS = 50
#%%
for I in range(SETS):
    test, train = allocate(tk, .07, for_train=for_train, for_test=nonwords, drop=True, keys=[4, 5, 6])

    #%%
    trainfreqs = {word: freq for k, v in train.items() for word, freq in zip(v['wordlist'], v['frequency'])}
    testwords = [word for k, v in test.items() for word in v['wordlist']]

    p = .93
    maxf = max(trainfreqs.values())
    K = p/np.log(maxf)


    trainfile = open('data/taraban_crossval/train{}.csv'.format(I), 'w')
    trainfile.write('{},{},{}\n'.format('word','freq','freq_scaled'))

    testfile = open('data/taraban_crossval/test{}.csv'.format(I), 'w')

    for i, word in enumerate(words):
        if word in trainfreqs.keys():    
            freq = trainfreqs[word]
            freq_scaled = scale(freq, K)
            trainfile.write('{},{},{}\n'.format(word, freq, freq_scaled))
        else:
            testfile.write('{}\n'.format(word))
    trainfile.close()
    testfile.close()


    with open('data/taraban_crossval/taraban{}-K.txt'.format(I), 'w') as f:
        f.write('{}\n'.format(K))
    f.close()


    # save traindata and testdata
    # saves:
    save(test, 'data/taraban_crossval/test{}.traindata'.format(I))
    save(train, 'data/taraban_crossval/train{}.traindata'.format(I))


    print('Done.')
#%%