
"""Acquiring data for the Learner learning to read multisyllabic words that include the experimental lists from Jared et al. (1990) and Chateau & Jared (2003)"""



#%%
from Learner import Learner
import pandas as pd
from utilities import *


test = load('data/js1990/test.traindata')
train = load('data/js1990/train.traindata')

words = get_words(train, verbose=False) + get_words(test, verbose=False)

# phonreps and orthreps
phonreps = loadreps('../inputs/js1990/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# get frequencies for all words across train and test:
frequencies = {word: v['frequency'][i] for v in train.values() for i, word in enumerate(v['wordlist'])}
# add the test frequencies
for k, v in test.items():
    for i, word in enumerate(v['wordlist']):
        frequencies[word] = v['frequency'][i]

orth_features = train[4]['orth'].shape[2]
phon_features = train[4]['phonSOS'].shape[2]
# probabilities for sampling phonological lengths during fitcycle() for training
probs = [.15, .23, .27, .20, .10, .05]
assert len(probs) == len(train.keys()), 'Pick probabilities that match the phonological lengths in train.traindata'

K = pd.read_csv('data/js1990/K.txt', header=None)[0].tolist()[0]

learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='js1990', hidden=400, devices=False)
# %%
gentests = 99
cycles = 9
for j in range(1, 12): # stop on the 11th cycle
    cycle = j*cycles
    learner.fitcycle(cycles=cycles, epochs=1, probs=probs, K=K, outpath='../outputs/js1990', evaluate=True, cycle_id=str(cycle))
    print(cycle, 'done')
    if cycle == gentests:
        learner.model.save('../outputs/js1990/model-epoch{}'.format(cycle))
        model_json = learner.model.to_json()
        with open('../outputs/js1990/model-epoch{}.json'.format(cycle), 'w') as json_f:
            json_f.write(model_json)

        learner.model.save_weights('../outputs/js1990/model-epoch{}-weights.h5'.format(cycle))
        # calculate and write item performance data at end of training for the training items
        # should take about 45 minutes per 1000 words
        colnames = 'word,train_test,freq,phon_read,phonemes_right,phonemes_wrong,phonemes_proportion,phonemes_sum,phonemes_average,phoneme_dists,stress,wordwise_dist\n'

        steps = len([word for data in test.values() for word in data['wordlist']]) + len(learner.trainwords)
        step = 1

        with open('../outputs/js1990/generalization-epoch{}.csv'.format(cycle), 'w') as gt:
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
