


#%%
from Learner import Learner
import numpy as np
import pandas as pd
import keras


from utilities import *

# phonreps and orthreps
phonreps = loadreps('../inputs/taraban/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')


SETS = 50

for I in range(SETS):

    test = load('data/taraban_crossval/test{}.traindata'.format(I))
    train = load('data/taraban_crossval/train{}.traindata'.format(I))
    # get frequencies for all words across train and test:
    frequencies = {word: v['frequency'][i] for k, v in train.items() for i, word in enumerate(v['wordlist'])}
    # add the test frequencies
    for k, v in test.items():
        for i, word in enumerate(v['wordlist']):
            frequencies[word] = v['frequency'][i]

    orth_features = train[4]['orth'].shape[2]
    phon_features = train[4]['phonSOS'].shape[2]
    # probabilities for sampling phonological lengths during fitcycle() for training
    probs = [.3, .4, .2, .075, .025]
    assert len(probs) == len(train.keys()), 'Pick probabilities that match the phonological lengths in train.traindata'

    K = pd.read_csv('data/taraban_crossval/taraban{}-K.txt'.format(I), header=None)[0].tolist()[0]
    
    learner = Learner.Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, testdata=test, loss='binary_crossentropy', modelname='taraban', hidden=400, devices=False)

    cycles = 9
    for j in range(1, 9): # stop on the 4th cycle
        cycle = j*cycles
        learner.fitcycle(cycles=cycles, epochs=1, probs=probs, K=K, outpath='../outputs/taraban_crossval/{}'.format(I), evaluate=True, cycle_id=str(cycle))
        learner.model.save('../outputs/taraban_crossval/{}/model-epoch{}'.format(I, cycle))
        model_json = learner.model.to_json()
        with open('../outputs/taraban_crossval/{}/model-epoch{}.json'.format(I, cycle), 'w') as json_f:
            json_f.write(model_json)

        learner.model.save_weights('../outputs/taraban_crossval/{}/model-epoch{}-weights.h5'.format(I, cycle))
        print(cycle, 'done')

    keras.backend.clear_session()


# %%
