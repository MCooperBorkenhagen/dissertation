

#%%
from Learner import Learner
import numpy as np
import pandas as pd
import keras

#%%
from utilities import load, loadreps, reshape, choose

#%%


#%%
# load
data = load('../inputs/left.traindata')
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)


#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# %% get monosyllabic words

syllabics = pd.read_csv('../inputs/syllabics.csv')
#%%
#%%


mono = load('../inputs/mono.traindata')


#%%
orth_features = data[4]['orth'].shape[2]
phon_features = data[4]['phonSOS'].shape[2]
#%% probabilities for sampling phonological lengths during fitcycle() for data
ps = [.2, .3, .2, .2, .05, .05] 
#%% and for mono
mps = [.3, .4, .2, .075, .025]
#%%




learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=mono, hidden=400, mask_phon=False, devices=False)


#%% using same function for sampling probabilities from Seidenberg & McClelland (1989)
frequencies = {word: v['frequency'][i] for k, v in mono.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(frequencies.values())
K = p/np.log(maxf)



#%%
learner.fitcycle(batch_size=70, cycles=20, probs=mps, K=K, evaluate=False) 


#%%
#learner.model.save('base-model')
#m = keras.models.load_model('./base-model')

# %%
learner.read('quilt', ties='sample')
# %%
__, __, yp = learner.find('think')

# %%
tmp = learner.test('think', return_phonform=True, returns='all', ties = 'sample', phonreps=None)
# %%

#%%

def split(traindata, n, seed = 652):
    from random import sample, seed
    
    seed(seed)

    s = [word for k, v in data.items() for word in v['wordlist']]

    if type(n) == float:
        n = round(n*len(s))

    r = sample(s, n)

    holdout = {}
    train = {}
    for k, v in data.items():
        subset = []
        primary = []
        wordlist = []
        trainwords = []
        for i, word in enumerate(v['wordlist']):
            if word in r:
                subset.append(i)
                wordlist.append(word)
            else:
                primary.append(i)
                trainwords.append(word)
        train[k] = {'phonSOS':v['phonSOS'][primary], 'phonEOS':v['phonEOS'][primary], 'orth':v['orth'][primary], 'wordlist':trainwords, 'frequency':v['frequency'][primary]}
        holdout[k] = {'phonSOS':v['phonSOS'][subset], 'phonEOS':v['phonEOS'][subset], 'orth':v['orth'][subset], 'wordlist':wordlist, 'frequency':v['frequency'][subset]}
        
    return holdout, train



# %%
