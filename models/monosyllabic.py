#%%
from Learner import Learner
import numpy as np
import pandas as pd
import keras
from utilities import load, loadreps, reshape, choose, split


mono = load('../inputs/mono.traindata')
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
test, train = split(mono, .05)



#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')



#%%
orth_features = train[4]['orth'].shape[2]
phon_features = train[4]['phonSOS'].shape[2]
#%% probabilities for sampling phonological lengths during fitcycle() for training
probs = [.3, .4, .2, .075, .025]
#%%




learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, modelname='monosyllabic', hidden=400, mask_phon=False, devices=False)


#%% using same function for sampling probabilities from Seidenberg & McClelland (1989)
frequencies = {word: v['frequency'][i] for k, v in mono.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(frequencies.values())
K = p/np.log(maxf)



#%%
learner.fitcycle(batch_size=70, cycles=10, probs=probs, K=K, evaluate=False) 


#%%
learner.model.save('monosyllabic-model')
#m = keras.models.load_model('./base-model')

# %%
learner.read('hand', ties='sample')
# %%
__, __, yp = learner.find('think')

# %%
tmp = learner.test('think', return_phonform=True, returns='all', ties = 'sample', phonreps=None)
# %%

#%%


# %%
