

#%%
from Learner import Learner
import numpy as np
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

# %%
orth_features = data[4]['orth'].shape[2]
phon_features = data[4]['phonSOS'].shape[2]
#%% probabilities for sampling phonological lengths during fitcycle()
ps = [.2, .3, .2, .2, .05, .05] 
#%%
learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=data, hidden=400, mask_phon=False, devices=False)


#%% using same function for sampling probabilities from Seidenberg & McClelland (1989)
frequencies = {word: v['frequency'][i] for k, v in data.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(frequencies.values())
K = p/np.log(maxf)



#%%
learner.fitcycle(batch_size=70, cycles=1, probs=ps, K=K, evaluate=False) 
# %%
# just trying out this trivial versino of test()
# need to actually build the ability to test
learner.test('think', ties='sample')
# %%
