

#%%
from Learner import Learner
import numpy as np

#%%
from utilities import load, loadreps, reshape, choose

#%%


#%%
# load
left = load('../inputs/left.traindata')
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
data = {k:v for k, v in left.items() if k > 3}


#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# %%
orth_features = left[3]['orth'].shape[2]
phon_features = left[3]['phonSOS'].shape[2]
#%%
ps = [.2, .3, .2, .2, .05, .05] 
#%%
learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=data, hidden=400, mask_phon=False, devices=False)
#%%
learner.fitcycle(batch_size=70, cycles=2, probs=ps, evaluate=False) 

#%%
learner.model.save('base-model')
#%%

# %% determining K

np.log(8965)
# %%
