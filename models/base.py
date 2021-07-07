

#%%
from Learner import Learner
import numpy as np

#%%
from utilities import load, loadreps

#%%



# load
left = load('../inputs/left.traindata')
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
t = {k:v for k, v in left.items() if k > 3}


#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# %%
orth_features = left[4]['orth'].shape[2]
phon_features = left[4]['phonSOS'].shape[2]
#%%
learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=t, mask_phon=False, devices=False)

#%%
learner.fitcycle(batch_size=50, epochs=1, cycles=100)
#%%