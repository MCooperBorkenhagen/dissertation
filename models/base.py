

#%%
from Learner import Learner
import numpy as np

#%%
from utilities import load, loadreps, reshape

#%%



# load
left = load('../inputs/left.traindata')
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
t = {k:v for k, v in left.items() if k > 3}


#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')

# %%
orth_features = left[3]['orth'].shape[2]
phon_features = left[3]['phonSOS'].shape[2]
#%%



learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=t, hidden=400, mask_phon=False, devices=False)
#%%
learner.fitcycle(batch_size=70, cycles=4, evaluate=True) 

#%%
xO = t[4]['orth']
xP = t[4]['phonSOS']
y = t[4]['phonEOS']
i = 0

tmp = learner.model.evaluate([reshape(xO[i]), reshape(xP[i])], reshape(y[i]))


#%%

learner.fitcycle(batch_size=70, epochs=1, cycles=50)
#%%
learner.fitcycle(batch_size=10, epochs=1, cycles=10)
#%%
#%%
learner.model.save('base-model')
#%%

# evaluate all examples:
with open('../outputs/consistency.csv', 'w') as f:
    for k, v in t.items():
        for i, word in enumerate(v['wordlist']):
            acc = learner.model.evaluate(x=v[['orth']])





