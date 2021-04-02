#%%
from Reps import Reps as data

w = ['the', 'and']
d = data(w, eos=True, sos=False, onehot=False)
# %%
len(d.phonreps['_'])
# %%
