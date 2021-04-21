#%%
import pandas as pd
from Reps2b import Reps
import nltk
c = nltk.corpus.cmudict.dict()
# %%
words = ['the', 'and', 'if', 'they', 'something', 'hello']


# %%
d = Reps(words, phonpath='raw/phonreps.csv', onehot=False, terminals=True, test_reps=True)
# %%
d.traindata
# %%
import numpy as np

target_seq = np.zeros((1, 2, 3))
# %%
