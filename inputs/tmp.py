#%%
import pandas as pd
from Reps2 import Reps
import nltk
c = nltk.corpus.cmudict.dict()
# %%
words = ['the', 'and', 'if', 'they']
lengths = [len(word) for word in words]
traindata = {}
for i, e in enumerate(words):
    length = len(e)
    w = {}
    w['i'] = i
    w['e']
    traindata[length] = e

# %%
d = Reps(words, phonpath='raw/phonreps.csv')
# %%
