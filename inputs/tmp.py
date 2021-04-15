#%%
import pandas as pd

import nltk
c = nltk.corpus.cmudict.dict()
# %%
t = c['the'][0]


# %%
def n_syllables(x):
    count = 0
    for e in x:
        if any(ch.isdigit() for ch in e):
            count += 1
    return(count)