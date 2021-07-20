
#%%
# This script processes the morphological variants to bolster the training pool.

from Traindata import Reps
import nltk
import pandas as pd
import json



words = [word.strip() for word in pd.read_csv('raw/wcbc-morphological-variants-to-add.csv').new_variant.tolist()]
cmuwords = list(nltk.corpus.cmudict.dict().keys())


#%%
cmudict = {word: phonforms[0] for word, phonforms in nltk.corpus.cmudict.dict().items() if word in words}

with open('kidwords-morphological-variants')


#%%
missing = [word for word in words if word not in cmuwords]

# %%
tmp = [word for word in missing if len(word) <= 8]
# %%
