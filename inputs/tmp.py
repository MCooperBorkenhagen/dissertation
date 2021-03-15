# %%
import json
import nltk
from Reps import Reps as data
import csv

with open('./raw/kidwords-missing-from-cmudict.json', 'r') as f:
    d = json.load(f)
# %%
cmu = nltk.corpus.cmudict.dict()
# %%


with open('raw/wcbc-outliers.csv', 'r') as f:
    r = csv.reader(f, delimiter = ',')
    l = list(r)
#%%
nl = [i[0] for i, e in enumerate(l)]

# %%
