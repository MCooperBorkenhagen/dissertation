"""Generate traindata for modeling"""
#%%
from Traindata import Reps as d
import pandas as pd
import numpy as np
import json
import csv
import pickle
from utilities import save
#%%
# set variables
WORDPATH = './raw/wcbc-ranked.csv'
wcbc = pd.read_csv(WORDPATH)


# get outlier short words (ie words that fall within the length threshold that are weird)
outliers = pd.read_csv('./raw/wcbc-outliers.csv', header=None)[0].tolist()
# %%
MAXORTH = 8
MAXPHON = 8
MAXSYLL = 3

# get the string lebels/words
words = wcbc.orth.tolist()

#%%
more = ['ration', 'nation', 'shriek']
words.extend(more)

#%%
left = d(words, outliers=outliers, cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', maxorth=MAXORTH, maxphon=MAXPHON, maxsyll=MAXSYLL, justify='left', terminals=True, onehot=False, orthpad=9)
right = d(words, outliers=outliers, cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', maxorth=MAXORTH, maxphon=MAXPHON, maxsyll=MAXSYLL, justify='right', terminals=True, onehot=False, orthpad=9)

#%%
# test that the words for each set are the same:
for length, traindict in left.traindata.items():
    assert left.traindata[length]['wordlist'] == right.traindata[length]['wordlist']
    print('Length', length, 'passed')
#%%
# save the traindata to a pickle file with extension *.traindata

save(left.traindata, 'left.traindata')
save(right.traindata, 'right.traindata')
# %%
with open('phonreps-with-terminals.json', 'w') as t:
    json.dump(left.phonreps, t)