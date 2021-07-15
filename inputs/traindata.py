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
morphological_variants = [word.strip() for word in pd.read_csv('raw/wcbc-morphological-variants-to-add.csv').new_variant.tolist()]

# get outlier short words (ie words that fall within the length threshold that are weird)
outliers = pd.read_csv('./raw/wcbc-outliers.csv', header=None)[0].tolist()
# %%
MAXORTH = 8
MAXPHON = 8
MAXSYLL = 3

# get the string lebels/words
words = wcbc.orth.tolist()
words.extend(morphological_variants)


# add together the missing words into a single JSON file to supply to d
with open('./raw/kidwords-missing-from-cmudict.json') as f:
    missing1 = json.load(f)

with open('./raw/wcbc-morphological-variants-nin-cmudict.json') as f:
    missing2 = json.load(f)

missing1.update(missing2)

#%%
with open('./raw/missing-words.json', 'w') as all_missing:
    json.dump(missing1, all_missing, indent=5)


# frequency data, read and compile into dictionary
elp = pd.read_csv('raw/elp_5.27.16.csv')



frequency = {}
missing = [word for word in set(words) if word not in elp['Word'].tolist()]
for index, row in elp.iterrows():
    word = row['Word'].lower()
    if word in words:
        frequency[word] = row['Freq_HAL'] + 2


#%%
for word in set(words):
    if word in missing:
        frequency[word] = 2




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

#%%