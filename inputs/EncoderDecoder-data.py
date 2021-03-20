# %%
from Reps import Reps as data
import pandas as pd
import numpy as np
import json
import csv
#%%
# set variables
WORDPATH = './raw/wcbc-ranked.csv'
wcbc = pd.read_csv(WORDPATH)

"""
No SOS-EOS in this one. These data allow for a pure autoencoder in the phonological portion of the model.
"""
# removed by hand: nan,1,8219
# get outlier short words (ie words that fall within the length threshold that are weird)
outliers = pd.read_csv('./raw/wcbc-outliers.csv', header=None)[0].tolist()
# %%
MAXORTH = 8
MAXPHON = 8

# get the string lebels/words
words = wcbc.orth.tolist()

## non-terminals
#%%
right = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, terminals=False, justify='right', onehot=False, orthpad=9)
left = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, terminals=False, justify='left', onehot=False, orthpad=9)
assert right.pool == left.pool, 'Pools are different, check call to Reps'
## Terminals
#%% SOS and EOS versions:
right_ = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, terminals=True, justify='right', onehot=False, orthpad=9)
left_ = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, terminals=True, justify='left', onehot=False, orthpad=9)
assert right_.pool == left_.pool, 'Pools are different, check call to Reps'

#%%
##########
## SAVE ##
##########
## orth ##
np.save('orth-left.npy', left.orthforms_array)
np.save('orth-right.npy', right.orthforms_array)
## phon for non-terminals ##
np.save('phon-left.npy', left.phonforms_array)
np.save('phon-right.npy', right.phonforms_array)

## With terminals ##
np.save('phon-sos-right.npy', right_.phonformsSOS_array)
np.save('phon-eos-right.npy', right_.phonformsEOS_array)
np.save('phon-sos-left.npy', left_.phonformsSOS_array)
np.save('phon-eos-left.npy', left_.phonformsEOS_array)

## the string labels/ words
with open('encoder-decoder-words.csv', 'w') as f:
    w = csv.writer(f)
    for word in left.pool:
        w.writerow([word])
f.close()

with open('phonreps.json', 'w') as p:
    json.dump(left.phonreps, p)

# get the phonreps for the data with terminals:
with open('phonreps-with-terminals.json', 'w') as t:
    json.dump(left_.phonreps, t)

#%%