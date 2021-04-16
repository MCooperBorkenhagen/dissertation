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
MAXSYLL = 3

# get the string lebels/words
words = wcbc.orth.tolist()

#%%
more = ['ration', 'nation', 'horatio', 'rational', 'shriek']
words.extend(more)
## non-terminals
#%%
right = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, maxsyll=MAXSYLL, eos=False, sos=False, justify='right', onehot=False, orthpad=9)

#%%
left = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, maxsyll=MAXSYLL, eos=False, sos=False,  justify='left', onehot=False, orthpad=9)

#%%
assert right.pool == left.pool, 'Pools are different, check call to Reps'
## Terminals
#%% SOS and EOS versions:
right_ = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, maxsyll=MAXSYLL,  eos=True, sos=True, justify='right', onehot=False, orthpad=9)
left_ = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, maxsyll=MAXSYLL,  eos=True, sos=True, justify='left', onehot=False, orthpad=9)
assert right_.pool == left_.pool, 'Pools are different, check call to Reps'
#%% EOS only versions
right_eosonly = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, maxsyll=MAXSYLL, eos=True, sos=False, justify='right', onehot=False, orthpad=9)
left_eosonly = data(words, phonpath='raw/phonreps.csv', cmudict_supplement='./raw/kidwords-missing-from-cmudict.json', outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, maxsyll=MAXSYLL, eos=True, sos=False, justify='left', onehot=False, orthpad=9)
assert right_eosonly.pool == left_eosonly.pool, 'Pools are different, check call to Reps'


#%%
##########
## SAVE ##
##########
## orth ##
np.save('orth-left.npy', left.orthforms_array)
np.save('orth-right.npy', right.orthforms_array)
## phon for non-terminals ##
np.save('phon-left-no-terminals.npy', left.phonforms_array)
np.save('phon-right-no-terminals.npy', right.phonforms_array)

## With terminals ##
np.save('phon-sos-terminals-right.npy', right_.phonformsSOS_array)
np.save('phon-eos-terminals-right.npy', right_.phonformsEOS_array)
np.save('phon-sos-terminals-left.npy', left_.phonformsSOS_array)
np.save('phon-eos-terminals-left.npy', left_.phonformsEOS_array)
## eos only:
np.save('phon-inputs-for-eos-right.npy', right_eosonly.phonformsX_array)
np.save('phon-with-eos-right.npy', right_eosonly.phonformsEOS_array)
np.save('phon-inputs-for-eos-left.npy', left_eosonly.phonformsX_array)
np.save('phon-with-eos-left.npy', left_eosonly.phonformsEOS_array)



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

with open('phonreps-with-eos-only.json', 'w') as t:
    json.dump(left_eosonly.phonreps, t)

#%%