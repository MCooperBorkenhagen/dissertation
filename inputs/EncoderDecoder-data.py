# %%
from Reps import Reps as data
import pandas as pd
import numpy as np
import csv
#%%
# set variables
WORDPATH = './raw/wcbc-ranked.csv'
wcbc = pd.read_csv(WORDPATH)

"""
No SOS-EOS in this one. These data allow for a pure autoencoder in the phonological portion of the model.
"""
# get outlier short words (ie words that fall within the length threshold that are weird)
outliers = pd.read_csv('./raw/wcbc-outliers.csv', header=None)[0].tolist()
#outliers.append('rope')

MAXORTH = 8
MAXPHON = 8

words = wcbc.orth.tolist()
#%%
right = data(words, outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, terminals=False, justify='right', onehot=False, orthpad=9)

np.save('orth-right.npy', right.orthforms_array)
np.save('phon-right.npy', right.phonforms_array)

# %%
left = data(words, outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, terminals=False, justify='left', onehot=False, orthpad=9)



np.save('orth-left.npy', left.orthforms_array)
np.save('phon-left.npy', left.phonforms_array)

#%%
assert right.pool == left.pool, 'Pools are different, check call to Reps'


#%%
with open('encoder-decoder-words.csv', 'w') as f:
    w = csv.writer(f)
    for word in left.pool:
        w.writerow([word])
f.close()

#%%