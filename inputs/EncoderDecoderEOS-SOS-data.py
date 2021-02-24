# %%
from Reps import Reps as data
import pandas as pd
import numpy as np
import csv
#%%
# set variables
WORDPATH = './raw/wcbc-ranked.csv'
wcbc = pd.read_csv(WORDPATH)

# removed by hand: nan,1,8219
# get outlier short words (ie words that fall within the length threshold that are weird)
outliers = pd.read_csv('./raw/wcbc-outliers.csv', header=None)[0].tolist()
#outliers.append('rope')

MAXORTH = 8
MAXPHON = 8

words = wcbc.orth.tolist()
#%%
right = data(words, outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, terminals=True, justify='right', onehot=False, orthpad=9)

#%%
o = right.orthforms_array
pI = right.phonformsSOS_array
pO = right.phonformsEOS_array
#%%
np.save('orth-right.npy', right.orthforms_array)
np.save('phon-sos-right.npy', right.phonformsSOS_array)
np.save('phon-eos-right.npy', right.phonformsEOS_array)

# %%
left = data(words, outliers=outliers, maxorth=MAXORTH, maxphon=MAXPHON, terminals=True, justify='left', onehot=False, orthpad=9)



np.save('orth-left.npy', left.orthforms_array)
np.save('phon-sos-left.npy', left.phonformsSOS_array)
np.save('phon-eos-left.npy', left.phonformsEOS_array)
#%%

with open('encoder-decoder-words.csv', 'w') as f:
    w = csv.writer(f)
    for word in words:
        w.writerow([word])
f.close()

#%%