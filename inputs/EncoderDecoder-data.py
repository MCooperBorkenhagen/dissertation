# %%
import pandas as pd
import nltk
import random
import numpy as np
import os
import json
import re
import random
import nltk
import copy
from utilities import *

with open('./raw/params.json') as j:
    cfg = json.load(j)

# set variables
WORDPATH = './raw/wcbc-ranked.csv'


# get cmu words
cmudict = nltk.corpus.cmudict.dict() # the raw cmudict object
wcbc = pd.read_csv(WORDPATH)

# get outlier short words (ie words that fall within the length threshold that are weird)
outliers = pd.read_csv('./raw/wcbc-outliers.csv', header=None)[0].tolist()

# %%
maxorth = 8
maxphon = 8



# %% write objects
syllabics = {'orth': o, 'phon': p, 'syllables': s, 'phon_input': phonInputLabels, 'phon_output': phonOutputLabels}
syllabics = pd.DataFrame.from_dict(syllabics)
syllabics.to_csv('syllabics-encoderDecoder.csv', index=False)

# write arrays and labels:
with open('orth-onehot-masked-right.npy', 'wb') as f:
    np.save(f, orthArrayRight)
with open('orth-onehot-masked-left.npy', 'wb') as f:
    np.save(f, orthArrayLeft)
with open('phon-sos-masked-right.npy', 'wb') as f:
    np.save(f, phonArrayRight)
with open('phon-eos-masked-right.npy', 'wb') as f:
    np.save(f, phonArrayLeft)
with open('phon-sos-masked-left.npy', 'wb') as f:
    np.save(f, phonArrayRight)
with open('phon-eos-masked-left.npy', 'wb') as f:
    np.save(f, phonArrayLeft)


# %%
