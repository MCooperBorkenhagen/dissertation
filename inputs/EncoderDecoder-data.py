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


# subset only kidwords for which we have phonological representations in cmudict (Kaycie is getting the rest)
# add kaycie's data here when she gets it to you:
missing = []
for word in wcbc.orth.tolist():
    if word not in cmudict.keys():
        missing.append(word)
missing = list(set(missing))
print(len(missing), 'words from wcbc not in cmudict')


pool = [word for word in wcbc.orth.tolist() if word in cmudict.keys()] 
pool = [word for word in pool if len(word) <= cfg['maxorth']]
pool = [word for word in pool if word not in outliers]


# get binary representations
phonreps = getreps(cfg['phonreps'])
with open(cfg['orthreps']) as j:
    orthreps = json.load(j)


# clean the data a bit
# skip words that are one letter long

for word in pool:
    if len(word) == 1:
        pool.remove(word)
        print(word, 'removed from pool')


#
def phonize(codelist, code_index): # from representPhon
    assert type(codelist[0]) == list, 'A list of lists is needed'
    code = codelist[code_index]
    rep = []
    for phone in code:
        rep.append(phonreps[phone])
    return(rep)
# check that all the phonological units are proper:
phones = []
for word in pool:
    for phonform in cmudict[word]:
           for phoneme in phonform:
               phones.append(phoneme)
assert set(phones).issubset(phonreps.keys()), 'Some phonemes are missing in your phonreps'

# %%



# 

# generate the binary phonological representation for all words 
# in the set (ie, in pool).

# specify which of all the phonological representations
# should be selected from cmudict. There is always at least
# one, but sometimes more than one. An alternative here
# would be to sample a random one. This part of the cleaning
# is implemented here because it is the point at which
# phonological wordforms are selected

# Remember that you present a restriction on the number of phonemes 
# included here, which can be changed
pi = 0
# remove words that have too many phonemes
toomanyphones = [] = []
for word in pool:
    rep = phonize(cmudict[word], pi)
    if len(rep) > cfg['maxphon']:
        toomanyphones.append(word)

pool = [word for word in pool if word not in toomanyphones]
print(toomanyphones, 'removed because they have too many phonemes')


withPhonreps = {word:phonize(cmudict[word], pi) for word in pool}

for k, v in withPhonreps.items():
    if len(v) > cfg['maxphon']:
        withPhonreps.pop(word)
        pool.remove(word)




# orthography:
# check that all the letters in pool are represented in orthreps:
letters = []
for word in pool:
    for l in word:
        letters.append(l)
assert set(letters).issubset(orthreps.keys()), 'there are missing binary representations for letters in the set of words'


# function for generating binary reps for orthforms:

def orthize(orthform):
    rep = []
    for c in orthform:
        rep.append(orthreps[c])
    return(rep)
# perform a test on all letters, making sure that we have a binary representation for it
withOrthreps = {word: orthize(word) for word in pool}

assert set(withOrthreps.keys()) == set(withPhonreps.keys()), 'The keys in your orth and phon dictionaries do not match'
 # %%

 # let's generate 