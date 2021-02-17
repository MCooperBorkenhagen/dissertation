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

# padded forms
# generate pad
orthPaddedRight = {}
phonPaddedRight = {}
orthPaddedLeft = {}
phonPaddedLeft = {}
orthpad = orthreps['_']
phonpad = phonreps['_']
# replace with mask values because the decoder needs an explicit mask
# to be interpretable as a mask
for i, e in enumerate(phonpad):
    phonpad[i] = 9

# %%

phon_sos_masked_right = {}
phon_eos_masked_right = {}
phon_sos_masked_left = {}
phon_eos_masked_left = {}

orth_onehot_masked_right = {}
orth_onehot_masked_left = {}

# %%
# first create the dictionary with the right pads for the phon inputs and corresponding outputs
pi = 1 # set index of which cmudict encoding to use
for word in pool:
    padlen = cfg['maxphon']-phonlengths[word]
    p = phonreps['#']
    p.append(phonize(cmudict([word]))
    for slot in range(padlen):
        p.append(phonpad)
    phon_sos_masked_right[word] = p
    
    
    
    # now the left pad sos
    pad = []
    p = withPhonrepsL[word]
    for slot in range(padlen):
        pad.append(phonpad)
    pad.extend(p)
    phonPaddedLeft[word] = pad


# then do the same for the left pads
for word in pool:

    

# generate orth dictionary
# first, right pads
for word in pool:
    padlen = cfg['maxorth']-len(withOrthreps[word])
    p = withOrthrepsR[word]
    for slot in range(padlen):
        p.append(orthpad)
    orthPaddedRight[word] = p
# now left
for word in pool:
    padlen = cfg['maxorth']-len(withOrthreps[word])
    p = withOrthrepsL[word]
    pad = []
    for slot in range(padlen):
        pad.append(orthpad)
    pad.extend(p)
    orthPaddedLeft[word] = pad

assert set(pool) == set(orthPaddedRight.keys()) == set(phonPaddedRight.keys()) == set(orthPaddedLeft.keys()) == set(phonPaddedLeft.keys()), 'Words from your padded repsets and your wordlist do not match'






# %% Array form
phonArrayRight = []
phonArrayLeft = []
orthArrayRight = []
orthArrayLeft = []
labels = []
for word in pool:
    orthArrayRight.append(orthPaddedRight[word])
    orthArrayLeft.append(orthPaddedLeft[word])
    phonArrayRight.append(phonPaddedRight[word])
    phonArrayLeft.append(phonPaddedLeft[word])
# create the arrays
orthArrayRight = np.array(orthArrayRight)
orthArrayLeft = np.array(orthArrayLeft)
phonArrayRight = np.array(phonArrayRight)
phonArrayLeft = np.array(phonArrayLeft)






# %%
# generate syllabic data, for reference:
o = []
p = []
s = []


for word in pool:
    phonform = cmudict[word][pi]
    o.append(word)
    p.append(phonform)
    s.append(count_numerals(phonform))

assert not (orthArrayLeft == orthArrayRight).all(), 'Your orthographic arrays are identical but they should not be'
assert not (phonArrayLeft == phonArrayRight).all(), 'Your phonological arrays are identical but they should not be'


# %% reconstruct just to make sure the reps are correct

assert reconstruct(orthArrayRight, o, reps='orth'), 'The right padded orthographic representations do not match their string representations'
assert reconstruct(orthArrayLeft, o, reps='orth'), 'The left padded orthographic representations do not match their string representations'
assert reconstruct(phonArrayRight, p, reps='phon'), 'The right padded phonological representations do not match their string representations'
assert reconstruct(phonArrayLeft, p, reps='phon'), 'The left padded phonological representations do not match their string representations'



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






phonInputLabels = ['#'+word for word in pool]
phonOutputLabels = [word+'%' for word in pool]
phonOutputs = {}