"""Generate traindata for modeling"""
#%%
from Traindata import Reps as d
import pandas as pd
import numpy as np
import json
import csv
import pickle
from utilities import save, collapse
from syllabics import *

#%%
# set variables
#%% separate one for 3k
words = pd.read_csv('./raw/3k.csv', names=['word'])['word'].tolist()
# remove 'null' because it causes problems
for i, word in enumerate(words):
    if type(word) == float:
        print('word #', i, 'removed ("null") because it causes problems')
        words.pop(i)

# %%
MAXORTH = 8
MAXPHON = 8
MAXSYLL = 3
MINORTH = 2
MINPHON = 3

# frequency data, read and compile into dictionary
elp = pd.read_csv('raw/elp_5.27.16.csv')

frequencies = {}
missing = [word for word in set(words) if word not in elp['Word'].tolist()]
for index, row in elp.iterrows():
    word = row['Word'].lower()
    if word in words:
        frequencies[word] = row['Freq_HAL'] + 2


#%%
for word in set(words):
    if word in missing:
        frequencies[word] = 2


#%%
tk = d(words, cmudict_supplement='./raw/3k_missing.json', maxorth=MAXORTH, maxphon=MAXPHON, minorth=MINORTH, minphon=MINPHON, maxsyll=MAXSYLL, justify='left', terminals=True, onehot=False, orthpad=9, frequency=frequencies)

#%%
# save the traindata to a pickle file with extension *.traindata

save(tk.traindata, '3k.traindata')

with open('phonreps-with-terminals.json', 'w') as t:
    json.dump(left.phonreps, t)



#%% # syllabics
with open('syllabics_3k.csv', 'w') as s:
    s.write('word,foot,heart,core,leg,head,body,phon,onset,anchor,nucleus,oncleus,coda,rime,nsyll,contour,vowels\n')

    for orthform, phonform in tk.cmudict.items():

        foot =  onset(orthform) #orthographic onset
        heart = first_vowel(orthform) # first orthographic vowel
        core = nucleus(orthform) # orthographic nucleus
        leg = oncleus(orthform) # orthographic oncleus
        head = coda(orthform)# orthographic coda
        body = rime(orthform) # orthographic rime
        phon = collapse(phonform, delimiter='-')
        onset_ = collapse(onset(phonform, orthography=False), delimiter='-')
        anchor = first_vowel(phonform, orthography=False) # first phonological vowel
        nucleus_ = collapse(nucleus(phonform, orthography=False), delimiter='-')
        oncleus_ = collapse(oncleus(phonform, orthography=False), delimiter='-')
        coda_ = collapse(coda(phonform, orthography=False), delimiter='-')
        rime_ = collapse(rime(phonform, orthography=False), delimiter='-')
        nsyll = syllables(phonform)
        contour_ = collapse(contour(phonform), delimiter='-') # stress pattern
        vowels = collapse(get_vowels(phonform), delimiter='-') # all phonological vowels

        s.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
            orthform, foot, heart, core, leg, head, body, phon, onset_, 
            anchor, nucleus_, oncleus_, coda_, rime_, nsyll, contour_, vowels))
    

s.close()
#%%