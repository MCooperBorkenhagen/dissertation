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
from utilities import getreps, phonemedict, numphones

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

save(tk.traindata, '3k/3k.traindata')

with open('phonreps-with-terminals.json', 'w') as t:
    json.dump(tk.phonreps, t)



#%% # syllabics
with open('3k/syllabics.csv', 'w') as s:
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

        s.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            orthform, foot, heart, core, leg, head, body, phon, onset_, 
            anchor, nucleus_, oncleus_, coda_, rime_, nsyll, contour_, vowels))
    

s.close()
#%% 
syllabics = pd.read_csv('3k/syllabics.csv', keep_default_na=False)
#%%

# %%

maxfoot = max([len(foot) for foot in syllabics['foot'].tolist()])
maxbody = max([len(body) for body in syllabics['body'].tolist()])
maxonset = max([numphones(onset) for onset in syllabics['onset'].tolist()])
maxrime = max([numphones(rime) for rime in syllabics['rime'].tolist()])
#%%



oplabs = []

for i, row in syllabics.iterrows():
    # orth
    lop = (maxfoot-len(row['foot']))*'_'
    rop = (maxbody-len(row['body']))*'_'
    orthpadded = lop+row['word']+rop
    # phon
    lpp = ['_' for e in range(maxonset-numphones(row['onset']))]
    rpp = ['_' for e in range(maxrime-numphones(row['rime']))]
    phonpadded = lpp + tk.cmudict[row['word']] + rpp
    oplabs.append((orthpadded, phonpadded))


# %%

with open('raw/orthreps.json', 'r') as f:
    orthreps = json.load(f)

phonreps = getreps('https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/phonreps.csv', terminals=False)
# %%
orth = np.empty((len(oplabs), len(orthpadded)*len(orthreps['_'])))
phon = np.empty((len(oplabs), len(phonpadded)*len(phonreps['_'])))
# %%
oi = 0
pi = 1
for i, t in enumerate(oplabs):
    x = []
    print(t[oi])
    for letter in t[oi]:
        x.extend(orthreps[letter])
    orth[i] = np.array(x)
    print(t[pi])
    y = []
    for phone in t[pi]:
        y.extend(phonreps[phone])
    phon[i] = np.array(y)
    
# %%
np.savetxt('3k/orth.csv', orth, delimiter=',')
np.savetxt('3k/phon.csv', phon, delimiter=',')
# %%
with open('3k/words.csv', 'w') as f:
    for t in oplabs:
        word = t[oi].replace('_', '')
        f.write('{}\n'.format(word))

f.close()
# %%

with open('3k/phonreps.json', 'w') as t:
    json.dump(phonreps, t)

# %%
