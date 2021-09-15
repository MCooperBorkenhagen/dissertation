#%%
import numpy as np
import pandas as pd
import keras
import tensorflow as tf

import time
#%%
words = pd.read_csv('3k/words.csv', header=None)[0].tolist()
#%%
tmd = pd.read_csv('raw/taraban_etal_1987_words.csv')

tmdr = tmd[tmd['condition'] != 'nonword']['word'].tolist()
# %%
for word in tmdr:
    if word not in words:
        print(word)
# %%
missing = [word for word in tmdr if word not in words]
# %%



# %%
#%%
def alloc(l1, l2=None, filler='XXXXXXXXXX'):
    out = {}
    for e in l1:
        if l2 is None:
            out[e] = filler
        else:
            sub = {}
            for g in l2:
                sub[g] = filler
            out[e] = sub

    return out

l1 = words
l2 = ['orthform', 'foot', 'heart', 'core', 'leg', 'head', 'body', 'phon', 'onset_', 'anchor', 'nucleus_', 'oncleus_', 'coda_', 'rime_', 'nsyll', 'contour_', 'vowels']

syll = alloc(l1, l2)
#%%
for orthform, phonform in tj.cmudict.items():

    syll[orthform]['foot'] =  onset(orthform) #orthographic onset
    syll[orthform]['heart'] = first_vowel(orthform) # first orthographic vowel
    syll[orthform]['core'] = nucleus(orthform) # orthographic nucleus
    syll[orthform]['leg'] = oncleus(orthform) # orthographic oncleus
    syll[orthform]['head'] = coda(orthform)# orthographic coda
    syll[orthform]['body'] = rime(orthform) # orthographic rime
    syll[orthform]['phon'] = collapse(phonform, delimiter='-')
    syll[orthform]['onset_'] = collapse(onset(phonform, orthography=False), delimiter='-')
    syll[orthform]['anchor'] = first_vowel(phonform, orthography=False) # first phonological vowel
    syll[orthform]['nucleus_'] = collapse(nucleus(phonform, orthography=False), delimiter='-')
    syll[orthform]['oncleus_'] = collapse(oncleus(phonform, orthography=False), delimiter='-')
    syll[orthform]['coda_'] = collapse(coda(phonform, orthography=False), delimiter='-')
    syll[orthform]['rime_'] = collapse(rime(phonform, orthography=False), delimiter='-')
    syll[orthform]['nsyll'] = syllables(phonform)
    syll[orthform]['contour_'] = collapse(contour(phonform), delimiter='-') # stress pattern
    syll[orthform]['vowels'] = collapse(get_vowels(phonform), delimiter='-') # all phonological vowels
    print(*)

# %%
