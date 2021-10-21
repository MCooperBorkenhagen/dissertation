


"""Generate traindata for modeling"""
#%%
from Traindata import Reps as d
import pandas as pd
import json
from utilities import save, collapse
from syllabics import *

#%%
# set variables
WORDPATH = './raw/wcbc-ranked.csv'
wcbc = pd.read_csv(WORDPATH)

# get chateau experimental lists for training:
ca = pd.read_csv('raw/chateau_etal_2003_a.csv').word.tolist()
cb = pd.read_csv('raw/chateau_etal_2003_b.csv').word.tolist()
cc = pd.read_csv('raw/chateau_etal_2003_c.csv').word.tolist()

# %%
chateau_words = list(set(ca + cb + cc))
morphological_variants = [word.strip() for word in pd.read_csv('raw/wcbc-morphological-variants-to-add.csv').new_variant.tolist()]


# get outlier short words (ie words that fall within the length threshold that are weird)
outliers = pd.read_csv('./raw/wcbc-outliers.csv', header=None)[0].tolist()
# %%
MAXORTH = 8
MAXPHON = 8
MAXSYLL = 3
MINORTH = 2
MINPHON = 3

# get the string labels/words
words = wcbc.orth.tolist()
words.extend(morphological_variants)
words.extend(chateau_words)

#%%
# add together the missing words into a single JSON file to supply to d
with open('./raw/kidwords-missing-from-cmudict.json') as f:
    missing1 = json.load(f)

with open('./raw/wcbc-morphological-variants-nin-cmudict.json') as f:
    missing2 = json.load(f)

# chateau words (there are only a few)
with open('./raw/chateau_etal_2003_missing.json') as m:
    missing3 = json.load(m)

missing1.update(missing2)
missing1.update(missing3)


with open('raw/missing-words-chateau-all.json', 'w') as f:
    json.dump(missing1, f, indent=5)

#%%
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
chateau = d(words, outliers=outliers, cmudict_supplement='raw/missing-words-chateau-all.json', maxorth=MAXORTH, maxphon=MAXPHON, minorth=MINORTH, minphon=MINPHON, maxsyll=MAXSYLL, justify='left', terminals=True, onehot=False, orthpad=9, frequency=frequencies)


#%%
# save the traindata to a pickle file with extension *.traindata
save(chateau.traindata, 'chateau/chateau.traindata')

with open('chateau/phonreps-with-terminals.json', 'w') as t:
    json.dump(chateau.phonreps, t)



#%% # syllabics

with open('chateau/syllabics.csv', 'w') as s:
    s.write('word,foot,heart,core,leg,head,body,phon,onset,anchor,nucleus,oncleus,coda,rime,nsyll,contour,vowels\n')

    for orthform, phonform in chateau.cmudict.items():

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