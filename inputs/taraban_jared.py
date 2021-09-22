

# %%
import pandas as pd
import json
from Traindata import Reps as d
import nltk

from statistics import pstdev as sd
from utilities import *
from syllabics import *

cmu = nltk.corpus.cmudict.dict()
#%% we will use the words from 3k, plus the additional words from Jared (1997) and Taraban and McClelland (1987)
jaredA = pd.read_csv('raw/jared_1997_appendixA.csv')
jaredC = pd.read_csv('raw/jared_1997_appendixC.csv')
taraban = pd.read_csv('raw/taraban_etal_1987_words.csv')



newwords = list(set(jaredA.word.tolist()+jaredC.word.tolist()+taraban.word.tolist()))
threek = pd.read_csv('3k/words.csv', header=None)[0].tolist()

words = [word.lower() for word in list(set(threek+newwords))]

#%% find and write missing to JSON to pass to Traindata
with open('raw/3k_missing.json') as j:
    missing = json.load(j)

with open('./raw/taraban_jared_missing.json') as j:
    taraban_missing = json.load(j)


with open('./raw/disyllabic_to_monosyllabic.json') as j:
    di_to_mono = json.load(j)

missing.update(taraban_missing)
missing.update(di_to_mono)

for word in words:
    assert word in missing.keys() or word in cmu.keys(), 'need a phonological code for {}'.format(word)

with open('raw/taraban_jared_threek_missing.json', 'w', encoding='utf-8') as f:
    json.dump(missing, f, ensure_ascii=False, indent=4)
# %% frequencies

elp = pd.read_csv('raw/elp_5.27.16.csv')

frequencies = {}
missing = [word for word in words if word not in elp['Word'].tolist()]
for index, row in elp.iterrows():
    word = row['Word'].lower()
    if word in words:
        frequencies[word] = row['Freq_HAL'] + 2

for word in words:
    if word in missing:
        frequencies[word] = 2
#%%

hf_words = taraban.word[taraban.frequency == 'high'].tolist()+jaredA.word[jaredA.frequency == 'high'].tolist()
lf_words = taraban.word[taraban.frequency == 'low'].tolist()+jaredA.word[jaredA.frequency == 'low'].tolist()

assert len(hf_words) == len(lf_words), 'You have a different number of words in HF and LF conditions. Check lists'

high = 260269
low = 55

for e in zip(lf_words, hf_words):
    frequencies[e[0]] = low
    frequencies[e[1]] = high


# shouldn't matter but include:
MAXORTH = 8
MAXPHON = 8
MAXSYLL = 1
MINORTH = 2
MINPHON = 3

#%% traindata
tj = d(words, cmudict_supplement='./raw/taraban_jared_threek_missing.json', maxorth=MAXORTH, maxphon=MAXPHON, minorth=MINORTH, minphon=MINPHON, maxsyll=MAXSYLL, justify='left', terminals=True, onehot=False, orthpad=9, frequency=frequencies)
#%%
# %%
# save the traindata to a pickle file with extension *.traindata

save(tj.traindata, 'taraban/taraban.traindata')

with open('taraban/phonreps-with-terminals.json', 'w') as t:
    json.dump(tj.phonreps, t)



# %%
with open('taraban/syllabics.csv', 'w') as s:
    s.write('word,foot,heart,core,leg,head,body,phon,onset,anchor,nucleus,oncleus,coda,rime,nsyll,contour,vowels\n')

    i = 1
    for orthform, phonform in tj.cmudict.items():

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
    
        print('{} of {} words done'.format(i, len(tj.cmudict.keys())))
        i += 1

s.close()

#%%

with open('taraban/words.csv', 'w') as f:
    for word in tj.pool:
        f.write('{}\n'.format(word))
f.close()
# %%
