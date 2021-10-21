
#%%
import nltk
from utilities import *
from Learner import Reader as R
import numpy as np
import os
from keras.backend import clear_session as clear

d = nltk.corpus.cmudict.dict()

phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')



#%%
PATH = '../outputs/taraban_crossval'
run_id = '35'
epochs = 63

orth_weights = read_weights(PATH, run_id, epochs, 'orth')
phon_weights = read_weights(PATH, run_id, epochs, 'phon')
output_weights = read_weights(PATH, run_id, epochs, 'output')

reader = R.Reader(orth_weights, phon_weights, output_weights, orthreps=orthreps, phonreps=phonreps, devices=False)


word = 'seize'
cmu = ["S", "IY1", "Z"] 
 

reader.read(word, cmu, ties='sample')


#%%

# %%
a, b, c = get_weights(m, 'phon')
# %%
