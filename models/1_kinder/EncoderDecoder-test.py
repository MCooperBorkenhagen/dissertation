
# %%
from seq2seq import Learner
import numpy as np
from utilities import changepad, key, decode, test_acts
import pandas as pd

import keras
from tensorflow.keras import layers
from scipy.spatial.distance import pdist as dist
from scipy.spatial.distance import squareform as matrix
#%%

# get words for reference
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None) 

############
# PATTERNS #
############
#%%
# left justified
Xo_ = np.load('../../inputs/orth-left.npy')
Xp_ = np.load('../../inputs/phon-left.npy')
Y_ = np.load('../../inputs/phon-left.npy')
Y_ = changepad(Y_, old=9, new=0)

# right justified
_Xo = np.load('../../inputs/orth-right.npy')
_Xp = np.load('../../inputs/phon-right.npy')
_Y = np.load('../../inputs/phon-right.npy')
_Y = changepad(_Y, old=9, new=0)

assert not (_Xo==Xo_).all(), 'Your Xo patterns are the same for right and left and should not be'
assert not (_Xp==Xp_).all(), 'Your Xp patterns are the same for right and left and should not be'


#%%
#########
# LEARN #
#########
left = Learner(Xo_, Xp_, Y_, epochs=10, devices=False)
right = Learner(_Xo, _Xp, _Y, epochs=10, devices=False)

########
# TEST #
########

# %% Test the activations across inputs
layer_index = 4
n = 4000 # number of activations to test (ie, words)
# right


#right_acts = keras.Model(inputs=right.model.input, outputs=[layer.output for layer in right.model.layers])

#%%
#acts_all_r = right_acts([Xo[:n], Xp[:n]])
#acts_mr = np.array(acts_all_r[layer_index])

#%%
# left
#left_acts = keras.Model(inputs=left.model.input, outputs=[layer.output for layer in left.model.layers])

#acts_all_l = left_acts([Xo_[:n], Xp_[:n]])
#acts_ml = np.array(acts_all_l[layer_index])
#assert acts_ml.shape == acts_mr.shape, 'Activations are different dimensions - something is wrong'
# %%
right_acts = test_acts([Xo[:n], Xp[:n]])



d1 = acts_mr.shape[0] # we could take dims from either ml acts or mr acts - should not make a difference
d2 = acts_mr.shape[1]*acts_mr.shape[2]


acts_mr = acts_mr.reshape((d1, d2))
acts_ml = acts_ml.reshape((d1, d2))




# %%

# applying the dist operation to each matrix takes about 2.5 minutes

dmr = dist(acts_mr)
dml = dist(acts_ml)

# %%
# pearson's r
cor = np.corrcoef(dmr, dml)
print(cor)

# %%
# spearman's rho
from scipy.stats import spearmanr as cor
print(cor(dmr, dml))

# %%
