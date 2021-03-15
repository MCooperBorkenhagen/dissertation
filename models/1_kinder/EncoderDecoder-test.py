
# %%
from EncoderDecoder import Learner
import numpy as np
from utilities import changepad, key, decode, test_acts, all_equal, cor_acts
import pandas as pd
from tensorflow.keras.utils import plot_model as plot

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

assert not np.array_equal(Xo_, _Xo), 'Your Xo patterns are the same for right and left and should not be'
assert not np.array_equal(Xp_, _Xp), 'Your Xp patterns are the same for right and left and should not be'
assert not np.array_equal(Y_, _Y), 'Your Y patterns are equal for right and left padding and shouldn not be'

#%%
#########
# LEARN #
#########
left = Learner(Xo_, Xp_, Y_, epochs=10, devices=False, monitor=False)
right = Learner(_Xo, _Xp, Y_, epochs=10, devices=False, monitor=False)
assert not left == right, 'Your two models are the same and should not be'

#%%
########
# TEST #
########
n = 2000
left_acts = test_acts([Xo_[:n], Xp_[:n]], left, layer=3)
right_acts = test_acts([_Xo[:n], _Xp[:n]], right, layer=3)
assert not np.array_equal(right_acts, left_acts), 'Your activation arrays are equal and that might not be good'
# %%
cor_acts(right_acts, left_acts)
# %%

left_right_left_acts = test_acts([Xo_[:n], _Xp[:n]], left, layer=3)
cor_acts(left_right_left_acts, left_acts)
# %%

left_right_right_acts = test_acts([Xo_[:n], _Xp[:n]], right, layer=3)
cor_acts(left_right_right_acts, left_acts)
# %%
# include the SOS and EOS:
Xp_Sos = np.load('../../inputs/phon-sos-left.npy')
Yp_Eos = np.load('../../inputs/phon-eos-left.npy')
_XpSos = np.load('../../inputs/phon-sos-right.npy')
_YpEos = np.load('../../inputs/phon-eos-right.npy')

Yp_Eos = changepad(Yp_Eos, old=9, new=0)
_YpEos = changepad(_YpEos, old=9, new=0)

assert not np.array_equal(Xp_Sos, _XpSos), 'Your right and left aligned phon inputs are all equal and they should not be'
# %%
leftT = Learner(Xo_, Xp_Sos, Yp_Eos, epochs=20, devices=False, monitor=False)
rightT = Learner(_Xo, Xp_Sos, Yp_Eos, epochs=20, devices=False, monitor=False)
# %%
leftT_acts = test_acts([Xo_[:n], Xp_Sos[:n]], leftT, layer=4)
rightT_acts = test_acts([_Xo[:n], Xp_Sos[:n]], rightT, layer=4)
assert not np.array_equal(rightT_acts, leftT_acts), 'Your activation arrays are equal and that might not be good (but it might be really good'
# %%
cor_acts(rightT_acts, leftT_acts)
#
#%%
leftT_actsA =test_acts([Xo_[:n], Xp_Sos[:n]], leftT)
rightT_actsA = test_acts([_Xo[:n], Xp_Sos[:n]], rightT)

for layer in range(len(leftT_actsA)):
    if layer != 5:
        print(layer, ':')
        print(cor_acts(np.array(leftT_actsA[layer]), np.array(rightT_actsA[layer])))
# %%
plot(leftT.model, to_file='architecture.png', show_shapes=True)
# %%
