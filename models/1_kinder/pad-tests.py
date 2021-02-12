
# %%
from Learner import Learner
import numpy as np
import pandas as pd

import time
import json
import pickle
import os

import keras
from tensorflow.keras import layers
from scipy.spatial import distance_matrix as dist

# get representations
Xr = np.load("../../inputs/orth_pad_right.npy")
Yr = np.load("../../inputs/phon_pad_right.npy")

Xl = np.load("../../inputs/orth_pad_left.npy")
Yl = np.load("../../inputs/phon_pad_left.npy")


syllabics = pd.read_csv("../../inputs/syllabics.csv", sep = ",")
words = syllabics.orth.tolist()



with open('../../inputs/params.json') as f:
    cfg = json.load(f)
# %% right pad
mr = Learner(Xr, Yr, labels=words, train_proportion=(1-cfg['validation_split']), hidden_layers=cfg['hidden_layers'], hidden=cfg['hidden_size'], seed=cfg['seed'])
# left pad
ml = Learner(Xl, Yl, labels=words, train_proportion=(1-cfg['validation_split']), hidden_layers=cfg['hidden_layers'], hidden=cfg['hidden_size'], seed=cfg['seed'])


# %%
layer_index = 2

test_acts = keras.Model(inputs=mr.model.inputs, outputs=[layer.output for layer in mr.model.layers])
acts_all_r = test_acts(Xr)
acts_mr = np.array(acts_all_r[layer_index])

acts_all_l = test_acts(Xl)
acts_ml = np.array(acts_all_l[layer_index])
assert acts_ml.shape == acts_mr.shape, 'Activations are different dimensions - something is wrong'
# %%
d1 = acts_mr.shape[0] # we could take dims from either ml acts or mr acts - should not make a difference
d2 = acts_mr.shape[1]*acts_mr.shape[2]


acts_mr = acts_mr.reshape((d1, d2))
acts_ml = acts_ml.reshape((d1, d2))




# %%
from scipy.spatial.distance import pdist as dist
from scipy.spatial.distance import squareform as matrix
# applying the dist operation to each matrix takes about 2.5 minutes

dmr = dist(acts_mr)
dml = dist(acts_ml)



# %%
cor = np.corrcoef(dmr, dml)
# running this on the right versus left added data yielded a correlation of r = .5153
# %%
