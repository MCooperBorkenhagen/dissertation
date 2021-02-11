
# %%
from Learner import Learner
import numpy as np
import pandas as pd

# %%
import json
import pickle
import os
# %%
import keras
from tensorflow.keras import layers
from scipy.spatial import distance_matrix as dist


# %%
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
test_acts = keras.Model(inputs=mr.model.inputs, outputs=[layer.output for layer in mr.model.layers])
acts_all = test_acts(Xr)
acts_mr = acts_all[2]

# %%
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist as dist

def L2(x, y):
    return(np.linalg.norm(x-y))

v = Xr.shape[0]

dr = np.zeros([v, v])

tmp = np.array(acts_mr[0:3])
wrds = mr.labels[0:3]

da = {}
for e in range(tmp.shape[0]):
    for o in range(tmp.shape[0]):
        d1 = []
        for t in range(tmp[e].shape[0]):
            d1.append(L2(tmp[e][t], tmp[o][t]))
        if wrds[e] not in da.keys():        
            da[wrds[e]] = d1


# %%

for p in range(len(da)):
    print(p)    





# %%



da = np.array(da)
# %%
