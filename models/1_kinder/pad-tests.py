
# %%
from Learner import Learner
import numpy as np
import json
import pickle
import os
# %%
import keras.backend as K

# define a function to acquire hidden layer activations:
def acts(X, model, layer):

    f = K.function([model.layers[layer].input], [model.layers[(layer+1)].output])
    return(f(X))



# %%
# get representations
Xr = np.load("../../inputs/orth_pad_right.npy")
Yr = np.load("../../inputs/phon_pad_right.npy")

Xl = np.load("../../inputs/orth_pad_left.npy")
Yl = np.load("../../inputs/phon_pad_left.npy")

with open('../../inputs/params.json') as f:
    cfg = json.load(f)
# %% right pad

mr = Learner(Xr, Yr, train_proportion=(1-cfg['validation_split']), hidden_layers=cfg['hidden_layers'], hidden=cfg['hidden_size'], seed=cfg['seed'])

pickle.dump(ml, '')
mr.model.save('../../outputs/')



# left pad
ml = Learner(Xl, Yl, train_proportion=(1-cfg['validation_split']), hidden_layers=cfg['hidden_layers'], hidden=cfg['hidden_size'], seed=cfg['seed'])


# %%
