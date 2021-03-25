
# %%
from EncoderDecoder import Learner
import numpy as np
from utilities import changepad, key, decode, reshape, loadreps, test_acts, all_equal, cor_acts
import pandas as pd
from tensorflow.keras.utils import plot_model as plot

import keras
from tensorflow.keras import layers
from scipy.spatial.distance import pdist as dist
from scipy.spatial.distance import squareform as matrix
#%%

# get words for reference
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

############
# PATTERNS #
############
#%% load
Xo_ = np.load('../../inputs/orth-left.npy')
Xp_ = np.load('../../inputs/phon-sos-left.npy')
Yp_ = np.load('../../inputs/phon-eos-left.npy')

Yp_ = changepad(Yp_, old=9, new=0)
#%%
phonreps = loadreps('../../inputs/phonreps-with-terminals.json', changepad=True)
#phonreps = loadreps('../../inputs/phonreps.json', changepad=True)
orthreps = loadreps('../../inputs/raw/orthreps.json')
#%%
# dummy patterns:
Xo_dummy = np.zeros(Xo_.shape)
Xp_dummy = np.zeros(Xp_.shape)
Xo_masked = np.full(Xo_.shape, 9)
Xp_masked = np.full(Xp_.shape, 9)

def sample(n, Xo, Xp, Xy, labels = None, seed = 123):
    import random
    random.seed(seed)
    s = random.sample(range(Xo.shape[0]), n)
    if labels is not None:
        labels_sampled = [labels[i] for i, label in enumerate(labels)]
        return Xo[s], Xp[s], Xy[s], labels_sampled
    else:
        return Xo[s], Xp[s], Xy[s]

#%%
Xo_1, Xp_1, Yp_1, words_1 = sample(1000, Xo_, Xp_, Yp_, labels = words)
Xo_2, Xp_2, Yp_2, words_2 = sample(700, Xo_, Xp_, Yp_, labels = words)

# %% step one: phonological pretraining
left = Learner(Xo_, Xp_, Yp_, epochs=50, hidden=300, devices=False)

#%% step two: orth and phon together
left.model.fit([Xo_, Xp_], Yp_, epochs=5)


#%% step three: just orth to phon
left.model.fit([Xo_, Xp_dummy], Yp_, epochs=5)


#%% decode
word1 = [i for i, w in enumerate(words) if w == 'half'][0]
word2 = [i for i, w in enumerate(words) if w == 'calf'][0]

phonshape = (1, Xp_.shape[1], Xp_.shape[2])
orthshape = (1, Xo_.shape[1], Xo_.shape[2])
dummyP = np.zeros(phonshape)
dummyO = np.zeros(orthshape)
#%%
i = 6793
testword = words[i]
xo = reshape(Xo_[i])
xp = reshape(Xp_[i])
out = left.model.predict([xo, xp])



#%%
print(testword)
print('Predicted: ', decode(out, phonreps))

#%%
print('Actual: ', decode(Yp_[i], phonreps))
# %%


plot(left.model)
# %%



