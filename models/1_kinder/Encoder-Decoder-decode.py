
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
Yp_ = np.load('../../inputs/phon-sos-left.npy')
Yp_ = changepad(Yp_, old=9, new=0)
phonreps = loadreps('../../inputs/phonreps-with-terminals.json', changepad=True)
orthreps = loadreps('../../inputs/raw/orthreps.json')


# %% learn
leftT = Learner(Xo_, Xp_, Yp_, epochs=20, devices=False, monitor=False)

#%% decode
word1 = [i for i, w in enumerate(words) if w == 'half'][0]
word2 = [i for i, w in enumerate(words) if w == 'calf'][0]

phonshape = (1, Xp_.shape[1], Xp_.shape[2])
dummyP = np.zeros(phonshape)

dummyO = np.zeros((1, Xo_.shape[1], Xo_.shape[2]))
#%%
i = word1
p = word2 
testword = words[i]
xo = reshape(Xo_[i])
xp = reshape(Xp_[p])
out = leftT.model.predict([xo, xp])



#%%
print(testword)
print('Predicted: ', decode(out, phonreps))

#%%
print('Actual: ', decode(Yp_[i], phonreps))
# %%


