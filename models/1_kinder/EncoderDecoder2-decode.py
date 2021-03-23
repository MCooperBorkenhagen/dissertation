#%%
from EncoderDecoder2 import Learner
import numpy as np
from utilities import changepad, key, decode, reshape, loadreps, test_acts, all_equal, cor_acts
import pandas as pd
from tensorflow.keras.utils import plot_model as plot

Xo_ = np.load('../../inputs/orth-left.npy')
Xp_ = np.load('../../inputs/phon-left.npy')
Yp_ = np.load('../../inputs/phon-left.npy')
Yp_ = changepad(Yp_, old=9, new=0)
phonreps = loadreps('../../inputs/phonreps.json', changepad=True)
orthreps = loadreps('../../inputs/raw/orthreps.json')
#%%
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

#%%
l = Learner(Xp_, Xo_, Yp_, hidden=200)
# %%
plot(l.model)
# %%
phonshape = Xp_.shape
orthshape = Xo_.shape

Xp_dummy = np.zeros(phonshape)
Xo_dummy = np.zeros(orthshape)
# %% train on dummy
l_phononly = Learner(Xp_, Xo_dummy, Yp_, hidden=200)


#%%
i = 2109
print('word to predict:', words[i])
Yhat = l_phononly.model.predict([reshape(Xp_[i]), reshape(Xo_dummy[i])])
print('Predicted:', decode(Yhat, phonreps))
print('True:', decode(Yp_[i], phonreps))



# %%
