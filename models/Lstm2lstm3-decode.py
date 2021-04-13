from Lstm2lstm3 import Learner
import numpy as np
#%%
from utilities import changepad, key, decode, reshape, loadreps, test_acts, all_equal, cor_acts, pronounce, dists
import pandas as pd
from tensorflow.keras.utils import plot_model as plot
#%%
import keras
#from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input, LSTM, Dense

############
# PATTERNS #
############
#%% load
X = np.load('../inputs/orth-left.npy')
#Y = np.load('../inputs/phon-left.npy')

Y = np.load('../inputs/phon-eos-left.npy')


#%%
Y = changepad(Y, old=9, new=0)
phonreps = loadreps('../inputs/phonreps-with-eos-only.json', changepad=True)

orthreps = loadreps('../inputs/raw/orthreps.json')
words = pd.read_csv('../inputs/encoder-decoder-words.csv', header=None)[0].tolist()
#%%


left = Learner(X, Y, transfer_state=True, hidden=600, devices=False, batch_size=150, epochs=100)
#%%


#%%
plot(left.model)

# %%

def pronounce2(i, model, X, Y, labels=None, reps=None):
    print('word to predict:', labels[i])
    out = model.predict(reshape(X[i]))
    print('Predicted:', decode(out, reps))
    print('True phon:', decode(Y[i], reps))


pronounce2(7802, left.model, X, Y, labels=words, reps=phonreps)

# %%
