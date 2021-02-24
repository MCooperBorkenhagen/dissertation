
# %%
from seq2seq import Learner
import numpy as np



Xo = np.load('../../inputs/orth-left.npy')
Xp = np.load('../../inputs/phon-left.npy')
Y = np.load('../../inputs/phon-left.npy')


l = Learner(Xo, Xp, Y)
# %%
