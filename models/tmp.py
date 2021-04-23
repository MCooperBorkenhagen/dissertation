#%%
import numpy as np
from utilities import nearest_phoneme, load, loadreps
# %%
traindata = load('../inputs/right.traindata')
phonreps = loadreps('../inputs/phonreps-with-terminals.json')


# %%
tmp = traindata[3]['phonSOS'][0][0]
# %%
nearest_phoneme(tmp, phonreps, return_array=False, round=False, ties=False)
# %%
