#%%
from utilities import load, split, loadreps, save, scale
import pandas as pd
import numpy as np

tk = load('../inputs/taraban/taraban.traindata')
#words = pd.read_csv('../inputs/taraban/words.csv', header=None)
# phonreps and orthreps
phonreps = loadreps('../inputs/taraban/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')
# %%
