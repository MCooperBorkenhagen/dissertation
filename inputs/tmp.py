#%%

from utilities import load

d = load('left.traindata')

# %%
words = [word for v in d.values() for word in v['wordlist']]
# %%

# %%
import pandas as pd

syllabics = pd.read_csv('syllabics.csv')

# %%
