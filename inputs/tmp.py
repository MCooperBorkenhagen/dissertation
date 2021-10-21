#%%
import nltk
import pandas as pd
d = nltk.corpus.cmudict.dict()
# %%

js1990 = pd.read_csv('raw/jared_etal_1990_e1.csv').word.tolist()
# %%
