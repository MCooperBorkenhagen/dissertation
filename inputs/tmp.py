#%%
import pandas as pd

a = pd.read_csv('./raw/wcbc-ranked.csv').orth.tolist()
outliers = pd.read_csv('./raw/wcbc-outliers.csv', header=None)[0].tolist()

# %%
import nltk
c = nltk.corpus.cmudict.dict()
# %%
