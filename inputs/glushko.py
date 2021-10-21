
#%%
import pandas as pd
import nltk

cmu = nltk.corpus.cmudict.dict()

# %%
g1 = pd.read_csv('raw/glushko_a1.csv')
g2 = pd.read_csv('raw/glushko_a2.csv')
g3 = pd.read_csv('raw/glushko_a3.csv')
# %%
g1_missing = [word for word in g1.orth.tolist() if word not in cmu.keys()]
g2_missing = [word for word in g2.orth.tolist() if word not in cmu.keys()]
g3_missing = [word for word in g3.orth.tolist() if word not in cmu.keys()]

# %%
for g in [g1_missing, g2_missing, g3_missing]:
    print(len(g))
# %%
