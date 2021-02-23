# %%
from Reps import Reps
import nltk


cmu = nltk.corpus.cmudict.dict()

#%%
pool = ['the', 'and', 'but', 'this', 'a', 'of', 'in', 'these']
d = Reps(pool, outliers=['of'], onehot=True, terminals=False, justify='left')
d.phonforms




# %%
# now this doesn't work because there is something wrong with the arrays when they don't have SOS or EOS
d = Reps(pool, outliers=['of'], onehot=True, terminals=False, justify='right')
d.phonforms_array
d.phonforms_array[0]
# %%
d = Reps(pool, outliers=['of'], onehot=True, terminals=True, justify='left')
d.phonformsSOS_array


# %%
d = Reps(pool, outliers=['of'], onehot=False, terminals=True, justify='right')
d.phonformsEOS_array
d.orthforms_array
# %%
