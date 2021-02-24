# %%
from Reps import Reps


#%%
pool = ['the', 'and', 'but', 'this', 'a', 'of', 'in', 'these']
d = Reps(pool, outliers=['of'], onehot=False, terminals=False, justify='left')


