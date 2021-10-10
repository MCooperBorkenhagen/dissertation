#%%
from utilities import load, get, loadreps
# %%
train = load('data/taraban_pilot//taraban-train.traindata')
test = load('data/taraban_pilot//taraban-test.traindata')

i = 10
xo = get(train, 'them', data='orth')
xp = get(train, 'them', data='phonSOS')

# %%
