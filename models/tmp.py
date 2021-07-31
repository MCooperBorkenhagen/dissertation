#%%

import keras
from utilities import L2, load
# %%
mono = load('../inputs/mono.traindata')

# %%

m = keras.models.load_model('./monosyllabic-model')

#%%