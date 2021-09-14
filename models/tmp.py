#%%
import pandas as pd

words = pd.read_csv('../inputs/3k/words.csv', header=None)[0].tolist()

# %%
with open('tmp.csv', 'w') as f:
    for i in range(100):
        f.write('{}\n')