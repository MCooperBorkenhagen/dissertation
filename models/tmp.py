#%%
import numpy as np
# %%
num_decoder_tokens = 99
sampled_token_index = 42
target_seq = np.zeros((1, 1, num_decoder_tokens))
target_seq[0, 0, sampled_token_index] = 1.
# %%
