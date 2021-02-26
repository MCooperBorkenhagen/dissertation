
# %% Test the activations across inputs
layer_index = 4
n = 4000 # number of activations to test (ie, words)
# right


#right_acts = keras.Model(inputs=right.model.input, outputs=[layer.output for layer in right.model.layers])

#%%
#acts_all_r = right_acts([Xo[:n], Xp[:n]])
#acts_mr = np.array(acts_all_r[layer_index])

#%%
# left
#left_acts = keras.Model(inputs=left.model.input, outputs=[layer.output for layer in left.model.layers])

#acts_all_l = left_acts([Xo_[:n], Xp_[:n]])
#acts_ml = np.array(acts_all_l[layer_index])
#assert acts_ml.shape == acts_mr.shape, 'Activations are different dimensions - something is wrong'
# %%
right_acts = test_acts([_Xo[:n], _Xp[:n]], right, layer=4)
left_acts = test_acts([Xo_, Xp_], left, layer=4)

#%%
d1 = acts_mr.shape[0] # we could take dims from either ml acts or mr acts - should not make a difference
d2 = acts_mr.shape[1]*acts_mr.shape[2]


acts_mr = acts_mr.reshape((d1, d2))
acts_ml = acts_ml.reshape((d1, d2))




# %%

# applying the dist operation to each matrix takes about 2.5 minutes

dmr = dist(acts_mr)
dml = dist(acts_ml)

# %%
# pearson's r
cor = np.corrcoef(dmr, dml)
print(cor)

# %%
# spearman's rho
from scipy.stats import spearmanr as cor
print(cor(dmr, dml))

# %%
