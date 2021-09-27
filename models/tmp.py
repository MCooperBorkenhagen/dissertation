
#%%
from utilities import *
#model = tf.keras.models.load_model('../outputs/taraban/taraban-model-epoch18')

test = load('data/taraban-test.traindata')
train = load('data/taraban-train.traindata')

xo = get(train, 'them', data='orth')
xp = get(train, 'them', data='phonSOS')

# %%
from Learner import Reader as r


#orth_weights0 = 
#orth_weights1 = 
orth_weights = [np.genfromtxt('data/taraban_orth_weights0.csv'), np.genfromtxt('data/taraban_orth_weights1.csv'), np.genfromtxt('data/taraban_orth_weights2.csv')]
phon_weights = [np.genfromtxt('data/taraban_phon_weights0.csv'), np.genfromtxt('data/taraban_phon_weights1.csv'), np.genfromtxt('data/taraban_phon_weights2.csv')]
output_weights = [np.genfromtxt('data/taraban_output_weights0.csv'), np.genfromtxt('data/taraban_output_weights1.csv')]


#%%
learner = r.Reader(orth_weights, phon_weights, output_weights, devices=False)
# %%
learner.model.predict([reshape(xo), reshape(xp)], verbose=0)
# %%
