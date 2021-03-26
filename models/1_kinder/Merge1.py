
#%%
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, TimeDistributed, Concatenate, Bidirectional
from utilities import changepad, loadreps, decode, reshape 
import numpy as np
import time
import pandas as pd

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#%%
from tensorflow.keras.utils import plot_model as plot


#def morepad(a, value=9):
#   for i in 


#%%

Xo = np.load('../../inputs/orth-left.npy')
Xp = np.load('../../inputs/phon-left.npy')
#%%
orthshape = Xo.shape
phonshape = Xp.shape


Xo_dummy = np.zeros(orthshape)
Xp_dummy = np.zeros(phonshape)

Xo_mask = np.where(Xo==1, 0, Xo)
Xp_mask = np.where(Xp==1, 0, Xp)
#%%




Y = np.load('../../inputs/phon-left.npy')
Y = changepad(Y, old=9, new=0)

words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()
#phonreps = loadreps('../../inputs/phonreps.json', changepad=True)

phonreps = loadreps('../../inputs/phonreps.json', changepad=True)
orthreps = loadreps('../../inputs/raw/orthreps.json', changepad=True)

#%%

# params
hidden = 300
optimizer='rmsprop'
loss="categorical_crossentropy"
transfer_function = 'sigmoid'

# %% learner
orth_inputs = Input(shape=(None, Xo.shape[2]), name='orth_input')
orth_inputs_masked = Masking(mask_value=9, name='orth_mask')(orth_inputs)
orth = LSTM(hidden, return_sequences=True, return_state=True, name = 'orthographic_lstm') # set return_sequences to True if no RepeatVector
orth_outputs, orth_hidden, orth_cell = orth(orth_inputs_masked)
orth_state = [orth_hidden, orth_cell]

orth_dense = Dense(500, activation=transfer_function, name='orth_dense')
orth_decomposed = orth_dense(orth_outputs)

phon_inputs = Input(shape=(None, Xp.shape[2]), name='phon_input')
phon_inputs_masked = Masking(mask_value=9, name='phon_mask')(phon_inputs)
phon = LSTM(hidden, return_sequences=True, return_state=True, name='phonological_lstm')
phon_outputs, phon_hidden, phon_cell = phon(phon_inputs_masked, initial_state=orth_state)
phon_state = [phon_hidden, phon_cell]


phon_dense = Dense(500, activation=transfer_function, name='phon_dense')
phon_decomposed = phon_dense(phon_outputs)


#%%
#merge = Bidirectional(LSTM(1000, return_sequences=True, name='merge'))
#merge_output = merge([orth_outputs, phon_outputs], initial_state=[orth_state, phon_state])


#%%
merge = Concatenate(name='merge')
merge_outputs = merge([phon_decomposed, orth_decomposed])
#%%

deep = Dense(200, activation='sigmoid', name='deep_layer2')
deep_outputs = deep(merge_outputs)

#%%
phon_output = TimeDistributed(Dense(Xp.shape[2], activation='sigmoid', name='...'), name='phon_output')
output_layer = phon_output(deep_outputs)


model = Model([orth_inputs, phon_inputs], output_layer)
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]
#%%

model.compile(optimizer=optimizer, loss=loss, metrics=metric)
model.summary()

t1 = time.time()
model.fit([Xo, Xp], Y, batch_size=100, epochs=20, validation_split=(1-.9))

learntime = round((time.time()-t1)/60, 2)
# %%
plot(model, to_file='merge.png')
#%%
# inspect a prediction
def pronounce(i, model, Xo, Xp, Y, labels=None, reps=None):
    print('word to predict:', labels[i])
    out = model.predict([reshape(Xo[i]), reshape(Xp[i])])
    print('Predicted:', decode(out, reps))
    print('True phon:', decode(Y[i], reps))
    

pronounce(5987, model, Xo, Xp_dummy, Y, labels=words, reps=phonreps)
# %%


model.fit([Xo, Xp], Y, batch_size=batch_size, epochs=5, validation_split=(1-train_proportion))

# %%
i = 203
print('word to predict:', words[i])
out = model.predict([reshape(Xo[i]), reshape(Xp[i])])
print('Predicted:', decode(out, phonreps))
print('True:', decode(Y[i], phonreps))
# %%
model.fit([Xo, Xp_dummy], Y, batch_size=batch_size, epochs=20, validation_split=(1-train_proportion))


# %%
i = 203
print('word to predict:', words[i])
out = model.predict([reshape(Xo[i]), reshape(Xp_dummy[i])])
print('Predicted:', decode(out, phonreps))
print('True:', decode(Y[i], phonreps))
# %%
