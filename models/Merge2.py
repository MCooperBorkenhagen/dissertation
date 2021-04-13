
#%%
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, TimeDistributed, Concatenate, Bidirectional
from utilities import changepad, loadreps, decode, reshape, addpad
import numpy as np
import time
import pandas as pd

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#%%
from tensorflow.keras.utils import plot_model as plot


#%%
#%% load
Xo = np.load('../inputs/orth-left.npy')

#%%

orthreps = loadreps('../inputs/raw/orthreps.json')
orthpadX = np.array(loadreps('../inputs/raw/orthreps.json', changepad=True, newpad=9)['_'])
orthpadY = np.array(orthreps['_'])

Xo = addpad(np.load('../inputs/orth-left.npy'), orthpadX)

Xp = np.load('../inputs/phon-for-eos-left.npy')

Yo = np.load('../inputs/orth-left.npy')
Yo = changepad(Yo, old=9, new=0)
Yo = addpad(Yo, orthpadY)

Yp = np.load('../inputs/phon-for-eos-left.npy')
Yp = changepad(Yp, old=9, new=0)
phonreps = loadreps('../inputs/phonreps-with-eos-only.json', changepad=True)

words = pd.read_csv('../inputs/encoder-decoder-words.csv', header=None)[0].tolist()

#%%



#%%
orthshape = Xo.shape
phonshape = Xp.shape


Xo_dummy = np.zeros(orthshape)
Xp_dummy = np.zeros(phonshape)

Xo_mask = np.full(orthshape, 9)
Xp_mask = np.full(phonshape, 9)



#%%
hidden = 300
optimizer='rmsprop'
loss="categorical_crossentropy"
transfer_function = 'sigmoid'
epochs = 30
batch_size = 100
train_proportion = .8


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
phon_outputs, phon_hidden, phon_cell = phon(phon_inputs_masked)
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

deep = Dense(200, activation='sigmoid', name='deep_layer')
deep_outputs = deep(merge_outputs)

#%%
phon_out = TimeDistributed(Dense(Xp.shape[2], activation='sigmoid', name='...'), name='phon_output')
orth_out = TimeDistributed(Dense(Xo.shape[2], activation='sigmoid', name='...'), name='orth_output')
phon_out = phon_out(deep_outputs)
orth_out = orth_out(deep_outputs)

model = Model([orth_inputs, phon_inputs], [orth_out, phon_out])
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]
#%%

model.compile(optimizer=optimizer, loss=loss, metrics=metric)
model.summary()

plot(model, to_file='merge2.png')
#%%
t1 = time.time()
model.fit([Xo, Xp], [Yo, Yp], batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))


learntime = round((time.time()-t1)/60, 2)
# %%
plot(model, to_file='merge.png')
#%%
# inspect a prediction
def pronounce(i, model, Xo, Xp, Yo, Yp, labels=None, phonreps=None, orthreps=None):
    print('word to predict:', labels[i])
    out = model.predict([reshape(Xo[i]), reshape(Xp[i])])
    return out[0]
    print('Predicted orth:', decode(out[0], orthreps))
    print('True orth:', decode(Yo[i], orthreps))
    print('#######')
    print('Predicted phon:', decode(out[1], phonreps))
    print('True phon:', decode(Yp[i], phonreps))
    


orthout = pronounce(0, model, Xo, Xp, Yo, Yp, labels=words, phonreps=phonreps, orthreps=orthreps)
tmp = np.around(orthout)
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
