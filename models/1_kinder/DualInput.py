
#%%
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, TimeDistributed, Concatenate
from utilities import changepad, loadreps, decode, reshape 
import numpy as np
import time
import pandas as pd



Xo = np.load('../../inputs/orth-left.npy')
Xp = np.load('../../inputs/phon-left.npy')

Y = np.load('../../inputs/phon-left.npy')
Y = changepad(Y, old=9, new=0)
words = pd.read_csv('../../inputs/encoder-decoder-words.csv', header=None)[0].tolist()
phonreps = loadreps('../../inputs/phonreps.json', changepad=True)


hidden = 100
optimizer='rmsprop'
loss="categorical_crossentropy"
epochs = 2
batch_size = 100
train_proportion = .9

orth_inputs = Input(shape=(None, Xo.shape[2]), name='orth_input')
orth_inputs_masked = Masking(mask_value=9)(orth_inputs)
orth = LSTM(hidden, return_sequences=False, name = 'orthographic_lstm') # set return_sequences to True if no RepeatVector
orth_outputs = orth(orth_inputs_masked)



phon_inputs = Input(shape=(None, Xp.shape[2]), name='phon_input')
phon_inputs_masked = Masking(mask_value=9)(phon_inputs)
phon = LSTM(hidden, return_sequences=False, name='phonological_lstm')
phon_outputs = phon(phon_inputs_masked)

merge = Concatenate()
merge_output = merge([phon_outputs, orth_outputs])

deep = Dense(hidden, activation='sigmoid')
deep_outputs = deep(merge_output)
phon_dense = TimeDistributed(Dense(Y.shape[2], activation='sigmoid'), name='phon_output')

phon_outputs = phon_dense(phon_outputs)
model = Model(orth_inputs, phon_outputs)
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]


model.compile(optimizer=optimizer, loss=loss, metrics=metric)
model.summary()

t1 = time.time()
model.fit([Xo, Xp], Y, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))


learntime = round((time.time()-t1)/60, 2)
# %%
# inspect a prediction
i = 203
print('word to predict:', words[i])
out = model.predict(reshape(X[i]))
print('Predicted:', decode(out, phonreps))
print('True:', decode(Y[i], phonreps))
# %%
