
#%%
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, TimeDistributed
from utilities import changepad, loadreps, decode, reshape 
import numpy as np
import time
import pandas as pd


#X = np.load('../../inputs/orth-left.npy')

X = np.load('../../inputs/phon-left.npy')

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

orth_inputs = Input(shape=(None, X.shape[2]), name='orth_input')
orth_inputs_masked = Masking(mask_value=9)(orth_inputs)
orth = LSTM(hidden, return_state=True, return_sequences=True, name = 'orthographic_lstm') # set return_sequences to True if no RepeatVector
orth_outputs, state_h, state_c = orth(orth_inputs_masked)
orth_states = [state_h, state_c]
deep = LSTM(hidden, return_state=True, return_sequences=True, name = 'deep_lstm') # set return_sequences to True if no RepeatVector
deep_outputs, state_h, state_c = deep(orth_outputs)

phon_lstm = LSTM(hidden, return_sequences=True, return_state=True)
phon_outputs, _, _ = phon_lstm(deep_outputs)
phon_dense = TimeDistributed(Dense(Y.shape[2], activation='sigmoid'), name='phon_output')

phon_outputs = phon_dense(phon_outputs)
model = Model(orth_inputs, phon_outputs)
metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]


model.compile(optimizer=optimizer, loss=loss, metrics=metric)
model.summary()

t1 = time.time()
model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))


learntime = round((time.time()-t1)/60, 2)
# %%
# inspect a prediction
i = 203
print('word to predict:', words[i])
out = model.predict(reshape(X[i]))
print('Predicted:', decode(out, phonreps))
print('True:', decode(Y[i], phonreps))
# %%
