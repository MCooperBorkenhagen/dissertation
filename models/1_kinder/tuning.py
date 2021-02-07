
# %%

import numpy as np
import keras
import tensorflow as tf
from utilities import divide
from keras.models import Sequential
from keras import layers
import time
import pandas as pd
import json
import csv

# devices:
tf.debugging.set_log_device_placement(True)





# load the data
x = np.load("../../inputs/orth_pad_right.npy")
y = np.load("../../inputs/phon_pad_right.npy")
labels = pd.read_csv("../../inputs/syllabics.csv", sep = ",")

# config
with open('./params.json', 'r') as f:
    cfg = json.load(f)

np.random.seed(cfg['seed'])


train_frac = 0.9
x_train, y_train, x_test, y_test = divide(x, y, train_frac)

EPOCHS = [100]
BATCHES = [100, 150, 200, 250, 300, 350, 400]
LAYERS = [1]
UNITS = [600, 700, 800, 900, 1000]

d = {}
run = 1


runs = len(EPOCHS)*len(BATCHES)*len(LAYERS)*len(UNITS)
# %%
# build the model

with open('tuning.csv', 'w') as f:
    f.write(','.join(['run_id', 'epoch', 'loss_train', 'loss_test', 'acc_train', 'acc_test', 'num_layers', 'num_epochs', 'batch_size', 'hidden_size', 'learntime']) + '\n')

    for LAYER in LAYERS:
        for EPOCH in EPOCHS:
                for BATCH in BATCHES:
                    for UNIT in UNITS:
                        t1 = time.time()
                        SEQ = layers.LSTM
                        model = Sequential()

                        model.add(SEQ(UNIT, input_shape=x_train[0].shape, name = 'orth'))
                        model.add(layers.RepeatVector(len(y[0])))
                        for _ in range(LAYER):
                            nm = 'hidden' + str(_)
                            model.add(SEQ(UNIT, return_sequences=True, name = nm))
                        model.add(layers.TimeDistributed(layers.Dense(len(y[0][0]), activation="sigmoid"), name = 'phon'))
                        model.compile(loss=cfg['loss'], optimizer="adam",
                                        metrics=[tf.keras.metrics.BinaryAccuracy(name = cfg['accuracy'], dtype = None, threshold=0.5)])
                        model.summary()

                        # train the network
                        cb = model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCH, validation_split=cfg['validation_split'])
                        h = cb.history

                        t2 = time.time()
                        """
                        # gather data for d:
                        h['layers'] = LAYER
                        h['epochs'] = EPOCH
                        h['batch_size'] = BATCH
                        h['hidden_size'] = UNIT
                        h['learntime'] = round((t2-t1)/60, 2)
                        d[run] = h"""
                        for i in range(EPOCH):
                            f.write("{run_id:d}, {epoch:d}, {loss_train:.8f},{loss_test:.8f},{acc_train:.8f},{acc_test:.8f},{num_layers:d},{num_epochs:d},{batch_size},{hidden_size:d},{learntime:.8f}\n".format(
                                run_id = run,
                                epoch = i,
                                loss_train = cb.history['loss'][i],
                                loss_test = cb.history['val_loss'][i],
                                acc_train = cb.history['binary_accuracy'][i],
                                acc_test = cb.history['val_binary_accuracy'][i],
                                num_layers = LAYER,
                                num_epochs = EPOCH,
                                batch_size = BATCH,
                                hidden_size = UNIT,
                                learntime = round((t2-t1)/60, 2)))

                        
                        print('#####################################')
                        print('#####################################')
                        print('#####################################')
                        print('#####################################')
                        print('Run', run, 'of', runs, 'done')
                        print('Run', run, 'of', runs, 'done')
                        print('Run', run, 'of', runs, 'done')
                        print('Run', run, 'of', runs, 'done')
                        print('#####################################')
                        print('#####################################')
                        print('#####################################')
                        print('#####################################')
                        run += 1

f.close()
