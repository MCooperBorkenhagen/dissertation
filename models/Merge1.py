

#%%
from tensorflow.keras.utils import plot_model as plot
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed, Concatenate, RepeatVector
import numpy as np
import time

class Learner():

    def __init__(self, Xo, Xp, Y, orth_density=500, phon_density=500, output_density=500, modelname='Merge1', labels=None, op_names=True, train_proportion=.9, hidden=300, batch_size=100, epochs=20, transfer_state=False, transfer_function='sigmoid', optimizer='rmsprop', loss="categorical_crossentropy", accuracy='binary', seed=886, devices=True, memory_growth=True):

        """
        transfer_state : bool
            Indicate whether to transfer the orth state to the phonological lstm. (default is False)


        """

        np.random.seed(seed)
        if type(Xo) == str:
            Xo = np.load(Xo)
        if type(Xp) == str:
            Xp = np.load(Xp)
        if type(Y) == str:
            Y = np.load(Y)

        if devices:
            tf.debugging.set_log_device_placement(True)
        else:
            pass

        if memory_growth:
            devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(devices[0], enable=True)


        
        self.labels = labels
        self.Xo = Xo
        self.Xp = Xp
        self.Y = Y


        # set as attrbutes a number of important input parameters to init:
        self.hidden_units = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.loss_function = loss
        self.seed = seed



        # learner:
        if op_names:
            input1_name = 'orth_input'
            input2_name = 'phon_input'
            output_name = 'phon_output'
        else:
            input1_name = 'input_1'
            input2_name = 'input_2'
            output_name = 'output'


        orth_inputs = Input(shape=(None, Xo.shape[2]), name='orth_input')
        orth_inputs_masked = Masking(mask_value=9, name='orth_mask')(orth_inputs)
        orth = LSTM(hidden, return_sequences=True, return_state=True, name = 'orthographic_lstm') # set return_sequences to True if no RepeatVector
        orth_outputs, orth_hidden, orth_cell = orth(orth_inputs_masked)
        orth_state = [orth_hidden, orth_cell]

        orth_dense = Dense(orth_density, activation=transfer_function, name='orth_dense')
        orth_decomposed = orth_dense(orth_outputs)

        phon_inputs = Input(shape=(None, Xp.shape[2]), name='phon_input')
        phon_inputs_masked = Masking(mask_value=9, name='phon_mask')(phon_inputs)
        phon = LSTM(hidden, return_sequences=True, return_state=True, name='phonological_lstm')
        if transfer_state:
            phon_outputs, phon_hidden, phon_cell = phon(phon_inputs_masked, initial_state=orth_state)
        else:
            phon_outputs, phon_hidden, phon_cell = phon(phon_inputs_masked)
        phon_state = [phon_hidden, phon_cell]

        phon_dense = Dense(phon_density, activation=transfer_function, name='phon_dense')
        phon_decomposed = phon_dense(phon_outputs)

        # merge outputs of lstms
        merge = Concatenate(name='merge')
        merge_outputs = merge([orth_decomposed, phon_decomposed])

        # merged layer (have toyed with)
        deep = Dense(250, activation='sigmoid', name='deep_layer')
        deep_outputs = deep(merge_outputs)

        repeater = RepeatVector(Y.shape[1])
        output_input = repeater(deep_outputs)

        # output layer
        phon_output = TimeDistributed(Dense(Xp.shape[2], activation='sigmoid', name='phon_output'))
        output_layer = phon_output(output_input)
        model = Model([orth_inputs, phon_inputs], output_layer)




        self.orth_lstm = orth
        self.phon_lstm = phon
        self.orth_dense = orth_dense
        self.phon_dense = phon_dense
        #self.deep_dense = deep

        # train
        if accuracy == 'binary':
            metric = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5)]
        else:
            metric = accuracy

        model.compile(optimizer=optimizer, loss=loss, metrics=metric)
        

        self.model = model
        self.summary = model.summary()

        t1 = time.time()
        cb = model.fit([Xo, Xp], Y, batch_size=batch_size, epochs=epochs, validation_split=(1-train_proportion))
        cb.history['learntime'] = round((time.time()-t1)/60, 2)
        self.runtime = time.ctime()
        self.history = cb.history
        self.model = model
        self.plot = plot(self.model, to_file=modelname+'.png')



    def evaluate(self, X, Y):        
        return(self.model.evaluate(X, Y))



if __name__ == "__main__":
    Learner()