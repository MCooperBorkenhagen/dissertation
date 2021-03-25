
#%%
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
from tensorflow.keras.utils import plot_model as plot

input_layer = Input(shape=(3, 1), name='layer_1')
lstm_layer, h, c = LSTM(1, return_sequences=True, return_state=True, name = 'layer_2')(input_layer)
learner = Model(inputs=input_layer, outputs=[lstm_layer, h, c])
#%%
X = array([.8, 6, .1]).reshape((1,3,1))
output, hidden_state, cell_state = learner.predict(X)
# %%
plot(learner)
# %%
# figuring out activations:
layer_index = 1
test_acts = keras.Model(inputs=learner.inputs, outputs=[layer.output for layer in learner.layers])

acts_all = test_acts(X)
acts = array(acts_all[layer_index])

# %%
