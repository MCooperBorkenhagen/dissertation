
#%%
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
from tensorflow.keras.utils import plot_model as plot

input_layer = Input(shape=(2, 1), name='layer_1')
lstm_layer, h, c = LSTM(1, return_sequences=False, return_state=False, name = 'layer_2')(input_layer)
layers = Model(inputs=input_layer, outputs=[lstm_layer, h, c])
#%%
X = array([.8, 6]).reshape((1,2,1))
output, hidden_state, cell_state = layers.predict(X)
# %%
plot(layers)
# %%
