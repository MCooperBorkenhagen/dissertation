
#%%
import tensorflow as tf
from keras.models import model_from_json
from utilities import *
#model = tf.keras.models.load_model('../outputs/taraban/taraban-model-epoch18')

test = load('data/taraban-test.traindata')
train = load('data/taraban-train.traindata')

json_file = open('../outputs/taraban/model-epoch18.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('../outputs/taraban/model-epoch18-weights.h5')
loaded_model.compile(loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")])
# %%
xo = get(train, 'them', data='orth')
xp = get(train, 'them', data='phonSOS')


#%%
y_hat = loaded_model.predict([reshape(xo), reshape(xp)])
# %%
def load_model(architecture, weights, metrics=None, loss='binary_crossentropy', compile=True):

    """Load a model from its architecture (json) and weights (h5)

    Parameters
    ----------
    architecture : str
        The object corresponding to this path should be a json
        file containing the architecture of the model you wish
        to load.

    weights : str
        This object should be an object with extension *.h5
        which contains saved weights from a pretrained model.

    metrics : list
        A list of metrics should be used to calculate accuracy
        in the returned model, if compiled.

    loss : str
        The loss function passed to compile().
        (Default is binary crossentropy)

    compile : bool
        Specify whether to compile the model (True) or not
        (False). (Default is True)


    Returns
    -------
    keras engine
        The return object is an object of type 
        keras.engine.functional.Functional
    
    """
    from keras.models import model_from_json

    if metrics==None:
        import tensorflow as tf
        metrics =  metrics=[tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]

    j = open(architecture, 'r')
    l = j.read()

    model = model_from_json(l)
    model.load_weights(weights)
    if compile:
        model.compile(loss=loss, metrics=metrics)

    return model

# %%


m = load_model('../outputs/taraban/model-epoch18.json', '../outputs/taraban/model-epoch18-weights.h5')