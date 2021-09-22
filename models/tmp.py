
#%%
import tensorflow as tf
import keras


metrics = [tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold=0.5), tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")]

MeanSquaredError = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

reduction = 'auto'
model = tf.keras.models.load_model('../outputs/taraban/taraban-model-epoch54', custom_objects={'reduction': reduction})

# %%
