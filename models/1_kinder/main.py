
# %%
from Learner import Learner
import numpy as np
import pandas as pd
import tensorflow as tf

#tf.debugging.set_log_device_placement(monitor) 

x = np.load("../../inputs/orth_pad_right.npy")
y = np.load("../../inputs/phon_pad_right.npy")
labels = pd.read_csv("../../inputs/syllabics.csv", sep = ",")
words = labels.orth.tolist()
ml = Learner(x, y, labels=words, epochs=5)

# %%
model = ml.model
# %%

# %%
