
#%%
import numpy as np
from utilities import *


m = load_model('../outputs/taraban/model-epoch18.json', '../outputs/taraban/model-epoch18-weights.h5', compile=False)


#%%
# orth weights from taraban model
np.savetxt('data/taraban_orth_weights0.csv', m.layers[3].get_weights()[0])
np.savetxt('data/taraban_orth_weights1.csv', m.layers[3].get_weights()[1])
np.savetxt('data/taraban_orth_weights2.csv', m.layers[3].get_weights()[2])


# phon weights from taraban model
np.savetxt('data/taraban_phon_weights0.csv', m.layers[4].get_weights()[0])
np.savetxt('data/taraban_phon_weights1.csv', m.layers[4].get_weights()[1])
np.savetxt('data/taraban_phon_weights2.csv', m.layers[4].get_weights()[2])

# dense output layer weights
np.savetxt('data/taraban_output_weights0.csv', m.layers[5].get_weights()[0])
np.savetxt('data/taraban_output_weights1.csv', m.layers[5].get_weights()[1])






# %%



