
#%%
from utilities import load_model, get_weights
import numpy as np
from keras.backend import clear_session as clear
import os

"""gather weights for all the taraban crossval runs"""



total_runs = 50

PATH = '../outputs/taraban_crossval/'
epochs = [9, 18, 27, 36, 45, 54, 63, 72]


for run_id in range(total_runs):
    run = str(run_id)
    for epoch in epochs:

        model = load_model(os.path.join(PATH, str(run), 'model-epoch{}.json'.format(epoch)), os.path.join(PATH, run, 'model-epoch{}-weights.h5'.format(epoch)), compile=False)
        orth_weights = get_weights(model, 'orth')
        phon_weights = get_weights(model, 'phon')
        output_weights = get_weights(model, 'output')

        for i in range(len(orth_weights)):

            print(i)
            np.savetxt(os.path.join(PATH, run, 'weights{}_orth_epoch{}.csv'.format(i, epoch)), orth_weights[i])
            np.savetxt(os.path.join(PATH, run, 'weights{}_phon_epoch{}.csv'.format(i, epoch)), phon_weights[i])
            if i < len(output_weights):
                np.savetxt(os.path.join(PATH, run, 'weights{}_output_epoch{}.csv'.format(i, epoch)), output_weights[i])
        clear()
# %%
