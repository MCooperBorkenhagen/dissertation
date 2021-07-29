#%%
from Learner import Learner
import numpy as np
import pandas as pd
import keras
from utilities import load, loadreps, reshape, choose, split, collapse, flatten, shelve, flad, scale


mono = load('../inputs/mono.traindata')
# here k is equal to the phonological length of the word + 1 (because of the terminal segment)
test, train = split(mono, .07)
assert test.keys() == train.keys(), 'Phonological lengths represented in test and train are not the same. Resample'




#%% phonreps and orthreps
phonreps = loadreps('../inputs/phonreps-with-terminals.json')
orthreps = loadreps('../inputs/raw/orthreps.json')



#%%
orth_features = train[4]['orth'].shape[2]
phon_features = train[4]['phonSOS'].shape[2]
#%% probabilities for sampling phonological lengths during fitcycle() for training
probs = [.3, .4, .2, .075, .025]
#%%
learner = Learner(orth_features, phon_features, phonreps=phonreps, orthreps=orthreps, traindata=train, modelname='monosyllabic', hidden=400, mask_phon=False, devices=False)


#%% using same function for sampling probabilities from Seidenberg & McClelland (1989)
frequencies = {word: v['frequency'][i] for k, v in mono.items() for i, word in enumerate(v['wordlist'])}
p = .93
maxf = max(frequencies.values())
K = p/np.log(maxf)


#%%
C = 2 # how many runs of fitcycle do you want to run. # might be too many to get variability on accuracy, especially with respect to frequency
for i in range(1, C+1):
    cycle_id = str(i)
    learner.fitcycle(batch_size=70, cycles=10, probs=probs, K=K, evaluate=True, evaluate_when=11, cycle_id=cycle_id) 

# %% save the train and test words for the future. We will also save its scaled frquency along with it
#%%
with open('train-test-items.csv', 'w') as tt:

    tt.write('word,freq_scaled,train-test\n')
    for word in learner.words:
        sf = str(scale(frequencies[word], K))
        tt.write(word+','+sf+','+'train'+'\n')
    
    for v in test.values():
        for word in v['wordlist']:
            tt.write(word+','+''+','+'test'+'\n')

tt.close()
#%%
learner.model.save('monosyllabic-model')
# if you want to load the previously saved model:
#m = keras.models.load_model('./base-model')

#%%
# assessments for the holdout items:
colnames = 'word,freq,phon_read,phonemes_right,phonemes_wrong,phonemes_proportion,phonemes_sum,phonemes_average,phoneme_dists,stress,wordwise_dist\n'

steps = len([word for data in test.values() for word in data['wordlist']])
step = 1


with open('posttest-holdout-words.csv', 'w') as ht:
    ht.write(colnames)

    for length, data in test.items():
        if len(data['wordlist']) > 0:
            for i, word in enumerate(data['wordlist']):
                if i in [0, 1]:
                    print('on word', step, 'of', steps, 'total words')
                    wd = learner.test(word, target=data['phonEOS'][i], return_phonform=True, returns='all', ties='identify')
                    ht.write(word+','+str(frequencies[word])+','+flatten(wd))
                    step += 1
ht.close()



# %% calculate and write item performance data at end of training for the training items
colnames = 'word,freq,phon_read,phonemes_right,phonemes_wrong,phonemes_proportion,phonemes_sum,phonemes_average,phoneme_dists,stress,wordwise_dist\n'
with open('posttest-trainwords.csv', 'w') as at:
    at.write(colnames)
    for word in learner.words:
        wd = learner.test(word, return_phonform=True, returns='all', ties='sample')
        at.write(word+','+str(frequencies[word])+','+flatten(wd))

at.close()








# calculate true phonological outputs for all training and test items
maxphon = max(train.keys())
assert maxphon == max(test.keys()), 'The phonological lengths in your train and test data are not identical. Check your train-test split'

with open('posttest-train-outputs.csv', 'w') as to:
    for word in learner.words:
        y = learner.read(word, returns='patterns', ties='sample')
        assert y is not None, 'No output for the word. Check your word'
        to.write(word+','+shelve(flad(y, pads=maxphon-y.shape[0], pad=phonreps['_'])))

to.close()
#%%
with open('posttest-test-outputs.csv', 'w') as te:
    for length, data in test.items():
        for word in data['wordlist']:
            y = learner.read(word, returns='patterns', ties='sample')
            assert y is not None, 'No output for the word. Check your word'
            te.write(word+','+shelve(flad(y, pads=maxphon-y.shape[0], pad=phonreps['_'])))

te.close()

# %%
