#%%

from utilities import load, get

test = load('data/taraban-pilot-test.traindata')
train = load('data/taraban-pilot-train.traindata')


# write train and test words to file for pairing with pairwise distance measures later
trainwords = [word for traindata in train.values() for word in traindata['wordlist']]
testwords = [word for testdata in test.values() for word in testdata['wordlist']]

with open('../outputs/taraban-pilot/order.csv', 'w') as f:
    for i, word in enumerate(trainwords+testwords): # it has to be in this order to match the order of the distance matrices
        if word in trainwords:
            condition = 'train'
        elif word in testwords:
            condition = 'test'
        f.write('{},{},{}\n'.format(i, word, condition))
f.close()

# %%

traindata = load('../inputs/taraban-pilot/taraban-pilot.traindata')

with open('../outputs/taraban-pilot/frequency.csv', 'w') as f:
    for word in trainwords+testwords:
        freq = get(traindata, word, 'frequency')
        f.write('{},{}\n'.format(word, freq))
f.close()
# %%
