
from utilities import split

#%% check that the split method is working properly
# %%
a, b = split(data, .04)
# %%
assert a.keys() == b.keys() == data.keys()
for k, v in data.items():
    for i, word in enumerate(v['wordlist']):
        if word in a[k]['wordlist']:
            j = a[k]['wordlist'].index(word)
            assert (a[k]['phonSOS'][j] == data[k]['phonSOS'][i]).all()
            assert (a[k]['phonEOS'][j] == data[k]['phonEOS'][i]).all()
            assert (a[k]['orth'][j] == data[k]['orth'][i]).all()
            assert (a[k]['frequency'][j] == data[k]['frequency'][i]).all()
        elif word in b[k]['wordlist']:
            h = b[k]['wordlist'].index(word)
            assert (b[k]['phonSOS'][h] == data[k]['phonSOS'][i]).all()
            assert (b[k]['phonEOS'][h] == data[k]['phonEOS'][i]).all()
            assert (b[k]['orth'][h] == data[k]['orth'][i]).all()
            assert (b[k]['frequency'][h] == data[k]['frequency'][i]).all()




def split(traindata, n, seed = 652):
    from random import sample, seed
    
    seed(seed)

    s = [word for k, v in data.items() for word in v['wordlist']]

    if type(n) == float:
        n = round(n*len(s))

    r = sample(s, n)

    holdout = {}
    train = {}
    for k, v in data.items():
        subset = []
        primary = []
        wordlist = []
        trainwords = []
        for i, word in enumerate(v['wordlist']):
            if word in r:
                subset.append(i)
                wordlist.append(word)
            else:
                primary.append(i)
                trainwords.append(word)
        train[k] = {'phonSOS':v['phonSOS'][primary], 'phonEOS':v['phonEOS'][primary], 'orth':v['orth'][primary], 'wordlist':trainwords, 'frequency':v['frequency'][primary]}
        holdout[k] = {'phonSOS':v['phonSOS'][subset], 'phonEOS':v['phonEOS'][subset], 'orth':v['orth'][subset], 'wordlist':wordlist, 'frequency':v['frequency'][subset]}
        
    return holdout, train

