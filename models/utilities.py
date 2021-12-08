#%%
from numpy.random import choice
import numpy as np
import json
import tensorflow as tf
import os

#%%

def drop_empty_values(traindata):
    empty = []
    for k, v in traindata.items():
        if len(v['wordlist']) == 0:
            empty.append(k)

    for k in empty:
        traindata.pop(k)

    return traindata


def loadreps(PATH, changepad=True, newpad=0):
    with open(PATH, 'r') as p:
        phonreps = json.load(p)
    if changepad:
        pad = [newpad for e in phonreps['_']]
        phonreps['_'] = pad
    return(phonreps)


def reshape(a):
    shape = (1, a.shape[0], a.shape[1])
    return(np.reshape(a, shape))


def load(PATH):
    import pickle
    f = open(PATH, 'rb')
    return(pickle.load(f))


def printspace(lines, symbol='#', repeat=25):
    for i in range(lines):
        print(repeat*symbol, '\n')


def L2(a, v):
    return(np.linalg.norm(a-np.array(v)))


def choose(x, n, probabilities):
    assert len(x) == len(probabilities), 'Your values to sample and associated probabilities have different lengths. Respecify x or probabilities'
    return(choice(x, n, p=probabilities, replace=False))

def scale(x, K):
    p = K*np.log(x)
    return p
    

def key(dict, value):
    for k, v in dict.items():
        if value == v:
            return(k)

def mean(x):
    return(sum(x)/len(x))


def has_numeric(x):
    return any(c.isdigit() for c in x)

def get_vowels(x, index=True):
    if index:
        return [i for i, v in enumerate(x) if has_numeric(v)]
    else:
        return [v for v in x if has_numeric(v)]


def syllables(phonform):

    """Count the number of syllables in the phonological wordform.

    Parameters
    ----------
    phonform : list
        A phonological form expressed in a list with phones
        defined as two-letter ARPAbet phonemic segments
        (i.e., as in cmudict). Each element in the list
        is a single phonemic segment.

    Returns
    -------
    int
        The number of syllables in phonform.

    """
    phonform = clean(phonform, orthography=False)
    ns = sum([numeral_detect(e) for e in phonform])
    if ns == 0:
        raise ValueError('Your phonological form has no vowels. Check phonform: {}'.format(phonform))
    else:
        return(ns)



def split(traindata, n, seed = 652, drop=True, keys=None):
    from random import sample, seed
    
    seed(seed)

    if keys is None:
        s = [word for k, v in traindata.items() for word in v['wordlist']]
    else:
        s = [word for k, v in traindata.items() for word in v['wordlist'] if k in keys]

    if type(n) == float:
        n = round(n*len(s))

    r = sample(s, n)

    holdout = {}
    train = {}
    for k, v in traindata.items():
        iho = []
        itr = []
        testwords = []
        trainwords = []
        for i, word in enumerate(v['wordlist']):
            if word in r:
                iho.append(i)
                testwords.append(word)
            else:
                itr.append(i)
                trainwords.append(word)
        train[k] = {'phonSOS':v['phonSOS'][itr], 'phonEOS':v['phonEOS'][itr], 'orth':v['orth'][itr], 'wordlist':trainwords, 'frequency':v['frequency'][itr]}
        holdout[k] = {'phonSOS':v['phonSOS'][iho], 'phonEOS':v['phonEOS'][iho], 'orth':v['orth'][iho], 'wordlist':testwords, 'frequency':v['frequency'][iho]}
    
    if drop:
        holdout = drop_empty_values(holdout)
        train = drop_empty_values(train)

    return holdout, train



def split_but_skip(traindata, n, skip=None, seed = 652, drop=True, keys=None):

    """This is a version of split() but where the user can specify words to skip when
    splitting up a traindata object into test and holdout examples. If skip is set to 
    None, this method is identical to split(). This method will be deprecated in the 
    future by merge with split().

    Parameters
    ----------
    traindata : dict
        A traindata dictionary to split.
    
    n : int or float
        The number of holdout items to select, or if float the proportion.

    skip : list
        Words (if any) to make sure are present in the training set after
        split. (Default is None)

    seed : int
        Random seed to be passed to random.seed(). (Default is 652)

    drop : bool
        Whether to drop empty values (keys) from the return objects.
        This is equivalent to making sure that no empty key-value
        pairs are present after the split. (Default is True)

    keys : list
        The keys (phoneme lengths) if any to include (exclusively)
        in the return splits. Keys not present in this list but
        present in traindata will be skipped when splitting the
        train and test data. (Default is None)

    Returns
    -------
    dict
        Two dictionaries are returned. The first is the holdout set and the second
        is the training set having been split from the input dict.
    """

    from random import sample, seed
    
    seed(seed)

    if keys is None:
        s = [word for k, v in traindata.items() for word in v['wordlist'] if word not in skip]
    else:
        s = [word for k, v in traindata.items() for word in v['wordlist'] if k in keys and word not in skip]

    if type(n) == float:
        n = round(n*len(s))

    r = sample(s, n)

    holdout = {}
    train = {}
    for k, v in traindata.items():
        iho = []
        itr = []
        testwords = []
        trainwords = []
        for i, word in enumerate(v['wordlist']):
            if word in r:
                iho.append(i)
                testwords.append(word)
            else:
                itr.append(i)
                trainwords.append(word)
        train[k] = {'phonSOS':v['phonSOS'][itr], 'phonEOS':v['phonEOS'][itr], 'orth':v['orth'][itr], 'wordlist':trainwords, 'frequency':v['frequency'][itr]}
        holdout[k] = {'phonSOS':v['phonSOS'][iho], 'phonEOS':v['phonEOS'][iho], 'orth':v['orth'][iho], 'wordlist':testwords, 'frequency':v['frequency'][iho]}
    
    if drop:
        holdout = drop_empty_values(holdout)
        train = drop_empty_values(train)

    return holdout, train


def allocate(traindata, n, for_train=None, for_test=None, seed = 652, drop=True, keys=None, verbose=True):

    """This is a version of split() and split_but_skip() but where the user can specify 
    words to allocate specifically to the train and test sets when splitting up a 
    traindata object into test and holdout examples. If skip is set to None, this 
    method is identical to split(). This method will be deprecated in the future by 
    merge with split() and split_but_skip().

    Parameters
    ----------
    traindata : dict
        A traindata dictionary to split.
    
    n : int or float
        The number of holdout items to select, or if float the proportion.

    for_train : list
        Words (if any) to make sure are present in the training set after
        split. (Default is None)

    for_test : list
        Words (if any) to make sure are present in the test set after the
        split. These words will be added to the test pool and be above and
        beyond the number of items specified in the n argument. (Default is None)

    seed : int
        Random seed to be passed to random.seed(). (Default is 652)

    drop : bool
        Whether to drop empty values (keys) from the return objects.
        This is equivalent to making sure that no empty key-value
        pairs are present after the split. (Default is True)

    keys : list
        The keys (phoneme lengths) if any to include (exclusively)
        in the holdout split. (Default is None)

    verbose : bool
        If True the words that are excluded from the return objects
        are printed for reference. (Default is True)

    Returns
    -------
    dict
        Two dictionaries are returned. The first is the holdout set and the second
        is the training set having been split from the input dict (traindata).
    """

    from random import sample, seed
    
    seed(seed)

    # s is the eligible selection pool of words that might be sampled for holdout
    if keys is None:
        s = [word for v in traindata.values() for word in v['wordlist'] if word not in for_test+for_train]
    else:
        s = [word for k, v in traindata.items() for word in v['wordlist'] if k in keys and word not in for_test+for_train]

    if type(n) == float:
        n = round(n*len(s))

    # r is the set of sampled items that will be held out (plus items specified in for_test)
    r = list(set(sample(s, n) + for_test))

    holdout = {}
    train = {}

    for k, v in traindata.items():
        iho = []
        itr = []
        testwords = []
        trainwords = []
        for i, word in enumerate(v['wordlist']):
            if word in r:
                iho.append(i)
                testwords.append(word)
            else:
                itr.append(i)
                trainwords.append(word)
        train[k] = {'phonSOS':v['phonSOS'][itr], 'phonEOS':v['phonEOS'][itr], 'orth':v['orth'][itr], 'wordlist':trainwords, 'frequency':v['frequency'][itr]}
        holdout[k] = {'phonSOS':v['phonSOS'][iho], 'phonEOS':v['phonEOS'][iho], 'orth':v['orth'][iho], 'wordlist':testwords, 'frequency':v['frequency'][iho]}
    
    if drop:
        holdout = drop_empty_values(holdout)
        train = drop_empty_values(train)

    if verbose:
        inwords = get_words(traindata, verbose=False)
        testwords = get_words(holdout, verbose=False)
        trainwords = get_words(train, verbose=False)
        missing = []
        for word in inwords:
            if word not in testwords+trainwords:
                missing.append(word)
        if len(missing) == 0:
            print('All words in traindata are present in train or test sets returned')
        elif len(missing) > 0:
            print('The following words are missing from your return sets:')
            print(missing)

    return holdout, train



def subset(traindata, words):

    d = {}
    for k, v in traindata.items():
        I = []
        wordlist = []
        for i, word in enumerate(v['wordlist']):          
            if word in words:
                I.append(i)
                wordlist.append(word)
        d[k] = {'phonSOS':v['phonSOS'][I], 'phonEOS':v['phonEOS'][I], 'orth':v['orth'][I], 'wordlist':wordlist, 'frequency':v['frequency'][I]}       
    return d


def collapse(x, delimiter='-'):

    """Collapse elements of x into a pretty string

    Parameters
    ----------
    x : list
        A list of strings to be collapsed.

    delimiter : str
        A delimiter of your choice.

    Returns
    -------
    str
        Each element of x collapsed into a string,
        and delimited with the delimiter.

    """

    s = ''

    for i in range(len(x)):
        if i < len(x)-1:
            s = s + str(x[i]) + delimiter
        else:
            s = s + str(x[i])
    return(s)


def flatten(x, newline=True, unlist=True, delimiter=','):
    y = ''
    for i, e in enumerate(x):
        if i < len(x)-1:
            if unlist:
                if type(e) == list:
                    e = collapse(e)
            y = y + str(e) + delimiter
        elif i == len(x)-1: # when you  to the last element
            if unlist:
                if type(e) == list:
                    e = collapse(e)
                y = y + str(e)
    if newline:
        y = y + '\n'
    return y



def shelve(x, ty=str, delimiter=',', newline=True):
    y = delimiter.join(map(ty, x))
    if newline:
        return(y+'\n')
    else:
        return(y)


def flad(a, pads=0, pad=None):
    if pads == 0:
        return(a.flatten())
    else:
        return(np.append(a, pad*pads))


def get_words(traindata, verbose=True):
    words = [word for k, v in traindata.items() for word in v['wordlist']]

    if verbose:
        print(words)

    return(words)




def get(traindata, x, data='phonEOS'):
    return [v[data][i] for k, v in traindata.items() for i, word in enumerate(v['wordlist']) if word == x][0]




def memory_growth(grow=True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, grow)



def save(x, PATH):
    import pickle
    f = open(PATH, 'wb')
    pickle.dump(x, f)
    f.close()

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

def get_weights(model, layer, combine=True):

    layers_ = {'orth':3, 'phon':4, 'output':5}

    if layer == 'orth' or layer == 'phon':
        kernel = model.layers[layers_[layer]].get_weights()[0]
        recurrent_kernel = model.layers[layers_[layer]].get_weights()[1]
        bias = model.layers[layers_[layer]].get_weights()[2]
        print('Three weight objects returned: kernel, recurrent kernel, and bias')
        if combine:
            return [kernel, recurrent_kernel, bias]
        else:
            return kernel, recurrent_kernel, bias
    if layer == 'output':
        weights = model.layers[layers_[layer]].get_weights()[0]
        bias = model.layers[layers_[layer]].get_weights()[1]
        print('Two weight objects returned: the weights and the biases')
        if combine:
            return [weights, bias]
        else: 
            return weights, bias


def read_weights(PATH, run_id, epochs, layer, combine=True):

    if layer == 'orth' or layer == 'phon':
        kernel = np.genfromtxt(os.path.join(PATH, run_id, 'weights0_{}_epoch{}.csv'.format(layer, epochs)))
        recurrent_kernel = np.genfromtxt(os.path.join(PATH, run_id, 'weights1_{}_epoch{}.csv'.format(layer, epochs)))
        bias = np.genfromtxt(os.path.join(PATH, run_id, 'weights2_{}_epoch{}.csv'.format(layer, epochs)))
        if combine:
            return [kernel, recurrent_kernel, bias]
        else:
            return kernel, recurrent_kernel, bias

    elif layer == 'output':
        # treated as having only two weight matrices
        weights = np.genfromtxt(os.path.join(PATH, run_id, 'weights0_{}_epoch{}.csv'.format(layer, epochs)))
        biases = np.genfromtxt(os.path.join(PATH, run_id, 'weights1_{}_epoch{}.csv'.format(layer, epochs)))
        if combine:
            return [weights, biases]
        else:
            return weights, biases



def test_model(model, traindata, testdata=None, null_phon=False, outpath='.', modelname='--', id='--'):

    # set up file for item data if evaluate set to True
    itemdata = open(os.path.join(outpath, 'item-data'+ '-' + modelname + '-' + id + '.csv'), 'w')
    itemdata.write("cycle, word, train_test, phonlength, binary_acc, mse, loss\n")


    for length in traindata.keys():
        Xo = traindata[length]['orth']
        Xp = traindata[length]['phonSOS']
        Y = traindata[length]['phonEOS']
        if null_phon:
            Xp[:, 1:, :] = 0

        for i, word in enumerate(traindata[length]['wordlist']):
            loss, binary_acc, mse = model.evaluate([reshape(Xo[i]), reshape(Xp[i])], reshape(Y[i]), verbose=False)
            itemdata.write("{}, {}, {}, {}, {}, {}, {}\n".format(id, word, 'train', length, binary_acc, mse, loss))

    if testdata is not None:
        for length in testdata.keys():
            Xo = testdata[length]['orth']
            Xp = testdata[length]['phonSOS']
            Y = testdata[length]['phonEOS']

            if null_phon:
                Xp[:, 1:, :] = 0

            for i, word in enumerate(testdata[length]['wordlist']):
                loss, binary_acc, mse = model.evaluate([reshape(Xo[i]), reshape(Xp[i])], reshape(Y[i]), verbose=False)
                itemdata.write("{}, {}, {}, {}, {}, {}, {}\n".format(id, word, 'test', length, binary_acc, mse, loss))

    itemdata.close()