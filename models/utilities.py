#%%
from numpy.random import choice
import numpy as np
import json
import tensorflow as tf

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


def allocate(traindata, n, train=None, test=None, seed = 652, drop=True, keys=None):

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

    train : list
        Words (if any) to make sure are present in the training set after
        split. (Default is None)

    test : list
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
        s = [word for k, v in traindata.items() for word in v['wordlist'] if word not in test]
    else:
        s = [word for k, v in traindata.items() for word in v['wordlist'] if k in keys and word not in test]

    if type(n) == float:
        n = round(n*len(s))

    r = list(set(sample(s, n) + test))

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
        elif i == len(x)-1: # when you get to the last element
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
    return(words)
    
    if verbose:
        print(words)



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

