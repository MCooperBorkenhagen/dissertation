
import numpy as np
import json


def divide(x, y, frac):
    """Divide data into training and testing sets

    Parameters
    ----------
    x : arr
        A numpy 3D array of binary representations; the input feature vectors
    
    y : arr
        A numpy 3D array of binary representations; the output feature vectors

    frac : float
        A fraction of x and y to be partitioned as training and test data.

    Returns
    -------
        numpy 3D array of booleans, the train input feature vectors
        numpy 3D array of booleans, the train output feature vectors
        numpy 3D array of booleans, the test input feature vectors
        numpy 3D array of booleans, the test output feature vectors
    """
    size = len(x)
    assert size == len(y), "number of input and output feature vectors do not match"

    train_size = int(size * frac)
    indices = np.arange(size)
    np.random.shuffle(indices)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return(x_train, y_train, x_test, y_test)



def changepad(X, old=None, new=None):
    X[X==old] = new
    return(X)

def test_acts(inputs, learner, layer='all'):
    import keras
    test = keras.Model(inputs=learner.model.input, outputs=[layer.output for layer in learner.model.layers])
    acts = test(inputs)
    if layer == 'all':
        return(acts)
    else:
        return(np.array(acts[layer]))

def key(dict, value):
    for k, v in dict.items():
        if value == v:
            return(k)



def decode(x, reps, join=True):
    d = [key(reps, rep) for rep in x]
    if join:
        return(''.join(d))
    else:
        return(d)
    
def reshape(a):
    shape = (1, a.shape[0], a.shape[1])
    return(np.reshape(a, shape))



def all_equal(X, Y):
    return((Y == Y).all())


def cor_acts(X, Y, method='pearson'):
    from scipy.spatial.distance import pdist as dist
    assert X.shape == Y.shape, 'X and Y have different shapes and cannot be correlated'
    d1 = X.shape[0] # we could take dims from either ml acts or mr acts - should not make a difference
    d2 = X.shape[1]*X.shape[2]

    dX = dist(X.reshape((d1, d2)))
    dY = dist(Y.reshape((d1, d2)))

    if method == 'pearson':
        return(np.corrcoef(dX, dY))
    elif method == 'spearman':
        from scipy.stats import spearmanr as cor
        return(cor(dX, dY))





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



def dists(a, reps, ties=True):
    d = {np.linalg.norm(a-np.array(v)):k for k, v in reps.items()}
    min_ = min(d.keys())

    if ties:
        u = [k for k, v in d.items() if k == min_]
        assert len(u) == 1, 'More than one minumum value for pairwise distances. Ties present.'

    return(d[min_])


def decode(a, reps, round=True):
    if a.ndim == 3:
        a = a[0]
    a = np.around(a)
    word = []
    for segment in a:
        word.append(dists(segment, reps))
    return(word)



def sample(n, Xo, Xp, Xy, labels = None, seed = 123):
    import random
    random.seed(seed)
    s = random.sample(range(Xo.shape[0]), n)
    if labels is not None:
        labels_sampled = [labels[i] for i, label in enumerate(labels)]
        return Xo[s], Xp[s], Xy[s], labels_sampled
    else:
        return Xo[s], Xp[s], Xy[s]


def pronounce2(i, model, xo, xp, yp, labels=None, reps=None):
    """Pronounce a word from two inputs.
    """

    print('word to predict:', labels[i])
    out = model.predict([reshape(xo[i]), reshape(xp[i])])
    print('Predicted:', decode(out, reps))
    print('True phon:', decode(yp[i], reps))


def pronounce1(i, model, xo, yp, labels=None, reps=None):
    """Pronounce a word from one input.
    """
    print('word to predict:', labels[i])
    out = model.predict(reshape(xo[i]))
    print('Predicted:', decode(out, reps))
    print('True phon:', decode(yp[i], reps))


def generalize(xo, xp, model, reps, label=None):
    """
    """

    print('word to predict:', label)
    out = model.predict([reshape(xo), reshape(xp)])
    print('Predicted:', decode(out, reps))



def addpad(a, pad):
    from numpy import reshape as rs
    from numpy import append as ap

    if a.ndim == 2:
        axis0 = a.shape[0]+1
        axis1 = a.shape[1] 
        return(rs(ap(a, pad), (axis0, axis1)))

    if a.ndim == 3:
        axis0 = a.shape[0]
        axis1 = a.shape[1]+1
        axis2 = a.shape[2]
        n = np.empty((axis0, axis1, axis2), dtype=a.dtype)
        for i, a in enumerate(a):
            n[i] = addpad(a, pad)
        return(n)
