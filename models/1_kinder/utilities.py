
import numpy as np

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
    import numpy
    X[X==old] = new
    return(X)

def test_acts(inputs, learner, layer='all'):
    import keras
    import numpy
    test = keras.Model(inputs=learner.model.input, outputs=[layer.output for layer in learner.model.layers])
    acts = test(inputs)
    if layer == 'all':
        return(acts)
    else:
        return(numpy.array(acts[layer]))

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