
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