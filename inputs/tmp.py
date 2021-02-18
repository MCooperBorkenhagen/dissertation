# %%
from Reps import Reps
import nltk
from copy import deepcopy as cp
cmudict = nltk.corpus.cmudict.dict()

pool = ['the', 'and', 'but', 'this', 'a', 'of', 'in', 'these']
r = Reps(pool, outliers=['of'], onehot=True, terminals=True)


# %%
x = r.phonformsSOS_array
y = r.pool

#%%

def remove_all(x, element):
    return(list(filter(lambda e: e != element, x)))


def key(dict, value):
    for k, v in dict.items():
        if value == v:
            return(k)


def reconstruct(x, y, repdict=None, join=True, axis=0):

    """Reconstruct a string representation of a pattern from binary sequence.

    Parameters
    ----------
    x : numpy array
        An array containing binary representations from which the reconstruction
        will occur.
    
    y : list
        Each element of the list will be the string-based representation of each
        element in x. The structure of each element will be inferred from reps. For
        reps='phon', each element will be a list; for reps='orth', each element will
        be a string.

    reps : str
        Specify the dictionary containing binary representations for each element
        within examples in x. (Default is None)

    join :  bool
        Join the string representatation generated from reconstruction. This is
        necessary if the elements in r are orthographic wordforms.

    axis : int
        The axis of x over which iteration should occur. (default is 0)

    Returns
    -------
    bool
        A True value is provided if reconstructed x matches the representations
        in y. Else, a False value is returned.
    """

    def reconstruct_(example):
        return([key(repdict, e) for e in example.tolist()])

    r = []

    for ex in range(x.shape[0]):
        r.append(reconstruct_(x[ex]))

    r = [remove_all(e, '_') for e in r]
    
    if not join:
        return(r == y)
    elif join:
        for e in r:
        return([''.join(e) for e in r] == y)
