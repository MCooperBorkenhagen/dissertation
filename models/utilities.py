#%%
from numpy.random import choice
import numpy as np
import json


#%%



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
    