

import numpy as np
import pandas as pd
import json


# %%
def getreps(PATH, terminals=False):
    """Binary phonological reps from CSV.

    Parameters
    ----------
    PATH : str
        Path to the csv containing phonological representations.
    terminals : bool
        Specify whether to add end-of-string and start-of-string
        features to reps (default is not/ False). If true
        the character "%" is used for eos, and "#" for sos.
        Note that if set to true a new key-value pair is created
        in return dict for each of these characters, and a feature
        for each is added to every value in the dictionary.

    Returns
    -------
    dict
        A dictionary of phonemes and their binary representations
    """
    df = pd.read_csv(PATH)
    df = df[df['phone'].notna()]
    df = df.rename(columns={'phone': 'phoneme'})
    df = df.set_index('phoneme')
    feature_cols = [column for column in df if column.startswith('#')]
    df = df[feature_cols]
    df.columns = df.columns.str[1:]
    dict = {}
    for index, row in df.iterrows():
        dict[index] = row.tolist()

    if terminals:
        for k, v in dict.items():
            dict[k].append(0)
        dict['#'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        dict['%'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    return(dict)




# utility methods
def numeral_detect(x):
    return(any(c.isdigit() for c in x))


    
def count_numerals(x):
    return(sum(numeral_detect(p) for p in x))


def key(dict, value):
    for k, v in dict.items():
        if value == v:
            return(k)



def remove_all(x, element):
    return(list(filter(lambda e: e != element, x)))

def reconstruct(x, y, reps='phon', axis=0):

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
        Phonological and orthographic wordforms require different binary 
        representations for their reconstruction. Specifying 'phon' here will
        reference the dictionary containing phonological representations (ie, phonreps),
        and specifying 'orth' here will reference those containing orthographic 
        representations (ie, orthreps). (default is 'phon')

    axis : int
        The axis of x over which iteration shoulc occur. (default is 0)

    Returns
    -------
    bool
        A True value is provided if reconstructed x matches the representations
        in y. Else, a False value is returned.
    """
    if reps == 'phon':
        repdict = getreps('./raw/phonreps.csv')
    elif reps == 'orth':
        with open('./raw/orthreps.json') as j:
            repdict = json.load(j)



    def reconstruct_(example):
        return([key(repdict, e) for e in example.tolist()])

    r = []

    for ex in range(x.shape[0]):
        r.append(reconstruct_(x[ex]))

    r = [remove_all(e, '_') for e in r]
    
    if reps == 'phon':
        return(r == y)
    elif reps == 'orth':
        return([''.join(e) for e in r] == y)

def save(x, PATH):
    import pickle
    f = open(PATH, 'wb')
    pickle.dump(x, f)
    f.close()

def phontable(PATH):
    df = pd.read_csv(PATH)
    df = df.set_index('phone')
    feature_cols = [column for column in df if column.startswith('#')]
    df = df[feature_cols]
    df.columns = df.columns.str[1:]
    return(df)

def phonemedict(PATH, terminals=False):
    """Binary phonological reps from CSV.

    Parameters
    ----------
    PATH : str
        Path to the csv containing phonological representations.
    
    sos : bool
        Specify whether to add start-of-string
        feature to reps (default is not/ False). If true
        the character "#" is used. Note that if set to 
        true a new key-value pair is created in return 
        dict for this character, and a feature node is 
        added to every value in the dictionary.

    eos : bool
        Specify whether to add end-of-string
        feature to reps (default is not/ False). If true
        the character "%" is used. Note that if set to 
        true a new key-value pair is created in return 
        dict for this character, and a feature node is 
        added to every value in the dictionary.

    Returns
    -------
    dict
        A dictionary of phonemes and their binary representations
    """
    df = phontable(PATH)

    dict = {}
    for index, row in df.iterrows():
        dict[index] = row.tolist()

    if terminals:
        for k, v in dict.items():
            dict[k].extend([0, 0])
        dict['#'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        dict['%'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    return(dict)


def collapse(x, delimiter=','):

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


def numphones(x, delimiter='-'):
    count = 0
    if len(x) == 0:
        return(count)
    elif len(x) > 0:
        for i in x:
            if i == delimiter:
                count += 1
        return(count+1)



def numphones(x, delimiter='-'):
    count = 0
    if len(x) == 0:
        return(count)
    elif len(x) > 0:
        for i in x:
            if i == delimiter:
                count += 1
        return(count+1)

def mean(x):
    return sum(x)/len(x)



def load(PATH):
    import pickle
    f = open(PATH, 'rb')
    return(pickle.load(f))