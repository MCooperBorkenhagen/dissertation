
# %%

import pandas as pd
import numpy as np
import os
import json
import re
import random
import nltk
from copy import deepcopy as cp

# %%%

def pad(wordform, maxlen):
    padlen = maxlen - len(wordform)
    return(wordform + ('_'*padlen))


def remove(list, pattern = '[0-9]'): 
    """
    Remove a string from each element of a list, defaults
    to removing numeric strings.
    """
    list = [re.sub(pattern, '', i) for i in list] 
    return(list)

# %%
# utility functions:
def phontable(PATH):
    df = pd.read_csv(PATH)
    df = df.set_index('phone')
    feature_cols = [column for column in df if column.startswith('#')]
    df = df[feature_cols]
    df.columns = df.columns.str[1:]
    return(df)


def phondict(PATH, terminals=False):
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
    df = phontable(PATH)
    """
    df = pd.read_excel(PATH, engine='openpyxl')
    df = df[df['phone'].notna()]
    df = df.rename(columns={'phone': 'phoneme'})
    df = df.set_index('phoneme')
    feature_cols = [column for column in df if column.startswith('#')]
    df = df[feature_cols]
    df.columns = df.columns.str[1:]"""
    dict = {}
    for index, row in df.iterrows():
        dict[index] = row.tolist()

    if terminals:
        for k, v in dict.items():
            dict[k].append(0)
        dict['#'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        dict['%'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    return(dict)

# generate binary phonological representation for a wordform (will work for orth or phon)
def represent(wordform, representations):
    return([representations[e] for e in wordform])

# %%
class Reps():

    """This class delivers representations for words to be used in ANNs.
    It is similar to CMUdata, stored elsewhere, but different because of how
    opinionated it is about the structure of its outputs and how to clean its
    inputs (words).

    """


    def __init__(self, words, outliers=None, oneletter=False, maxorth=None, maxphon=None, onehot=True, orthpad=0, phonpad=9, phon_index=0, terminals=False, compounds=False, punctuation=False, tolower=True):
        """Initialize Reps with a values that specify representations over words.
        Parameters
        ----------
        words : list
            A list of ortohgraphic wordforms to be encoded into orthographic
            and phonological representations.

        outliers : list or None
            A list of words to be excluded from representations of words, or None
            if no excluding is required. (default None)

        phonlabel : str
            Label of phoneme to be specified when producing phontable and phonreps.
            "two-letter" or "IPA" are supported, though "IPA" hasn't been tested and
            may produce idiosyncratic behavior. 

        oneletter : bool
            Whether to exclude words that are one letter long or not. (Default is True)

        maxorth : int or None
            The maximum length of the orthographic wordforms to populate the pool
            for representations. This value is calculated inclusively. (Default is None)

        maxphon : int or None
            The maximum length of the phonological wordforms to populate the pool
            for representations. This value is calculated inclusively. (Default is None)

        onehot : bool
            Specify orthographic representations with onehot codings. (Default is True)

        orthpad : int or None
            The value to supply as an orthographic pad. (Default is 0)

        phonpad : int or None
            The value to supply as a phonological pad. (Default is 9)

        phon_index : int or string
            The index to use for specifying which phonological representation
            from cmudict to pass for conversion to binary phonological representation.

        terminals : bool
            Whether to construct phonological representations with terminal
            strings or not. This value is passed to phondict(). (default is False)
        """

        # clean data at initialization (parameters are passed at init)
        # skip words that are one letter long

        pool = cp(words)

        tmp = nltk.corpus.cmudict.dict() # the raw cmudict object
        cmudict = {word: phonforms[phon_index] for word, phonforms in tmp.items() if word in pool}

        if outliers is not None:
            if type(outliers) == str:
                outliers = [outliers]
            excluded = {word:"identified as outlier at class call" for word in outliers}
            pool = [word for word in pool if word not in outliers]
        else:
            excluded = {}

        if not oneletter:
            for word in pool:
                if len(word) == 1:
                    pool.remove(word)
                    print(word, 'removed from pool because it has one letter')
                    excluded[word] = "one letter word"

        for word in pool:
            if word not in cmudict.keys():
                excluded[word] = "missing from cmudict"
                print(word, 'removed from pool because it is missing in cmudict')
                pool.remove(word)

        if maxorth is not None:
            for word in pool:
                if len(word) > maxorth:
                    excluded[word] = "too many letters"
                    print(word, 'removed from pool because it has too many letters')
                    pool.remove(word)

        if maxphon is not None:
            toomanyphones = []
            for word in pool:
                if len(cmudict[word]) > maxphon:
                    excluded[word] = "too many phonemes"
                    print(word, 'removed from pool because it has too many phonemes')
                    pool.remove(word)


        # now clean cmudict:
        if not compounds:
            cmudict = {word: phonform for word, phonform in cmudict.items() if '-' not in word}

        if not punctuation:
            regex = re.compile('[^a-zA-Z]')
            for k in cmudict.keys(): # you have to convert to list to change key of dict in loop
                if not k.isalpha():
                    new = regex.sub('', k)
                    cmudict[new] = cmudict.pop(k)

        if tolower:
            for k in cmudict.keys():
                new = k.lower()
                cmudict[new] = cmudict.pop(k)
        
        self.cmudict = cmudict
        self.pool = pool

        phonpath = 'https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/phonreps.csv'
        self.phonpath = phonpath
        self.phontable = phontable(phonpath)

        phonreps = phondict(phonpath, terminals=terminals)
        self.phonreps =  phonreps
        if onehot:
            orthpath = 'raw/orthreps-onehot.json'
        elif not onehot:
            orthpath = 'raw/orthreps.json'
        with open(orthpath, 'r') as f:
            orthreps = json.load(f)
        self.orthreps = orthreps
        
        self.orthforms = {word: represent(word, orthreps) for word in pool}
        self.phonforms = {word: represent(cmudict[word], phonreps) for word in pool}
        self.orthlengths = {word: len(phonform) for word, phonform in self.phonform.items()}
        self.phonlengths = {word: len(orthform), for word, orthform in self.orthform.items())}
        self.outliers = outliers
        self.excluded = excluded

        #########
        # TESTS #
        #########
        # check that all the phonemes in words in pool are represented in phonreps:
        phones = [phone for word in pool for phone in cmudict[word]]
        assert set(phones).issubset(self.phonreps.keys()), 'Some phonemes are missing in your phonreps'

        # check that all the letters in pool are represented in orthreps:
        letters = []
        for word in self.pool:
            for l in word:
                letters.append(l)
        assert set(letters).issubset(self.orthreps.keys()), 'there are missing binary representations for letters in the set of words'

        # perform a test on all letters, making sure that we have a binary representation for it
        # need to change this to a different dictionary for each
        assert set(self.orthforms.keys()) == set(self.phonforms.keys()), 'The keys in your orth and phon dictionaries do not match'

        # generate the padded version:
        if maxorth is None:
            maxorth = max([len(orthform) for orthform in self.orthform.items()])

        
        padded = {}

        for word in pool:
            padlen = maxphon-phonlengths[word]
            p = phonreps['#']
            p.append(phonize(cmudict([word]))
            for slot in range(padlen):
                p.append(phonpad)
            phon_sos_masked_right[word] = p





if __name__ == "__main__":
    Reps()