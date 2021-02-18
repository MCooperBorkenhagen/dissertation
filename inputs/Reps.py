
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

def pad_(wordform, maxlen):
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


def phonemedict(PATH, terminals=False):
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
def represent(wordform, representations, embed=False):
    if embed:
        return([[representations[e]] for e in wordform])
    elif not embed:
        return([representations[e] for e in wordform])




def remove_all(x, element):
    return(list(filter(lambda e: e != element, x)))


def key(dict, value):
    for k, v in dict.items():
        if value == v:
            return(k)

def reconstruct(x, y, repdict=None, join=True, remove_=False, axis=0):

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

    remove_ : bool
        Remove "_" or not. This will be useful if values in x contain the
        spacer character "_". (Default is False)

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

    if remove_:
        r = [remove_all(e, '_') for e in r]
    
    if not join:
        return(r == y)
    elif join:
        for e in r:
            return([''.join(e) for e in r] == y)







# %%
class Reps():

    """This class delivers representations for words to be used in ANNs.
    It is similar to CMUdata, stored elsewhere, but different because of how
    opinionated it is about the structure of its outputs and how to clean its
    inputs (words).

    """


    def __init__(self, words, outliers=None, oneletter=False, maxorth=None, maxphon=None, onehot=True, orthpad=0, phonpad=9, phon_index=0, terminals=False, justify='left', compounds=False, punctuation=False, tolower=True, test_reps=True):
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
            strings or not. This value is passed to phonemedict(). Note that
            this determines whether or not to supply two sets of phonological
            representations, because if terminals is true, it is assumed
            that phonological inputs and outputs are required (default is False).

        justify : str
            How to justify the patterns output. This specification is applied to
            all patterns produced (orthography, phonology, and if terminals is
            set to True, both input and output phonology). Note that a left justification
            means that the pad is placed on the right side of the representations,
            and vice versa for right justification. (Default is left.)
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
        
        # from here the words in cmudict and pool are set
        self.cmudict = {word: phonform for word, phonform in cmudict.items() if word in pool}
        self.pool = pool

        phonpath = 'https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/phonreps.csv'
        self.phonpath = phonpath
        self.phontable = phontable(phonpath)
        self.phonreps = phonemedict(phonpath, terminals=terminals)
        
        if onehot:
            orthpath = 'raw/orthreps-onehot.json'
        elif not onehot:
            orthpath = 'raw/orthreps.json'
        with open(orthpath, 'r') as f:
            orthreps = json.load(f)
        self.orthreps = orthreps
        self.outliers = outliers
        self.excluded = excluded


        # generate the padded version:

        if phonpad != 0:
            padrep = []
            for f in self.phonreps['_']:
                padrep.append(phonpad)
            self.phonreps['_'] = padrep

        if orthpad != 0:
            padrep = []
            for f in self.orthreps['_']:
                padrep.append(orthpad)
            self.orthreps['_'] = padrep

        # test that all the phonological vectors are the same length        
        veclengths = set([len(v) for v in self.phonreps.values()])
        assert(len(veclengths) == 1), 'Phonological feature vectors across phonreps have different lengths.'
        # derive the length of a single phoneme
        self.phoneme_length = next(iter(veclengths))


        if not onehot:
            veclengths = set([len(v) for v in self.orthreps.values()])
            assert(len(veclengths) == 1), 'Orthographic feature vectors across phonreps have different lengths.'

        if terminals:
            cmudictSOS = {}
            cmudictEOS = {}
            for word, phonform in self.cmudict.items():
                sos = cp(phonform)
                eos = cp(phonform)

                sos.insert(0, '#')
                eos.append('%')

                cmudictSOS[word] = sos
                cmudictEOS[word] = eos

            self.cmudictSOS = cmudictSOS
            self.cmudictEOS = cmudictEOS
        
        if terminals:
            self.phonformsSOS = {word: represent(self.cmudictSOS[word], self.phonreps) for word in pool}
            self.phonformsEOS = {word: represent(self.cmudictEOS[word], self.phonreps) for word in pool}
        elif not terminals:
            self.phonforms = {word: represent(self.cmudict[word], self.phonreps) for word in pool}

        self.orthforms = {word: represent(word, self.orthreps) for word in pool}
        self.orthlengths = {word: len(orthform) for word, orthform in self.orthforms.items()}

        if terminals:
            self.phonlengths = {word: len(phonform) for word, phonform in self.phonformsSOS.items()} # could also use EOS here, would have same result
        elif not terminals:
            self.phonlengths = {word: len(phonform) for word, phonform in self.phonforms.items()}


        # maximum phonological length and orthographic length are derived if they aren't specified at class call
        if maxorth is None:
            self.maxorth = max(self.orthlengths.values())
        
        if maxphon is None:
            self.maxphon = max(self.phonlengths.values())

        self.pool_with_pad = {}
        for word in pool:
            ppl = self.maxphon-self.phonlengths[word]
            opl = self.maxorth-self.orthlengths[word]
            orthpad = ''.join(['_' for p in range(opl)])
            phonpad = ['_' for p in range(ppl)]
            if not terminals:
                if justify == 'left':
                    self.cmudict[word].extend(phonpad)
                    print(word)
                    print(phonpad)
                elif justify == 'right':
                    new = cp(cmudict[word])
                    phonpad.extend(new)
                    self.cmudict[word] = phonpad
                    print('word:', word)
                    print('phonpad:', phonpad)
                    print('new:', new)
            elif terminals:
                if justify == 'left':
                    self.cmudictSOS[word].extend(phonpad)
                    self.cmudictEOS[word].extend(phonpad)
                elif justify == 'right':
                    sos = cp(phonpad)
                    eos = cp(phonpad)
                    sos.extend(cmudictSOS[word])
                    eos.extend(cmudictEOS[word])
                    self.cmudictSOS[word] = sos
                    self.cmudictEOS[word] = eos
            if justify == 'left':
                self.pool_with_pad[word] = word+orthpad
            elif justify == 'right':
                self.pool_with_pad[word] = orthpad+word

        
        self.orthforms_padded = {word: represent(orthform, self.orthreps) for word, orthform in self.pool_with_pad.items()}
        if terminals:
            self.phonformsEOS_padded = {word: represent(phonform, self.phonreps) for word, phonform in self.cmudictEOS.items()}
            self.phonformsSOS_padded = {word: represent(phonform, self.phonreps) for word, phonform in self.cmudictSOS.items()}
        if not terminals:
            self.phonforms_padded = {word: represent(phonform, self.phonreps) for word, phonform in self.cmudict.items()}




        # %% Array form
        self.orthforms_array = []
        if terminals:
            self.phonformsSOS_array = []
            self.phonformsEOS_array = []
        elif not terminals:
            self.phonforms_array = []
        
        for word in self.pool:
            self.orthforms_array.append(self.orthforms_padded[word])
            if terminals:
                self.phonformsSOS_array.append(self.phonformsSOS_padded[word])
                self.phonformsEOS_array.append(self.phonformsEOS_padded[word])
            elif not terminals:
                self.phonforms_array.append(self.phonforms_padded[word])

        self.orthforms_array = np.array(self.orthforms_array)

        if terminals:
            self.phonformsSOS_array = np.array(self.phonformsSOS_array)
            self.phonformsEOS_array = np.array(self.phonformsEOS_array)
        elif not terminals:
            self.phonforms_array = np.array(self.phonforms_array)


        #########
        # TESTS #
        #########
        if test_reps:
            assert reconstruct(self.orthforms_array, [self.pool_with_pad[word] for word in self.pool], repdict=self.orthreps, join=True), 'The padded orthographic representations do not match their string representations'
            if terminals:
                assert reconstruct(self.phonformsSOS_array, [self.cmudictSOS[word] for word in self.pool], repdict=self.phonreps, join=False), 'SOS phonological representations do not match their string representations'
                assert reconstruct(self.phonformsEOS_array, [self.cmudictEOS[word] for word in self.pool], repdict=self.phonreps, join=False), 'EOS phonological representations do not match their string representations'
            elif not terminals:
                assert reconstruct(self.phonforms_array, [self.cmudict[word] for word in self.pool], repdict=self.phonreps, join=False), 'Phonological representations do not match their string representations'


        # check that all the phonemes in words in pool are represented in phonreps:
        phones = [phone for word in pool for phone in self.cmudict[word]]
        assert set(phones).issubset(self.phonreps.keys()), 'Some phonemes are missing in your phonreps'

        # check that all the letters in pool are represented in orthreps:
        letters = []
        for word in self.pool:
            for l in word:
                letters.append(l)
        assert set(letters).issubset(self.orthreps.keys()), 'there are missing binary representations for letters in the set of words'

        # perform a test on all letters, making sure that we have a binary representation for it
        # need to change this to a different dictionary for each
        if terminals:
            assert set(self.orthforms.keys()) == set(self.phonformsSOS.keys()) == set(self.phonformsEOS.keys()), 'The keys in your orth and phon (SOS/ EOS) dictionaries do not match'    
        elif not terminals:
            assert set(self.orthforms.keys()) == set(self.phonforms.keys()), 'The keys in your orth and phon dictionaries do not match'






if __name__ == "__main__":
    Reps()