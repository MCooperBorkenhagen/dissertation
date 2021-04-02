
# %%

import pandas as pd
import numpy as np
import os
import json
import re
import random
import nltk
from copy import deepcopy as cp
import string



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


def phonemedict(PATH, sos=False, eos=False):
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

    if eos and sos:
        for k, v in dict.items():
            dict[k].extend([0, 0])
        dict['#'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        dict['%'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif eos and not sos:
        for k, v in dict.items():
            dict[k].append(0)
        dict['%'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif sos and not eos:
        raise Exception('You selected sos = True and eos = False -- you probably do not want that')

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
    
    for ex in range(x.shape[axis]):
        if type(x[ex]) == list:
            print(ex)
            print(x[ex])
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


    def __init__(self, words, outliers=None, cmudict_supplement=None, phonpath=None, oneletter=False, maxorth=None, maxphon=None, onehot=True, orthpad=0, phonpad=9, phon_index=0, sos=False, eos=False, justify='left', punctuation=False, numerals=False, tolower=True, test_reps=True):
        """Initialize Reps with a values that specify representations over words.
        Parameters
        ----------
        words : list
            A list of ortohgraphic wordforms to be encoded into orthographic
            and phonological representations.

        outliers : list or None
            A list of words to be excluded from representations of words, or None
            if no excluding is required. (default None)

        cmudict_supplement : str or None
            A path to a json file to be used to supplement phonological
            transcriptions contained in cmudict. Keys in the dict object
            represented in the json file should match words provided.

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

        sos : bool
            Whether to construct phonological representations with sos terminal
            string ('#') or not. This value is passed to phonemedict(). Note that
            this determines whether or not to supply an extra phonological
            representation for the sos, because if true, it is assumed
            that phonological inputs and outputs are required (default is False).
            If this argument is set to True and eos is set to False and exception
            is raised because it isn't like that this circumstance was intentional.
            The sos representation is usually only required if eos is set to
            True (but the opposite is not the case).

        eos : bool
            Whether to construct phonological representations with eos terminal
            string ('%') or not. This value is passed to phonemedict(). Note that
            this determines whether or not to supply an extra phonological
            representation for the eos, because if true, it is assumed
            that a different set of phonological inputs and outputs are required 
            (default is False), where the representations containing the eos
            (and associated binary rep) are contained in the output representations.

        justify : str
            How to justify the patterns output. This specification is applied to
            all patterns produced (orthography, phonology, and if eos is
            set to True, both input and output phonology). Note that a left justification
            means that the pad is placed on the right side of the representations,
            and vice versa for right justification. (Default is left.)
        """

        # clean data at initialization (parameters are passed at init)
        # skip words that are one letter long

        pool = cp(words)

        cmudict = {word: phonforms[phon_index] for word, phonforms in nltk.corpus.cmudict.dict().items() if word in pool}

        if outliers is not None:
            if type(outliers) == str:
                outliers = [outliers]
            excluded = {word:"identified as outlier at class call" for word in outliers}
            pool = [word for word in pool if word not in outliers]
        else:
            excluded = {}

        if cmudict_supplement is not None:
            with open(cmudict_supplement, 'r') as f:
                supp = json.load(f)
            for word, phonforms in supp.items():
                if word in pool:
                    cmudict[word] = phonforms[phon_index]


        notin_cmu = [word for word in pool if word not in cmudict.keys()]
        pool = [word for word in pool if word not in notin_cmu]
        for word in notin_cmu:
            excluded[word] = "missing from cmudict"
            print(word, 'removed from pool because it is missing in cmudict')

        if not oneletter:
            for word in pool:
                print(word)
                if len(word) == 1:
                    pool.remove(word)
                    print(word, 'removed from pool because it has one letter')
                    excluded[word] = "one letter word"

        if maxorth is not None:
            toomanyletters = [word for word in pool if len(word) > maxorth]
            pool = [word for word in pool if word not in toomanyletters]
            for word in toomanyletters:
                excluded[word] = "too many letters"
                print(word, 'removed from pool because it has too many letters')

        if maxphon is not None:
            toomanyphones = [word for word in pool if len(cmudict[word]) > maxphon]
            pool = [word for word in pool if word not in toomanyphones]
            for word in toomanyphones:
                excluded[word] = "too many phonemes"
                print(word, 'removed from pool because it has too many phonemes')

        if not punctuation:
            punct = string.punctuation
            has_punct = [word for word in pool for ch in word if ch in punct]
            pool = [word for word in pool if word not in has_punct]
            for word in has_punct:
                excluded[word] = "puctuation present"
                print(word, 'removed because punctuation is present')

        if not numerals:
            has_numerals = [word for word in pool if any(ch.isdigit() for ch in word)]
            pool = [word for word in pool if word not in has_numerals]
            for word in has_numerals:
                excluded[word] = 'contains numerals'
                print(word, 'removed because it contains numerals')

        if tolower:
            pool = [word.lower() for word in pool]
        
        # from here the words in cmudict and pool are set
        self.cmudict = {word: phonform for word, phonform in cmudict.items() if word in pool}
        self.pool = pool

        if phonpath is None:
            phonpath = 'https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/phonreps.csv'
        self.phonpath = phonpath
        self.phontable = phontable(phonpath)
        self.phonreps = phonemedict(phonpath, sos=sos, eos=eos)
        
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

        if sos and eos:
            cmudictSOS = {}
            cmudictEOS = {}
            for word in self.pool:
                sos = cp(self.cmudict[word])
                eos = cp(self.cmudict[word])

                sos.insert(0, '#')
                eos.append('%')

                cmudictSOS[word] = sos
                cmudictEOS[word] = eos

            self.cmudictSOS = cmudictSOS
            self.cmudictEOS = cmudictEOS


        elif eos and not sos:
            cmudictX = {}
            cmudictEOS = {}
            for word in self.pool:
                X = cp(self.cmudict[word])
                eos = cp(self.cmudict[word])

                X.insert(len(X), '_')
                eos.append('%')

                cmudictX[word] = X
                cmudictEOS[word] = eos

            self.cmudictX = cmudictX
            self.cmudictEOS = cmudictEOS

        elif sos and not eos:
            raise Exception('You selected sos = True and eos = False -- you probably do not want that')

        if eos and sos:
            self.phonformsSOS = {word: represent(self.cmudictSOS[word], self.phonreps) for word in self.pool}
            self.phonformsEOS = {word: represent(self.cmudictEOS[word], self.phonreps) for word in self.pool}
        
        elif eos and not sos:
            self.phonformsX = {word: represent(self.cmudictX[word], self.phonreps) for word in self.pool}
            self.phonformsEOS = {word: represent(self.cmudictEOS[word], self.phonreps) for word in self.pool}

        elif not eos and not sos:
            self.phonforms = {word: represent(self.cmudict[word], self.phonreps) for word in self.pool}

        self.orthforms = {word: represent(word, self.orthreps) for word in self.pool}
        self.orthlengths = {word: len(orthform) for word, orthform in self.orthforms.items()}

        if eos:
            self.phonlengths = {word: len(phonform) for word, phonform in self.phonformsEOS.items()} # EOS used because SOS is never called without EOS
        elif not eos:
            self.phonlengths = {word: len(phonform) for word, phonform in self.phonforms.items()}


        # maximum phonological length and orthographic length are derived if they aren't specified at class call
        if maxorth is None:
            self.maxorth = max(self.orthlengths.values())
        else:
            self.maxorth = maxorth
        
        if maxphon is None:
            self.maxphon = max(self.phonlengths.values())
        else:
            self.maxphon = maxphon

        self.pool_with_pad = {}
        for word in self.pool:

            if eos:
                ppl = (self.maxphon+1)-self.phonlengths[word] # add 1 because maxphon doesn't take into account the terminal character
            elif not eos:
                ppl = self.maxphon-self.phonlengths[word]

            opl = self.maxorth-self.orthlengths[word]
            orthpad = ''.join(['_' for p in range(opl)])
            phonpad = ['_' for p in range(ppl)]

            if not eos: # note that sos is never called without eos, so not eos is enough here
                if justify == 'left':
                    self.cmudict[word].extend(phonpad)
                elif justify == 'right':
                    new = cp(cmudict[word])
                    phonpad.extend(new)
                    self.cmudict[word] = phonpad
            elif eos and sos:
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
            elif eos and not sos:
                if justify == 'left':
                    self.cmudictX[word].extend(phonpad)
                    self.cmudictEOS[word].extend(phonpad)
                elif justify == 'right':
                    X = cp(phonpad)
                    eos = cp(phonpad)
                    X.extend(cmudictX[word])
                    eos.extend(cmudictEOS[word])
                    self.cmudictX[word] = X
                    self.cmudictEOS[word] = eos

            if justify == 'left':
                self.pool_with_pad[word] = word+orthpad
            elif justify == 'right':
                self.pool_with_pad[word] = orthpad+word

        
        self.orthforms_padded = {word: represent(orthform, self.orthreps) for word, orthform in self.pool_with_pad.items()}
        if eos and sos:
            self.phonformsEOS_padded = {word: represent(phonform, self.phonreps) for word, phonform in self.cmudictEOS.items()}
            self.phonformsSOS_padded = {word: represent(phonform, self.phonreps) for word, phonform in self.cmudictSOS.items()}
        elif eos and not sos:
            self.phonformsX_padded = {word: represent(phonform, self.phonreps) for word, phonform in self.cmudictX.items()}
            self.phonformsEOS_padded = {word: represent(phonform, self.phonreps) for word, phonform in self.cmudictEOS.items()}
        elif not eos:
            self.phonforms_padded = {word: represent(phonform, self.phonreps) for word, phonform in self.cmudict.items()}




        # %% Array form
        self.orthforms_array = []
        if eos and sos:
            self.phonformsSOS_array = []
            self.phonformsEOS_array = []
        elif eos and not sos:
            self.phonformsX_array = []
            self.phonformsEOS_array = []
        elif not eos:
            self.phonforms_array = []
        
        for word in self.pool:
            self.orthforms_array.append(self.orthforms_padded[word])
            if eos and sos:
                self.phonformsSOS_array.append(self.phonformsSOS_padded[word])
                self.phonformsEOS_array.append(self.phonformsEOS_padded[word])
            elif eos and not sos:
                self.phonformsX_array.append(self.phonformsX_padded[word])
                self.phonformsEOS_array.append(self.phonformsEOS_padded[word])
            elif not eos:
                self.phonforms_array.append(self.phonforms_padded[word])

        self.orthforms_array = np.array(self.orthforms_array)

        if eos and sos:
            self.phonformsSOS_array = np.asarray(self.phonformsSOS_array)
            self.phonformsEOS_array = np.asarray(self.phonformsEOS_array)
        elif eos and not sos:
            self.phonformsX_array = np.asarray(self.phonformsX_array)
            self.phonformsEOS_array = np.asarray(self.phonformsEOS_array)
        if not eos:
            self.phonforms_array = np.array(self.phonforms_array)


        #########
        # TESTS #
        #########
        if test_reps:
            assert reconstruct(self.orthforms_array, [self.pool_with_pad[word] for word in self.pool], repdict=self.orthreps, join=True), 'The padded orthographic representations do not match their string representations'
            if eos:
                assert reconstruct(self.phonformsEOS_array, [self.cmudictEOS[word] for word in self.pool], repdict=self.phonreps, join=False), 'EOS phonological representations do not match their string representations'
            if sos:
                assert reconstruct(self.phonformsSOS_array, [self.cmudictSOS[word] for word in self.pool], repdict=self.phonreps, join=False), 'SOS phonological representations do not match their string representations' 
            if eos and not sos:
                assert reconstruct(self.phonformsX_array, [self.cmudictX[word] for word in self.pool], repdict=self.phonreps, join=False), 'X phonological representations (X values paired with EOS values but lacking SOS) do not match their string representations' 
            if not eos:
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
        # need to change this to a different dictionary for each?
        if eos and sos:
            assert set(self.orthforms.keys()) == set(self.phonformsSOS.keys()) == set(self.phonformsEOS.keys()), 'The keys in your orth and phon (SOS/ EOS) dictionaries do not match'    
        elif eos and not sos:
            assert set(self.orthforms.keys()) == set(self.phonformsX.keys()) == set(self.phonformsEOS.keys()), 'The keys in your orth and phon (X/ EOS) dictionaries do not match'    
        elif not eos:
            assert set(self.orthforms.keys()) == set(self.phonforms.keys()), 'The keys in your orth and phon dictionaries do not match'




if __name__ == "__main__":
    Reps()