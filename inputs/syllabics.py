#%%
"""
Here we have a bunch of simple functions that apply
over strings that can be extended for more complex
functions that operate over many wordforms.
"""
from utilities import phontable, phonemedict, numeral_detect
import pandas



def append(x, y):
    """Combine x to y with a type check.
    
    Parameters
    ----------
    x : str
        The orthographic or phonological segment to add to y.

    y : str or list
        The object to which x is added. If y is a string, it
        is assumed to be an orthographic form. If it is a list
        it is assumed to be phonological. When excecuted
        y is type checked as part of the process.
    
    Returns
    -------
    str or list

    """
    assert type(x) == str, 'The x you have provided is not a string and should be. Check x and try again.'
    assert type(y) == str or type(y) == list, 'The y you have provided is neither a string or list. Check y and try again.'

    if type(y) == str:
        return y + x
    else:
        y.append(x)
        return y


#%%

def vowels(wordform, orthography=True):

    """Acquire the orthographic or phonological vowels of English.

    Parameters
    ----------
    wordform : str or list
        An orthographic (str) of phonological (list) wordform.
        If an orthographic wordform is provided, orthography
        should be set to True, else False. An orthographic
        wordform is a sequence of letters representing a word.
        A phonological wordform is a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    orthography : bool
        Specify if wordform is an orthographic form (True)
        or not (False). (Default is True)

    Returns
    -------
    list
        The vowels of English in either orthographic or 
        phonological form.
    """

    if orthography:
        if 'y' == wordform[0]:
            return(['a', 'e', 'i', 'o', 'u'])
        else:
            return(['a', 'e', 'i', 'o', 'u', 'y'])
    else:
        phonpath = 'https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/phonreps.csv'
        phonreps = phonemedict(phonpath, terminals=False)
        return([v for v in phonreps.keys() if numeral_detect(v)])



def phones(phonpath = 'https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/phonreps.csv', terminals=False):

    """Acquire the phones from two-letter ARPAbet, representing the phones of English.

    Parameters
    ----------
    phonpath : str
        A path to the table containing the two-letter phones of
        ARPAbet (i.e., those found in cmudict). The default is
        a prespecified table, but an alternative could be
        provided as long as it satifies subsequent requirements.

    terminals : bool
        Whether or not to acquire the terminal segments when
        compiling the phones from ARPAbet ("#" and "%").
        (Default is False)

    Returns
    -------
    list
        The phones of two-letter ARPAbet.
    """


    import pandas
    phonreps = phonemedict(phonpath, terminals=terminals)
    return([k for k in phonreps.keys()])


def alphabet(x, return_chars=False):

    """Check each character in x to make sure it is a member of the English alphabet.

    Parameters
    ----------
    x : str
        A string of letters to be checked for membership in
        the English alphabet.

    return_chars = bool
        Specify if the lowercase letters of English should
        be returned when executed. (Default is False)

    """

    chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for c in x:
        assert c in chars, 'Check the word: {}. It contains letters not in the English alphabet.'.format(x)
    if return_chars:
        return('abcdefghijklmnopqrstuvwxyz')

ARPAbet = phones()

def clean(wordform, orthography=True):

    """Clean a word so its syllabics can be processed.

    Parameters
    ----------
    wordform : str or list
        An orthographic (str) of phonological (list) wordform.
        If an orthographic wordform is provided, orthography
        should be set to True, else False. An orthographic
        wordform is a sequence of letters representing a word,
        whose letters will be converted to lowercase at runtime.
        A phonological wordform is a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    orthography : bool
        Specify if wordform is an orthographic form (True)
        or not (False). (Default is True)

    Returns
    -------
    str or list
        A clean version of the wordform. If orthographic, a string
        is returned. If phonological, a list is returned.
    """

    if orthography:
        alphabet(wordform) # checks that the word is a permissible word in the English alphabet
        return wordform.lower()
    else:
        for e in wordform:
            assert e in ARPAbet, 'The phonform supplied contains a phone not in ARPAbet. Check phonform: {}'.format(wordform)
        return(wordform)

def heart(orthform):

    """Acquire the first vowel of an orthographic wordform.

    Parameters
    ----------
    orthform : string
        An orthographic wordform supplied as a string of letters.

    Returns
    -------
    str
        A string designating the element of the orthform that is
        the first vowel.
    """
    orthform = clean(orthform, orthography=True)

    vs = vowels(orthform, orthography=True)
    if 'qu' in orthform:
        orthform = orthform.replace('qu', '~')
        heart  = next((o for o in orthform if o in vs), None)
        orthform = orthform.replace('~', 'qu')
    else:
        heart = next((o for o in orthform if o in vs), None)
    if heart == None:
        raise TypeError('First vowel error. No vowel detected. Check orthform: {}'.format(orthform))
    else:
        return(heart)


def anchor(phonform):
    """Acquire the first vowel of a phonological wordform.

    Parameters
    ----------
    phonform : list
        A phonological wordform supplied as a list of phones.
        The wordform should be a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    Returns
    -------
    str
        A string designating the element of phonform that is
        the first vowel.

    """
    phonform = clean(phonform, orthography=False)
    vs = vowels(None, orthography=False)
    anchor = next((p for p in phonform if p in vs), None)
    if anchor == None:
        raise TypeError('First vowel error. No vowel detected. Check phonform: {}'.format(phonform))
    else:
        return(anchor)


#%%
def first_vowel(wordform, orthography = True):
    """Acquire the first vowel of a wordform.

    Parameters
    ----------
    wordform : str or list
        An orthographic (str) of phonological (list) wordform.
        If an orthographic wordform is provided, orthography
        should be set to True, else False. An orthographic
        wordform is a sequence of letters representing a word,
        whose letters will be converted to lowercase at runtime.
        A phonological wordform is a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    orthography : bool
        Specify if wordform is an orthographic form (True)
        or not (False). (Default is True)

        
    Returns
    -------
    str
        The element of wordform corresponding to its first vowel.

    """
    wordform = clean(wordform, orthography=orthography)
    if orthography:
        return(heart(wordform))
    elif not orthography:
        return(anchor(wordform))



#%%
def rime(wordform, orthography=True, return_length=False):
    """Acquire the rime of a wordform (or its length).

    Parameters
    ----------
    wordform : str or list
        An orthographic (str) of phonological (list) wordform.
        If an orthographic wordform is provided, orthography
        should be set to True, else False. An orthographic
        wordform is a sequence of letters representing a word,
        whose letters will be converted to lowercase at runtime.
        A phonological wordform is a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    orthography : bool
        Specify if wordform is an orthographic form (True)
        or not (False). (Default is True)

    return_length : bool
        Specify whether to return the length of the return
        object (True) or the object itself (False). 
        (Default is False)
        
    Returns
    -------
    str, list or int
        If the wordform is orthographic and the object is returned
        then it is of type string. If phonological, then the object
        is type list. If the length of the object rather than the
        object itself is returned, then the type is integer.

    """
    wordform = clean(wordform, orthography=orthography)
    fv = first_vowel(wordform, orthography=orthography)

    r = wordform[wordform.index(fv):]
    if not return_length:
        return(r)
    elif return_length:
        return(len(r))


#%%
def nucleus(wordform, orthography=True, return_length=False):
    """Acquire the coda of a wordform (or its length).

    Parameters
    ----------
    wordform : str or list
        An orthographic (str) of phonological (list) wordform.
        If an orthographic wordform is provided, orthography
        should be set to True, else False. An orthographic
        wordform is a sequence of letters representing a word,
        whose letters will be converted to lowercase at runtime.
        A phonological wordform is a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    orthography : bool
        Specify if wordform is an orthographic form (True)
        or not (False). (Default is True)

    return_length : bool
        Specify whether to return the length of the return
        object (True) or the object itself (False). 
        (Default is False)
        
    Returns
    -------
    str, list or int
        If the wordform is orthographic and the object is returned
        then it is of type string. If phonological, then the object
        is type list. If the length of the object rather than the
        object itself is returned, then the type is integer.

    """
    wordform = clean(wordform, orthography=orthography)
    if orthography:
        n = ''
    else:
        n = []

    r = rime(wordform, orthography=orthography)
    vs = vowels(wordform, orthography=orthography)
    for l in r:
        if l in vs:
            n = append(l, n)
        else:
            break
    if not return_length:
        return(n)
    elif return_length:
        return(len(n))




def onset(wordform, orthography=True, return_length=False):
    """Acquire the onset of a wordform (or its length).

    Parameters
    ----------
    wordform : str or list
        An orthographic (str) of phonological (list) wordform.
        If an orthographic wordform is provided, orthography
        should be set to True, else False. An orthographic
        wordform is a sequence of letters representing a word,
        whose letters will be converted to lowercase at runtime.
        A phonological wordform is a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    orthography : bool
        Specify if wordform is an orthographic form (True)
        or not (False). (Default is True)

    return_length : bool
        Specify whether to return the length of the return
        object (True) or the object itself (False). 
        (Default is False)
        
    Returns
    -------
    str, list or int
        If the wordform is orthographic and the object is returned
        then it is of type string. If phonological, then the object
        is type list. If the length of the object rather than the
        object itself is returned, then the type is integer.

    """
    wordform = clean(wordform, orthography=orthography)
    fv = first_vowel(wordform, orthography=orthography)
    
    o = wordform[0:wordform.index(fv)]
    if not return_length:
        return(o)
    elif return_length:
        return(len(o))




def oncleus(wordform, orthography=True, return_length=False):

    """Acquire the oncleus of a wordform (or its length).

    Parameters
    ----------
    wordform : str or list
        An orthographic (str) of phonological (list) wordform.
        If an orthographic wordform is provided, orthography
        should be set to True, else False. An orthographic
        wordform is a sequence of letters representing a word,
        whose letters will be converted to lowercase at runtime.
        A phonological wordform is a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    orthography : bool
        Specify if wordform is an orthographic form (True)
        or not (False). (Default is True)

    return_length : bool
        Specify whether to return the length of the return
        object (True) or the object itself (False). 
        (Default is False)
        
    Returns
    -------
    str, list or int
        If the wordform is orthographic and the object is returned
        then it is of type string. If phonological, then the object
        is type list. If the length of the object rather than the
        object itself is returned, then the type is integer.

    """
    wordform = clean(wordform, orthography=orthography)
    fv = first_vowel(wordform, orthography=orthography)
    n = nucleus(wordform, orthography=orthography)
    o = wordform[0:wordform.index(fv)+len(n)]
    if not return_length:
        return(o)
    else:
        return(len(o))


def coda(wordform, orthography=True, return_length = False):
    """Acquire the coda of a wordform (or its length).

    Parameters
    ----------
    wordform : str or list
        An orthographic (str) of phonological (list) wordform.
        If an orthographic wordform is provided, orthography
        should be set to True, else False. An orthographic
        wordform is a sequence of letters representing a word,
        whose letters will be converted to lowercase at runtime.
        A phonological wordform is a list whose elements are
        two-letter ARPAbet phones (as in cmudict).

    orthography : bool
        Specify if wordform is an orthographic form (True)
        or not (False). (Default is True)

    return_length : bool
        Specify whether to return the length of the return
        object (True) or the object itself (False). 
        (Default is False)
        
    Returns
    -------
    str, list or int
        If the wordform is orthographic and the object is returned
        then it is of type string. If phonological, then the object
        is type list. If the length of the object rather than the
        object itself is returned, then the type is integer.

    """
    wordform = clean(wordform, orthography=orthography)

    r = rime(wordform, orthography=orthography)
    n = nucleus(wordform, orthography=orthography)
    c = r[len(n):]
    if not return_length:
        return(c)
    elif return_length:
        return(len(c))
#%%
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
    

def contour(phonform):
    """Acquire the stress pattern over vowels of a phonological wordform.

    Parameters
    ----------
    phonform : list
        A phonological form expressed in a list with phones
        defined as two-letter ARPAbet phonemic segments
        (i.e., as in cmudict). Each element in the list
        is a single phonemic segment.

    Returns
    -------
    list
        A list is returned indicating the vowel pattern of
        the word with three possible stress designations:
        0, 1, and 2. 1 marks primary stress, 2 marks secondary
        stress, and 0 marks no stress.

    """
    phonform = clean(phonform, orthography=False)
    return([e[-1] for e in phonform if numeral_detect(e)])


def get_vowels(phonform):
    """Acquire the vowels of a phonological wordform.

    Parameters
    ----------
    phonform : list
        A phonological form expressed in a list with phones
        defined as two-letter ARPAbet phonemic segments
        (i.e., as in cmudict). Each element in the list
        is a single phonemic segment.

    Returns
    -------
    list
        A list is returned with the vowels from phonform
        as each element. The order of vowels in the
        return corresponds to the order of vowels in phonform.

    """
    phonform = clean(phonform, orthography=False)
    return([e for e in phonform if numeral_detect(e)])

