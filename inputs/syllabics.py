#%%
"""
Here we have a bunch of simple functions that apply
over strings that can be extended for more complex
functions that operate over many wordforms.
"""
from utilities import phontable, phonemedict, numeral_detect

def heart(orthform):
    if 'y' == orthform[0]:
        vowels = ['a', 'e', 'i', 'o', 'u']
    else:
        vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    if 'qu' in orthform:
        orthform = orthform.replace('qu', '~')
        heart  = next((o for o in orthform if o in vowels), None)
        orthform = orthform.replace('~', 'qu')
    else:
        heart = next((o for o in orthform if o in vowels), None)
    if heart == None:
        raise TypeError('First vowel error.')
        print(format('No vowel detected. Check orthform {}', orthform))
    else:
        return(heart)


def anchor(phonform, phonpath=None, terminals=True):
    import pandas
    
    
    if phonpath is None:
        phonpath = 'https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/phonreps.csv'

    phonreps = phonemedict(phonpath, terminals=terminals)

    vowels = [v for v in phonreps.keys() if numeral_detect(v)]

    anchor = next((p for p in phonform if p in vowels), None)
    if anchor == None:
        raise TypeError('First vowel error.')
        print(format('No vowel detected. Check phonform {}', phonform))
    else:
        return(anchor)


#%%
def first_vowel(wordform, orthography = True, phonpath=None, terminals=True, **kwargs):
    """Get the first vowel of a wordform.
    For the wordform provided first_vowel will be determined.

    Parameters
    ----------
    wordform : str or list
        The wordform for which the first_vowel will 
        be specified. If a string is provided, the method will
        expect orthography = True, if a list it expects that
        orthography = False. This is due to expectations about
        the form of orthographic versus phonological forms used.

    orthography : bool
        If True, return the first orthographic vowel (the "heart"), 
        else the phonological one (the "anchor").

    Returns
    -------
    str
        Returns the string corresponding to the first vowel in the word
    """
    if orthography:
        return(heart(wordform))
    elif not orthography:
        



def rime(wordform, first_vowel=None, orthography=True, return_length=False, phonpath=None, terminals=True):
    """Get the rime of a wordform.
    Either a phonolological or orthographic wordform can be supplied.
    Parameters
    ----------
    wordform : str
        Either a phonological or orthographic wordform.
    first_vowel : str
        The first vowel of wordform, identifying the point at which the rime starts.
    Returns
    -------
    str
        The rime of the wordform.
    """

    if first_vowel is None:
        first_vowel = first_vowel(wordform, orthography=orthography, phonpath=phonpath, terminals=terminals)

    r = wordform[wordform.index(first_vowel):]
    if not return_length:
        return(r)
    elif return_length:
        return(len(r))


def nucleus(wordform, first_vowel=None, vowels, return_length = False, phonpath=None, terminals=True):
    """Get the nucleus of a wordform.
    Either a phonolological or orthographic wordform can be supplied.
    Parameters
    ----------
    wordform : str
        Either a phonological or orthographic wordform.
    first_vowel : str
        The first vowel of wordform, identifying the point at which the nucleus starts.
    Returns
    -------
    str
        The nucleus of the wordform.
    """
    n = ''
    r = rime(wordform, first_vowel=first_vowel)
    for l in r:
        if l in vowels:
            n = n + l
        else:
            break
    if not return_length:
        return(n)
    elif return_length:
        return(len(n))




def onset(wordform, first_vowel=None, return_length = False):
    """Get the onset of a wordform.
    Either a phonological or orthographic wordform can be supplied.
    Parameters
    ----------
    wordform : str
        Either an orthographic or phonological wordform
    vowel : str
        The first vowel, identifying the point at which the onset terminates.
    Returns
    -------
    str
        The onset of the wordform.
    """
    if first_vowel == None:
        
        return(None)
    else:
        o = wordform[0:wordform.index(first_vowel)]
        if not return_length:
            return(o)
        elif return_length:
            return(len(o))


def oncleus(wordform, first_vowel, vowels, orthography=False, return_length=False):
    n = nucleus(wordform, first_vowel, vowels)
    o = wordform[0:wordform.index(first_vowel)+len(n)]
    if not return_length:
        return(o)
    else:
        return(len(o))
def coda(wordform, first_vowel, vowels, return_length = False):
    r = rime(wordform, first_vowel)
    n = nucleus(wordform, first_vowel, vowels)
    c = r[len(n):]
    if not return_length:
        return(c)
    elif return_length:
        return(len(c))
