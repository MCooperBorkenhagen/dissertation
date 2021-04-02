"""
Here we have a bunch of simple functions that apply
over strings that can be extended for more complex
functions that operate over many wordforms. One
exception here is that first_vowel can operate over
a dictionary of wordforms (see below) or over a single
string.
"""
def first_vowel(wordforms, orthography = True):
    """Get the first vowel of a wordform or wordforms.
    Provide a dictionary where keys are orthographic wordforms 
    and values are phonological ones. For each wordform the
    first_vowel will be determined. If as string is supplied
    only that wordform will be processed.
    Parameters
    ----------
    wordforms : dict or str
        The wordform or wordforms for which the first_vowel will 
        be specified.
    orthography : bool
        If True, return the first orthographic vowel (the "heart"), 
        else the phonological one (the "anchor").
    Returns
    -------
    dict or str
        If dict is provided, a dict is returned with each orthform
        as the key and its first vowel (either orthographic or
        phonological depending on the orthography parameter) as the
        value. If str is supplied, the first vowel is supplied as str.
    """
    if orthography:
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
            return(heart)
        if type(wordforms) == dict:
            hearts = {}
            for (orthform, phonform) in wordforms.items():
                hearts[orthform] = heart(orthform)
            return(hearts)
        else:
            try:
                return(heart(wordforms))
            except TypeError('Neither a string nor a dictionary supplied. Check input:'):
                print(wordforms)
    elif not orthography:
        import pandas
        d = pandas.read_excel('/Users/MJBorkenhagen/pytools/data/ARPAbet.xlsx')
        vowels = d['one-letter'][d['Sound class'] == 'V'].dropna().tolist()
        def anchor(phonform):
            anchor = next((p for p in phonform if p in vowels), None)
            if anchor == None:
                raise TypeError('First vowel error.')
                print(format('No vowel detected. Check phonform {}', phonform))
            return(anchor)
        if type(wordforms) == dict:
            anchors = {}
            for (orthform, phonform) in wordforms.items():
                a = anchor(phonform)
                anchors[orthform] = a
            return(anchors)
        else:
            try:
                return(anchor(wordforms))
            except TypeError('Neither a string nor a dictionary supplied. Check input.'):
                print(wordforms)
def rime(wordform, first_vowel, return_length = False):
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
    r = wordform[wordform.index(first_vowel):]
    if not return_length:
        return(r)
    elif return_length:
        return(len(r))
def nucleus(wordform, first_vowel, vowels, return_length = False):
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
    r = rime(wordform, first_vowel)
    for l in r:
        if l in vowels:
            n = n + l
        else:
            break
    if not return_length:
        return(n)
    elif return_length:
        return(len(n))
def onset(wordform, first_vowel, return_length = False):
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
   