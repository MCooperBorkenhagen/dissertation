
def nearest_phoneme(a, phonreps, round=True, ties='stop', return_array=False):
    """This is the updated version of dists() and is slightly faster than
    the previous method.

    Parameters
    ----------
    a : arr
        A numpy array to be compared with each value of phonreps.

    phonreps : dict
        A dictionary where every key is a string specifying symbolically the
        phoneme it represents, and each value is a numpy array to be compared
        with a.

    ties : str
        Test to see if ties are present. If a tie is present then an 
        exception will be raised. If set to 'stop', an exception is raised
        if the pairwise comparison across representations yields a tie. The
        alternative is the value 'sample' which yields a random representation
        selected from the ties if ties are present. (default is 'stop')

    round : bool
        Specify whether to round the input array or not prior to calculating
        the pairwise distances with values in phonreps. (default is True)

    return_array : bool
        Return an array representing the closest match to a, or return
        the symbolic string representing that array from phonreps.
        (default is True)

    Returns
    -------
    The phonological representation (array) that is nearest the array a,
    determined by pairwise comparisons across all values in phonreps using 
    the L2 norm for the distance calculation.

    """
    if round:
        a = np.around(a)

    d = {k:np.linalg.norm(a-np.array(v)) for k, v in phonreps.items()}
    mindist = min(d.values())
    
    u = [k for k, v in d.items() if v == mindist]

    if ties == 'stop': # selecting stop is equivalent to assuming that you want only a single phoneme to be selected
        assert len(u) == 1, 'More than one minumum value for pairwise distances for phonemes identified. Ties present.'
        s = u[0] # if the exception isn't raised then the selected phoneme is the single element of u  
    elif ties == 'sample':
        s = random.sample(u, 1)[0]

    if return_array:
        return(phonreps[s])
    elif not return_array:
        return(s)