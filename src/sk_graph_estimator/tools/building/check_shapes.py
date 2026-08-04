def shapes_equal(a, b):
    '''
    Checks if the shapes of two arrays are equal

    Parameters
    ----------
    a : array-like
        The first array
    
    b : array-like
        The second array
    
    Returns
    -------
    bool
        True if the shapes are equal
        
        False otherwise
    '''
    if len(a) != len(b):
        return False
    return all(
        (x == y) or (x is None) or (y is None)
        for x, y in zip(a, b)
    )