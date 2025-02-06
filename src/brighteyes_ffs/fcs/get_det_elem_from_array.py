import numpy as np

def get_elsum(n, s):
    """
    Return list of all elements within ring s
    E.g.    s = 0 --> central element
            s = 1 --> sum3x3
            s = 2 --> sum5x5

    Parameters
    ----------
    N : int
        Number of rows/columns of the array, typically 5.
    s : int
        Outer ring number, typically between 0 and 2.

    Returns
    -------
    list
        list of indices.

    """
    
    out = []
    for r in range(s+1):
        out = np.append(out, get_el_ring_array(n, r))
    return [int(x) for x in out]


def get_el_ring_array(n, r):
    """
    Return list of all elements of a square array that are in ring r around
    the center. E.g. for a 5x5 array, there are three rings, ring 0-2:
    2 2 2 2 2
    2 1 1 1 2
    2 1 0 1 2
    2 1 1 1 2
    2 2 2 2 2
    E.g. Ring 1 contains elements [6,7,8,11,13,16,17,18]

    Parameters
    ----------
    n : int
        Number of rows/columns of the array, typically 5.
    r : int
        Ring number, typically between 0 and 2.

    Returns
    -------
    TYPE
        list of indices.

    """
    
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.max([np.abs(i-np.floor(n/2)), np.abs(j-np.floor(n/2))])
    
    dist = np.reshape(dist, (n*n,1))
    
    return np.where(dist == r)[0]