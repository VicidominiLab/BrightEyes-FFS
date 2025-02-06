import numpy as np


def find_nearest(array, value):
    """
    Find nearest value and index of value in array.
    function call: [value2, idx] = find_nearest(array, value)

    Parameters
    ----------
    array : np.array
        1D array of values to look into.
    value : float
        Number to find in the array.

    Returns
    -------
    value2 : float
        Value in array that is closest to value.
    idx : int
        Index of this value in the array.

    """
    
    array = np.asarray(array)
    
    idx = (np.abs(array - value)).argmin()
    value2 = array[idx]
    
    return value2, idx
