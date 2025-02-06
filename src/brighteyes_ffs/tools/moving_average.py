import numpy as np


def moving_average(data, n=3):
    """
    calculate moving average from list of values

    Parameters
    ----------
    data : np.array()
        np.array with data values.
    n : int, optional
        width of the window. The default is 3.

    Returns
    -------
    np.array()
        np.array with data values with moving average applied.

    """
    
    dataCum = np.cumsum(data, dtype=float)
    dataCum[n:] = dataCum[n:] - dataCum[:-n]
    
    return dataCum[n-1:] / n
