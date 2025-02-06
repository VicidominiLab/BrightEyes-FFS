def add_sum_column(data):
    """
    Add a column to a 2D array with the sum of all the other columns

    Parameters
    ----------
    data : 2D numpy array
        2D array with numbers.

    Returns
    -------
    data : 2D numpy array
        2D array with an additional column containing the sum of the 
        other columns

    """
    
    # import functions
    import numpy as np
    
    # data array size
    N = np.size(data, 0)
    M = np.size(data, 1)
    
    data2 = data # back-up
    data = np.zeros((int(N), M + 1), int)
    data[:, :-1] = data2
    data[:, M] = np.sum(data2, 1)
    
    return(data)