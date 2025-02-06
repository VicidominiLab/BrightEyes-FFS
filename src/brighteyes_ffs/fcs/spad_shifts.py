import numpy as np


def spad_shifts():
    """

    Returns
    -------
    shift : 1D numpy array
        list of distances between SPAD elemenents (in a.u.).

    """
    
    shifts = []
    for xx in range(9):
        for yy in range(9):
            shifts.append(np.sqrt((xx-4)**2 + (yy-4)**2))
            
    shift = np.asarray(shifts)
    
    return shift