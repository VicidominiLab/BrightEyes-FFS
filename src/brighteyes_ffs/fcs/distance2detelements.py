import numpy as np

def distance2detelements(det1, det2):
    """
    Calculate the distance between two SPAD detector elements in a.u.

    Parameters
    ----------
    det1 : int
        Number of the first detector element.
    det2 : int
        Number of the second detector element.

    Returns
    -------
    distance : float
        Distance between the two elements.

    """
    
    row1 = np.floor(det1 / 5)
    col1 = np.mod(det1, 5)
    
    row2 = np.floor(det2 / 5)
    col2 = np.mod(det2, 5)
    
    distance = np.sqrt((row2 - row1)**2 + (col2 - col1)**2)
    
    return distance


def spad_coord_from_det_numb(det):
    x = np.mod(det, 5)
    y = int(np.floor(det / 5))
    return([y, x])


def spad_shift_vector_crosscorr(vector=[0, 0], n=5):
    """
    Return a list of all cross-correlations between two channels that are
    displaced by a given shift vector. Used for STICS analysis.

    Parameters
    ----------
    vector : list, optional
        List with two elements [shifty, shiftx] in units of pixels.
        The default is [0, 0].
    n : int, optional
        Number of detector elements of the square detector. The default is 5
        for a 5x5 detector

    Returns
    -------
    CC : list of strings
        Each string correspond to a cross-correlation between two detector
        elements whose displacement is given by the input vector.

    """
    CC = []
    shifty = vector[0]
    shiftx = vector[1]
    for det1 in range(25):
        [det1y, det1x] = spad_coord_from_det_numb(det1)
        for det2 in range(25):
            [det2y, det2x] = spad_coord_from_det_numb(det2)
            # only use subset of pixels: central 3x3 or 5x5 region
            distanceFromCenter = np.array([np.abs(det2y-2), np.abs(det2x-2), np.abs(det1y-2), np.abs(det1x-2)])
            if det2y-det1y == shifty and det2x-det1x == shiftx and np.all(distanceFromCenter <= (n-1)/2):
                CC.append('det' + str(det1) + 'x' + str(det2))
    return CC
    
