import numpy as np


def calcdist_from_coord(data, coord=None):
    """
    Calculate the distance from a fixed point to every cell of an array

    Parameters
    ----------
    data : np.array
        2d np.array (only the dimensions of the array are needed, the
        contents of the array can be whatever).
    coord : list, optional
        [idy, idx] indices of the data point from which to calculate
        the distance, default is center. The default is None.

    Returns
    -------
    dist : np.array
        2d np.array with for each cell the distance from that cell to
        the given coordinates.

    """

    dataShape = np.shape(data)
    
    Ny = dataShape[0]
    Nx = dataShape[1]

    if coord == None:
        # use center coordinate
        coord = [int(np.floor(Ny/2)), int(np.floor(Nx/2))]
    
    x = np.linspace(0, Nx-1, Nx)
    y = np.linspace(0, Ny-1, Ny)
    
    xv, yv = np.meshgrid(x, y)
    
    yv -= coord[0]
    xv -= coord[1]
    
    dist = np.sqrt(xv**2 + yv**2)

    return dist


def list_of_pixel_pairs_at_distance(dist, N=25, pixelsOff=[]):
    """
    Return list of pixel pairs located at a given shift vector from each other

    Parameters
    ----------
    dist : list
        two numbers indicating the shift in y and x.
    N : int, optional
        Total number of pixels in the square array. The default is 25.
    pixelsOff : list
        List of pixel numbers that are considered bad and should be removed from the list.

    Returns
    -------
    listPairs : list
        list of pixel pairs fullfilling the requirements.

    """
    listPairs = []
    for det1 in range(N):
        for det2 in range(N):
            if calc_shift_from_coord(det1, det2, int(np.sqrt(N))) == dist and det1 not in pixelsOff and det2 not in pixelsOff:
                listPairs.append([det1, det2])
    return listPairs


def calc_shift_from_coord(det1, det2, N=5):
    """
    Calculate shift vector between two detector elements

    Parameters
    ----------
    det1 : int
        pixel number.
    det2 : int
        second pixel number.
    N : int, optional
        Number of pixels in each direction of the array. The default is 5.

    Returns
    -------
    list
        two numbers with the y and x shift between the two pixels in arb. units.

    """
    # positive shifts are to the right and down
    row1 = np.floor(det1 / N)
    row2 = np.floor(det2 / N)
    col1 = np.mod(det1, N)
    col2 = np.mod(det2, N)
    
    deltay = int(row2 - row1)
    deltax = int(col2 - col1)
    
    return [deltay, deltax]