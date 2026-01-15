import numpy as np


def distance_between_elements(detector, elem1, elem2):
    """
    Calculate the distance between two SPAD detector elements in a.u.
    (nearest neighbours 1 unit)

    Parameters
    ----------
    detector : str
        Detector type: either 'airyscan', 'luminosa', or 'nsparc'
    elem1 : int
        Number of the first detector element.
    elem1 : int
        Number of the second detector element.

    Returns
    -------
    distance : float
        Distance between the two elements.

    """
    
    pos1 = detector_element_coordinates(detector, element=elem1)
    pos2 = detector_element_coordinates(detector, element=elem2)
    
    distance = np.sqrt(np.sum((pos2 - pos1)**2))
    
    return distance


def detector_element_coordinates(detector, element=None):
    """
    Return the xy coordinates of all detector elements for a given
    system.
    

    Parameters
    ----------
    detector : str
        Detector type: either 'airyscan', 'luminosa', or 'nsparc'
    element : int, optional
        Either None: return all coordinates for all elements
        Or int: return the xy coordinates of a single element only
    
        E.g. detector_element_coordinates('luminosa', element=20) returns
        [0, 0], i.e. the coordinates of the central element

    Returns
    -------
    sx : np.array()
        2D array with xy coordinates

    """
    
    detector = str.lower(detector)
    
    if detector == 'airyscan':
        
        #          22  21
        #     23  10  9   8  20
        #   24  11  2   1   7  19
        #     12  3   0   6  18  31
        #   25  13  4   5  17  30
        #     26 14  15  16  29
        #          27  28
    
        n_elements= 32
        
        shiftx = np.cos(30*np.pi/180)
        sx = np.zeros((n_elements,2))
        sx[0] = [0,0]
    
        sx[1] = [shiftx,1.5]
        sx[2] = [-shiftx,1.5]
        sx[3] = [-2*shiftx,0]
        sx[4] = [-shiftx,-1.5]
        sx[5] = [shiftx,-1.5]
        sx[6] = [2*shiftx,0]
    
        sx[7] = [3*shiftx,1.5]
        sx[8] = [2*shiftx,3]
        sx[9] = [0,3]
        sx[10] = [-2*shiftx,3]
        sx[11] = [-3*shiftx,1.5]
        sx[12] = [-4*shiftx,0]
        sx[13] = [-3*shiftx,-1.5]
        sx[14] = [-2*shiftx,-3]
        sx[15] = [0,-3]
        sx[16] = [2*shiftx,-3]
        sx[17] = [3*shiftx,-1.5]
        sx[18] = [4*shiftx,0]
    
        sx[19] = [5*shiftx,1.5]
        sx[20] = [4*shiftx,3]
        sx[21] = [shiftx,4.5]
        sx[22] = [-shiftx,4.5]
        sx[23] = [-4*shiftx,3]
        sx[24] = [-5*shiftx,1.5]
        sx[25] = [-5*shiftx,-1.5]
        sx[26] = [-4*shiftx,-3]
        sx[27] = [-shiftx,-4.5]
        sx[28] = [shiftx,-4.5]
        sx[29] = [4*shiftx,-3]
        sx[30] = [5*shiftx,-1.5]
        sx[31] = [6*shiftx,0]
        
        sx /= np.sqrt(3)
        
    elif detector in {"luminosa", "pda-23", "pda23"}:
        
        # 31  30  29  28  27
        #   26  25  24  23
        # 22  21  20  19  18
        #   17  16  15  14
        # 13  12  11  10  09
    
        n_elements= 32
        shiftx = np.cos(30*np.pi/180)
        
        sxx  = [4.0, 2.0, 0.0, -2.0, -4.0, 3.0, 1.0, -1.0, -3.0, 4.0,
               2.0, 0.0, -2.0, -4.0, 3.0, 1.0, -1.0, -3.0, 4.0, 2.0,
               0.0, -2.0, -4.0]
        
        syy = [-4*shiftx, -4*shiftx, -4*shiftx, -4*shiftx, -4*shiftx, -2*shiftx, -2*shiftx, -2*shiftx,
              -2*shiftx, 0.0, 0.0, 0.0, 0.0, 0.0, 2*shiftx, 2*shiftx, 2*shiftx, 2*shiftx,
              4*shiftx, 4*shiftx, 4*shiftx, 4*shiftx, 4*shiftx]
      
        
        sx = np.zeros((n_elements,2))
        sx[9:,0] = sxx
        sx[9:,1] = syy
        sx /= 2
        
    elif detector == 'nsparc' or detector == '5x5':
        
        #  0  1  2  3  4
        #  5  6  7  8  9
        # 10 11 12 13 14
        # 15 16 17 18 19
        # 20 21 22 23 24

        n_elements= 25

        sx = np.zeros((n_elements, 2))
        sx[:,0] = [np.mod(i, 5) for i in range(n_elements)]
        sx[:,1] = [i//5 for i in range(n_elements)]
        sx -= 2
        sx[:,1] *= -1

    else:
        return None
    
    if element is None:
        return sx
    else:
        return sx[element]
    