import numpy as np
import os
from numpy import genfromtxt
from .checkfname import checkfname

def csv2array(file, dlmt='\t'):
    """
    Import .csv file into 2D array

    Parameters
    ----------
    file : string
        file name.
    dlmt : string, optional
        delimiter. The default is '\t'.

    Returns
    -------
    data : np.array
        data array.

    """
    
    if dlmt == -1:
        #guess delimiter
        with open(file) as f:
            if '\t' in f.read():
                dlmt = '\t'
            else:
                dlmt = ','
    file = checkfname(file, 'csv')
    if os.path.getsize(file) == 0:
        return []
    data = genfromtxt(file, delimiter=dlmt)
    if type(data) is not np.ndarray or data.ndim < 1:
        # if it is a single number -> convert to array
        try:
            data = np.atleast_1d(data)
        except:
            pass
    return data


def array2csv(data, fname='test.csv', dlmt='\t', dtype=float):
    """
    Store 2D array into .csv file

    Parameters
    ----------
    data : np.array
        data array.
    fname : string, optional
        file name. The default is 'test.csv'.
    dlmt : string, optional
        delimiter. The default is '\t'.
    dtype : data type, optional
        data type. The default is float.

    Returns
    -------
    None.

    """
    
    data = np.asarray(data, dtype)
    fname = checkfname(fname, 'csv')
    if dtype == int:
        np.savetxt(fname, data, delimiter=dlmt, fmt='%i')
    else:
        np.savetxt(fname, data, delimiter=dlmt)