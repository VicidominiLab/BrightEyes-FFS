import numpy as np
import re
from .extract_spad_data_kw import keyword_2_ch


def extract_spad_photon_streams(data, mode):
    """
    Extract data from specific detector elements from SPAD-fcs time-tagging data

    Parameters
    ----------
    data : object
        Object with for each detector element a 2D array [t, m, F1, F2]
        data.det0, data.det1, etc. with
        - t: vector with macrotimes photon arrivals (ps)
        - m: microtimes (ps)
        - F1, F2, ... filter weights (based on microtimes), if any.
    mode : string
        Either
        - "C3+4+7+12" to custom sum over channels 3, 4, 7 and 12
        - int: to extract photons from a single channel
        - string with a predefined keyword, such as "sum3"

    Returns
    -------
    datasum : np.array()
        2D array with concatenated and sorted detector elements data.

    """
    
    if isinstance(mode, str):
        if mode[0] == 'C':
            # list of channels to be summed, e.g. "C1+3+12+42"
            listOfFields = ['det' + str(i) for i in re.findall(r'\d+', mode)]
        else:
            # get list from dictionary
            listOfFields = keyword_2_ch[mode]
            listOfFields  = ['det' + str(i) for i in listOfFields]
    else:
        # number given
        listOfFields = ['det' + str(mode)]
   
    # concatenate all photon streams
    for j in range(len(listOfFields)):
        if j == 0:
            datasum = getattr(data, listOfFields[j])
        else:
            datasum = np.concatenate((datasum, getattr(data, listOfFields[j])))
    
    # sort all photon streams
    datasum = datasum[datasum[:,0].argsort()]
    
    # remove photon pairs with the same macrotime
    [dummy, ind] = np.unique(datasum[:,0], return_index=True)
    datasum = datasum[ind,:]
    
    return datasum
