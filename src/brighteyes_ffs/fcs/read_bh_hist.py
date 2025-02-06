import numpy as np
from ..tools.savevar import savevar


class DataObj:
    pass

def read_hist(fname, store_pickle=False):
    """
    Read an ascii BH histogram file

    Parameters
    ----------
    fname : string
        *asc file name.
    storePickle : boolean, optional
        store data in pickle file. The default is False.

    Returns
    -------
    histogr : np.array
        np.array with histogram.

    """
    
    # OPEN FILE
    print("Opening .asc file.")
    with open(fname, mode='r') as file:
        rawdata = file.read()
    print("File opened.")
    
    # CREATE DATA OBJECT
    data = DataObj()

    # SEARCH fcs VALUE
    start = rawdata.find("*BLOCK ")
    if start == -1:
        # no fcs data found
        pass
    else:
        start = rawdata.find("\n", start) + 1
        stop =  rawdata.find("*END", start)
        Npoints = rawdata.count("\n", start, stop)
        histogr = np.zeros(Npoints)        
        for i in range(Npoints):
            stop = rawdata.find("\n", start)
            I = float(rawdata[start:stop])
            histogr[i] = I
            start = stop + 1
        
    if store_pickle:
        print("Storing .pickle file")
        savevar(data, fname[0:-4] + "_BHdata")
        print(".pickle file stored")
    
    return histogr
