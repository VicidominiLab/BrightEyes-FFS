import numpy as np
from ..tools.savevar import savevar


class DataObj:
    pass

def read_asc(fname, store_pickle=False):
    """
    Read an ascii BH fcs file which contains the autocorrelation curve and
    return this curve

    Parameters
    ----------
    fname : string
        *asc file name.
    storePickle : boolean, optional
        Store data in pickle file. The default is False.

    Returns
    -------
    data
        Object containing information about the fcs file.

    """
    
    # OPEN FILE
    print("Opening .asc file.")
    with open(fname, mode='r') as file:
        rawdata = file.read()
    print("File opened.")
    
    # CREATE DATA OBJECT
    data = DataObj()

    # SEARCH fcs VALUE
    start = rawdata.find("FCS_value")
    if start == -1:
        # no fcs data found
        pass
    else:
        start = rawdata.find("\n", start) + 1
        stop =  rawdata.find("*END", start)
        Npoints = rawdata.count("\n", start, stop)
        G = np.zeros([Npoints, 2])
        Gn = np.zeros([Npoints, 2])
        for i in range(Npoints):
            stop = rawdata.find(" ", start)
            tp = float(rawdata[start:stop])
            start = stop + 1
            stop = rawdata.find("\n", start)
            Gp = float(rawdata[start:stop])
            G[i, 0] = tp
            G[i, 1] = Gp
            Gn[i, 0] = 1e-6 * tp
            Gn[i, 1] = Gp - 1
            start = stop + 1
        data.G = G
        data.Gn = Gn
        
    
    if store_pickle:
        print("Storing .pickle file")
        savevar(data, fname[0:-4] + "_BHdata")
        print(".pickle file stored")
    
    return data
