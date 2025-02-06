import numpy as np
from ..tools.phasor import phasor


def data2phasor(data):
    """
    Calculate phasors for all histograms in data object

    Parameters
    ----------
    data : object
        Object with histograms for each channel.
        E.g. data.hist1 = np.array(N x 2)

    Returns
    -------
    data : object
        Same object with phasors included.

    """
    
    histList = [i for i in list(data.__dict__) if i.startswith('hist')]
    N = len(histList)
    
    phasors = np.zeros(N, dtype=complex)
    
    for i in range(N):
        hist = getattr(data, histList[i])
        hist = hist[:,1]
        phasors[i] = phasor(hist)
    
    data.phasors = phasors
    
    return data