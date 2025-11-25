import numpy as np
from ..tools.moving_average import moving_average
import copy


def align_hist(data, n=15):
    """
    Align lifetime histograms of different channels. Calculate moving averaged
    histograms, caculate derivative and find peak. Then shift everything
    so that the peaks are aligned.

    Parameters
    ----------
    data : object
        data object with microtime histograms.
    n : int, optional
        width of the window for the calculation of the moving average. The default is 15.

    Returns
    -------
    Object is changed in-place. Nothing is returned

    """
    
    histList = [i for i in list(data.__dict__.keys()) if i.startswith('hist')]
    
    hshifts = np.zeros(len(histList))
    
    i = 0
    for hist in histList:
        # get histogram
        histD = copy.deepcopy(getattr(data, hist))
        # calculate moving average
        IAv = moving_average(histD[:,1], n=n)
        # calculate derivative
        derivative = IAv[1:] - IAv[0:-1]
        # find maximum of derivative
        maxInd = np.argmax(derivative)
        # shift histogram
        histD[:,1] = np.roll(histD[:,1], -maxInd)
        # store shifted histogram in data object
        setattr(data, 'A' + hist, histD)
        # store shift value
        hshifts[i] = maxInd
        i += 1
    setattr(data, 'hshifts', hshifts)
