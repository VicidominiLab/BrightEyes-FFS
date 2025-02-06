import numpy as np
from ..tools.find_nearest import find_nearest

def g2difftime(g, smoothing=5):
    """
    Estimate diffusion time from fcs correlation curve without fitting
    
    Parameters
    ----------
    g : 2D np.array()
        Array with [tau, G] values.
    smoothing : int
        Number indicating moving average window for smoothing

    Returns
    -------
    t : float
        estimated diffusion time (same units as input)
    Gs : float
        correlation value at the diffusion time
    
    """
    # smooth function
    Gsmooth = np.convolve(g[1:,1], np.ones(smoothing)/smoothing, mode='valid')
    tausmooth = np.convolve(g[1:,0], np.ones(smoothing)/smoothing, mode='valid')
    
    # calculate derivative
    Gdiff = Gsmooth[1:] - Gsmooth[0:-1]
    Gdiff *= -1
    
    # normalized cumulative sum
    Gdcum = np.cumsum(Gdiff)
    Gdcum /= np.max(Gdcum)
    
    # now G is a function that starts at 0 and goes to 1. Diff time can be
    # estimated as tau such that G(tau) = 0.5
    [dummy, idx] = find_nearest(Gdcum, 0.5)
    
    return tausmooth[idx], Gsmooth[idx]