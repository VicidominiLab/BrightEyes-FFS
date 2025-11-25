import numpy as np


def lifetimehist(data, m_bins=260, laser_freq=80e6, microtime=1e-12):
    """
    Make lifetime histograms of all channels

    Parameters
    ----------
    data : object
        data object with macro and microtimes in ps.
    m_bins : int, optional
        number of microtime bins. The default is 260.
    laser_freq : float, optional
        laser frequency (Hz). The default is 80e6.

    Returns
    -------
    nothing, in-place change of data object

    """
    
    # get list of detector fields
    listOfFields = list(data.__dict__.keys())
    listOfFields = [i for i in listOfFields if i.startswith('det')]
    
    for i, det in enumerate(listOfFields):
        
        # macrotimes
        macroTime = getattr(data, det)[:,0] # ps
        
        # calculate proper microtimes
        microTime = getattr(data, det)[:,1]
        microTime = np.mod(microTime, 1 / microtime / laser_freq)
        microTime = -microTime + np.max(microTime)
        
        # make histogram of microtimes
        [Ihist, lifetimeBins] = np.histogram(microTime, m_bins)
        lifetimeBins = lifetimeBins[0:-1]
        
        # store histogram
        setattr(data, "hist" + str(det), np.transpose(np.stack((lifetimeBins, Ihist))))
        setattr(data, "det" + str(det), np.transpose([macroTime, microTime]))
        