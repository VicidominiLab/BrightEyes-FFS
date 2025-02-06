import numpy as np


def lifetimehist(data, m_bins=260, laser_freq=80e6):
    """
    Make lifetime histograms of all channels

    Parameters
    ----------
    data : object
        data object with macro and microtimes.
    m_bins : int, optional
        number of microtime bins. The default is 260.
    laser_freq : float, optional
        laser frequency (Hz). The default is 80e6.

    Returns
    -------
    data : object
        same data object as input, but with histograms added.

    """
    
    # get list of detector fields
    listOfFields = list(data.__dict__.keys())
    listOfFields = [i for i in listOfFields if i.startswith('det')]
    Ndet = len(listOfFields)
    
    for det in range(Ndet):
        
        # macrotimes
        macroTime = getattr(data, "det" + str(det))[:,0] # ps
        
        # calculate proper microtimes
        microTime = getattr(data, "det" + str(det))[:,1]
        microTime = np.mod(microTime, 1 / data.microtime / laser_freq)
        microTime = -microTime + np.max(microTime)
        
        # make histogram of microtimes
        [Ihist, lifetimeBins] = np.histogram(microTime, m_bins)
        lifetimeBins = lifetimeBins[0:-1]
        
        # store histogram
        setattr(data, "hist" + str(det), np.transpose(np.stack((lifetimeBins, Ihist))))
        setattr(data, "det" + str(det), np.transpose([macroTime, microTime]))
        
    return data
