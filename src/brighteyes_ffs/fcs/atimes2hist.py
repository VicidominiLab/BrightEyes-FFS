from .atimes_data import atimes_data_2_channels, atimes_data_2_duration
import numpy as np

def atimes_data_2_hist(data, bintime='auto', ttm_correction=False, laser_freq=None):
    """
    Calculate arrival time histograms

    Parameters
    ----------
    data : Correlations object
        Output from data=load_atimes_data(fname) with fields data.det0, data.det1, etc.
        Each field is a 2D np.array() with macrotimes and microtimes in ps
    bintime : float, optional
        Bin time for the histograms in ps.
        The default is 'auto', which uses the minimum granularity.
    ttm_correction : boolean, optional
        True for TTM data, False otherwise.
    laser_freq : float, optional
        Laser frequemcy. Only needed for TTM data. The default is None.

    Returns
    -------
    in-place function. Changes the input object directly without returning it

    """
    
    all_ch = atimes_data_2_channels(data)
    
    if bintime == 'auto':
        data_single_ch = getattr(data, all_ch[0])
        bintime = np.min(data_single_ch[data_single_ch>0])
        
    max_microtime = 1e12 * atimes_data_2_duration(data, 1e-12, subtract_start_time=False, return_period=True) # ps
    
    for det in all_ch:
        microtime = getattr(data, det)[:,1]
        if ttm_correction:
            microtime = np.mod(microtime, 1e12 / laser_freq)
            microtime = -microtime + np.max(microtime)
        tcspc_bins = np.arange(0, max_microtime, bintime)
        [Ihist, lifetime_bins] = np.histogram(microtime, tcspc_bins)
        setattr(data, "hist" + det[3:], np.transpose(np.stack((lifetime_bins[0:-1], Ihist))))
        