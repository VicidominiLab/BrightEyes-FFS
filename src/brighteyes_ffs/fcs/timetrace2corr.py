# -*- coding: utf-8 -*-

import numpy as np

def tt2corr(data1, data2, macro_time=1, m=50):
    """
    Calculate the auto/cross-correlation function for log spaced tau values using fft

    Parameters
    ----------
    data1 : np.ndarray
        1D array with intensity as a function of time.
    data2 : np.ndarray
        1D array with intensity as a function of time.
        For autocorr: data2 = data1
    macro_time : double, optional
        Dwell time. The default is 1.
    m : int, optional
        Number of lag times for which to calculate G. The default is 50.

    Returns
    -------
    output : np.ndarray
        2D array with first column lag times, second column G.

    """
    G = autocorrelation_wiener_khinchin(data1, data2, macro_time=macro_time)
    lags = G[:, 0]
    
    # Generate log-spaced indices for positive lags
    positive_lags = lags[lags > 0]
    log_indices = np.logspace(0, np.log10(int(len(positive_lags) / 2 - 1)), num=int(m), dtype=int)
    log_indices = [0] + list(np.unique(log_indices))  # Ensure indices are unique
    
    # Select log-spaced lags and autocorrelation values
    log_lags = lags[log_indices]
    log_autocorr = G[log_indices, 1]
    return np.transpose([log_lags, log_autocorr])
    

def autocorrelation_wiener_khinchin(data1, data2, macro_time=1.0):
    """
    Calculate the autocorrelation function using the Wiener-Khinchin theorem.

    Parameters
    ----------
    data1 : np.ndarray
        1D array with intensity as a function of time.
    data2 : np.ndarray
        1D array with intensity as a function of time.
    macro_time : double, optional
        Dwell time. The default is 1.

    Returns
    -------
    output : np.ndarray
        2D array with first column lag times, second column G.

    """
    
    # Number of points in the signal
    n = len(data1)
    
    # Perform FFT of the signal
    
    data1_pad = np.concatenate((data1, np.zeros(len(data1)-1)))
    data2_pad = np.concatenate((data2, np.zeros(len(data2)-1)))
    
    fft1 = np.fft.fft(data1_pad)  # Zero-pad to 2n for better resolution
    fft2 = np.fft.fft(data2_pad)
    
    psd = fft1 * np.conjugate(fft2)
    
    corr = np.fft.ifft(psd).real
    g = corr[:(corr.size//2)+1]
    
    # Normalize each lag by the number of overlapping points
    n = len(g)  # Original signal length (before padding)
    overlap = np.arange(1, n + 1)  # Overlap for positive lags
    g /= overlap[::-1]  # Normalize positive lags
    g /= np.mean(data1)
    g /= np.mean(data2)
    g -= 1
    
    # Generate time lags
    n = len(g)
    tau = np.linspace(0, n-1, n) * macro_time
    
    
    #g = np.concatenate((corr[n:], corr[:n]))
    
    return np.transpose([tau, g])