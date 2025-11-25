import numpy as np
import matplotlib.pyplot as plt
from .atimes_data import atimes_data_2_channels


def filter_ap(ks, tau, T, b, norm_exp=False, plot_fig=True):
    """
    Use fluorescence lifetime to generate filters that remove afterpulsing
    from fcs data, based on ideal exponential decay
    Based on Enderlein and Gregor, Rev. Sci. Instr., 76, 2005 and
    Kapusta et al., J. Fluor., 17, 2007.
    
    Function call examples:
        F = filter_ap(0, 23.78, T, 0.028)

    Parameters
    ----------
    ks : scalar
        Parameter describing STED decay rate (ks = 0 for no STED).
    tau : scalar
        Fluorescence lifetime from exponential fit of the histogram.
    T : scalar
        Number of data points in time.
    b : scalar
        Background value from exponential fit of the histogram.
    norm_exp : boolean, optional
        Normalize the histogram of the exponential decay by setting
        the sum of the bins to 1. For a full decay, this is always
        the case, since the integral of (1/tau * exp(-t/tau)) running
        from 0 to inf is 1.. The default is False.
    plot_fig : boolean, optional
        Plotting. The default is True.

    Returns
    -------
    F : 2D numpy array
        2D array with 2 rows, one row for each filter function.

    """

    t = np.linspace(0, T-1, int(T))

    # histogram 1 with the fluorescence decay
    decayNorm = (1/tau) * np.exp(-t/tau) / (1 + 0.5 * ks * t / tau)
    if norm_exp:
        decayNorm = decayNorm / np.sum(decayNorm)
    
    # histogram 2 with the background, always normalized
    decayNorm2 = np.ones((1,T)) / T
    
    # concatenate both row vectors vertically
    decayNorm = np.concatenate((np.transpose(decayNorm[:,np.newaxis]), decayNorm2), axis=0)
    
    # total average decay as expected from theory
    ItheorTOT = b + np.exp(-t/tau) / (1 + 0.5 * ks * t / tau)
    
    # matrix D as in Enderlein paper
    D = np.diag(1 / ItheorTOT)
    
    # filter formula
    F = np.matmul(np.matmul(decayNorm, D), np.transpose(decayNorm))
    F = np.linalg.inv(F)
    F = np.matmul(np.matmul(F, decayNorm), D)
    
    if plot_fig:
        plt.figure()
        plt.plot(F[0,:])
        plt.plot(F[1,:])
    
    return F


def filter_2c(hist1, hist2, b, plot_fig=True):
    """
    Use histogram information to generate filters to split the data in two
    time traces
    Function call examples:
        F = filter_ap(histFluor, np.array(ones(500)))

    Parameters
    ----------
    hist1 : 1D numpy array
        Vector with the histogram values for component 1
        (can be theoretical or experimental curve
         can - but does not have to - be normalized.
    hist2 : 1D numpy array
        Vector with the histogram values for component 2
        (can be theoretical or experimental curve
         can - but does not have to - be normalized..
    b : TYPE
        Number, scaling factor describing the weight of hist2
        with respect to hist1
        equal to amplitude(hist2) / amplitude(hist1) for unnormalized
        histograms.
    plot_fig : boolean, optional
        Plot figure. The default is True.

    Returns
    -------
    F : 2D numpy array
        2D array with 2 rows, one row for each filter function.

    """
    
    # normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # concatenate both row vectors vertically
    histAll = np.concatenate((np.transpose(hist1[:,np.newaxis]), np.transpose(hist2[:,np.newaxis])), axis=0)
    
    # total histogram
    histTot = hist1 / np.max(hist1) + b * hist2 / np.max(hist2)
    
    # matrix D as in Enderlein paper
    D = np.diag(1 / histTot)
    
    # filter formula
    F = np.matmul(np.matmul(histAll, D), np.transpose(histAll))
    F = np.linalg.inv(F)
    F = np.matmul(np.matmul(F, histAll), D)
    
    if plot_fig:
        plt.figure()
        plt.plot(F[0,:])
        plt.plot(F[1,:])
    
    return F


def filter_range(data, n_ch=25):
    """
    Calculate a reasonable filter range for the ACF.

    Parameters
    ----------
    data : correlations object
        output from load_atimes_data
        contains also fields histx with x the channel number containing the microtime histograms

    Returns
    -------
    None.

    """
    # find histogram peak and use only limited number of bins after it
    
    all_ch = atimes_data_2_channels(data)
    
    fit_range = np.zeros((len(all_ch), 2), dtype='int')
    for i, det in enumerate(all_ch):
        IhistSingle = getattr(data, "hist" + str(det[3:]))
        Ihist = IhistSingle[:, 1]
        idxStart = np.where(Ihist == np.max(Ihist))[0][0] + 1
        idxStop = np.where(Ihist[idxStart:] == np.min(Ihist[idxStart:]))[0][0] + idxStart + 1
        idxStop = np.minimum(idxStop, len(Ihist) - 1)
        fit_range[i, :] = [idxStart, idxStop]
    data.fit_range = fit_range
    