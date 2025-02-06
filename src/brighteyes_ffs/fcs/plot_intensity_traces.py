import matplotlib.pyplot as plt
import numpy as np
from ..tools.bindata import bindata


def plot_intensity_traces(data, dwell_time=1, binsize=1, yscale='log', list_of_channels="all"):
    """
    Plot intensity traces from SPAD-fcs data

    Parameters
    ----------
    data : object
        Data variable, i.e. output from binfile2data.
    dwell_time : scalar, optional
        Bin time [in µs]. The default is 1.
    binsize : scalar, optional
        Number of data points in one bin. The default is 1.
    yscale : string, optional
        "linear" or "log" y scale. The default is 'log'.
    list_of_channels : list of numbers or string, optional
        List with channel numbers to be plotted, e.g. [0, 12, 24]. The default is "all".

    Returns
    -------
    h : figure
        plot with all intensity traces.

    """

    # number of channels
    Nc = np.size(data, 1)

    # channels to plot
    if list_of_channels == "all":
        list_of_channels = list(range(np.min([Nc, 25])))
    
    # bin data
    databin = bindata(data, binsize)
    # number of time points in binned data
    Nt = np.size(databin, 0)
    
    # create figure
    leg = []  # figure legend
    h = plt.figure()
    
    # bin time
    binTime = 1e-6 * dwell_time * binsize
    
    # time vector
    time = list(range(0, Nt))
    time = [i * binTime for i in time]
    
    ymax = 0
    for i in list_of_channels:
        # rescale intensity values to frequencies
        PCR = databin[:, i] / binTime
        # for lag plot in Hz, else in kHz
        scaleFactor = 1000
        if yscale == 'log':
            scaleFactor = 1
        PCRscaled = PCR / scaleFactor
        plt.plot(time, PCRscaled)
        leg.append('Pixel ' + str(i))
        ymax = np.max([ymax, np.max(PCRscaled)])
    
    plt.xlabel('Time [s]')
    plt.xlim([0, 2*time[-1] - time[-2]])
    if yscale == 'linear':
        plt.ylim([0, 1.1 * ymax])
    if scaleFactor == 1000:
        plt.ylabel('Photon count rate [kHz]')
    else:
        plt.ylabel('Photon count rate [Hz]')
    if len(list_of_channels) < 10:
        plt.legend(leg)
    plt.rcParams.update({'font.size': 20})
    plt.tight_layout()
    plt.yscale(yscale)

    return h