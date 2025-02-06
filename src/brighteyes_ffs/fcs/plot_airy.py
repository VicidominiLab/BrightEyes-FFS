import matplotlib.pyplot as plt
import numpy as np
from .distance2detelements import distance2detelements
from ..tools.cast_data import cast_data
from brighteyes_ism.simulation.detector import hex_grid

def plot_fingerprint_airyscan(counts, plot=False):
    """
    Plot the airyscan fingerprint from a data set with the photon counts for
    each of the 32 detector elements

    Parameters
    ----------
    counts : np.array
        Array with the 32 photon counts of the airyscan.
        Channel numbers according to Zeiss (ch 0 = center)
    plot : boolean, optional
        Plot the finger print

    Returns
    -------
    hexb : np.array()
        Array for hexbin plotting

    """
    
    N = 11  # Number of elements
    s2 = hex_grid(N, np.linspace(0, N - 1, N))
    
    hexb = np.zeros((66))
    
    hexb[0:19] = 0
    hexb[20] = counts[27]
    hexb[21] = counts[28]
    hexb[22] = 0
    hexb[23] = 0
    hexb[24] = counts[26]
    hexb[25] = counts[14]
    hexb[26] = counts[15]
    hexb[27] = counts[16]
    hexb[28] = counts[22]
    hexb[29] = 0
    hexb[30] = counts[25]
    hexb[31] = counts[13]
    hexb[32] = counts[4]
    hexb[33] = counts[5]
    hexb[34] = counts[17]
    hexb[35] = counts[30]
    hexb[36] = counts[12]
    hexb[37] = counts[3]
    hexb[38] = counts[0]
    hexb[39] = counts[6]
    hexb[40] = counts[18]
    hexb[41] = counts[31]
    hexb[42] = counts[24]
    hexb[43] = counts[11]
    hexb[44] = counts[2]
    hexb[45] = counts[1]
    hexb[46] = counts[7]
    hexb[47] = counts[19]
    hexb[48] = counts[23]
    hexb[49] = counts[10]
    hexb[50] = counts[9]
    hexb[51] = counts[8]
    hexb[52] = counts[20]
    hexb[53] = 0
    hexb[54] = 0
    hexb[55] = 0
    hexb[56] = counts[22]
    hexb[57] = counts[21]
    hexb[58:65] = 0
    
    if plot:
        fig, ax = plt.subplots()
        ax.set_facecolor("black")
        plt.hexbin(s2[1], s2[0], C=hexb, gridsize=[6,5], cmap="inferno", linewidths=0.5, edgecolors="k")
        plt.colorbar(label="Cell Value")
        plt.xlim([-0.5,5.5])
        plt.ylim([2,8.5])
        plt.xticks([])
        plt.yticks([])
        ax.set_box_aspect(1)
    return s2, hexb

def plot_airy(data, show_perc=True, dtype='int64', normalize=False, savefig=0, vminmax = 'auto'):
    """
    Make finger print plot of SPAD-fcs data with 25 channels.

    Parameters
    ----------
    data : np.array()
        Nx26 or Nx25 array with the fcs data
        or data object with data.det0 etc. arrival times.
    show_perc : boolean, optional
        Show percentages. The default is True.
    dtype : string, optional
        Data type. The default is 'int64'.
    normalize : boolean, optional
        Convert total counts to average counts per bin if True. The default is False.
    savefig : int, optional
        Path to store figure. The default is 0.
    vminmax : vector or string, optional
        Vector with minimum and maximum color bar value. The default is 'auto'.

    Returns
    -------
    airy : np.array()
        26 element vector with the sum of the rows and plot.

    """
    
    if type(data) == np.ndarray:
        # data is numpy array with intensity traces
        if len(np.shape(data)) > 1:
            # if 2D array, convert to dtype and sum over all rows
            data = cast_data(data, dtype)
            airy = np.sum(data, axis=0)
        else:
            airy = data
        airy2 = airy[0:25]
    else:
        if hasattr(data, 'det24'):
            # data is fcs2arrivaltimes.ATimesData object with 25 elements
            airy2 = np.zeros(25)
            for det in range(25):
                airy2[det] = len(getattr(data, 'det' + str(det)))
        else:
            # data is fcs2arrivaltimes.ATimesData object with 21 elements
            airy2 = np.zeros(25)
            # *  0  1  2  *
            # 3  4  5  6  7
            # 8  9  10 11 12
            # 13 14 15 16 17
            # *  18 19 20 *
            dets = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23]
            for det in range(21):
                if hasattr(data, 'det' + str(det)):
                    airy2[dets[det]] = len(getattr(data, 'det' + str(det)))
                else:
                    airy2[dets[det]] = 0
        airy = airy2
    
    if normalize:
        airy2 = airy2 / np.size(data, 0)
    
    airyMax = np.max(airy2)
    airyMin = np.min(airy2)
    airyCentPerc = (0.2 * (airyMax - airyMin) + airyMin) / airyMax * 100
        
    airy2 = airy2.reshape(5, 5)
    
    
    plt.figure()
    fontSize = 20
    plt.rcParams.update({'font.size': fontSize})
    plt.rcParams['mathtext.rm'] = 'Arial'
    
    if vminmax == 'auto':
        plt.imshow(airy2, cmap='hot', interpolation='nearest')
    else:
        plt.imshow(airy2, cmap='hot', interpolation='nearest', vmin=vminmax[0], vmax=vminmax[1])
    ax = plt.gca()
    
    # Major ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Labels for major ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Minor ticks
    ax.set_xticks(np.arange(-0.5, 4.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5.5, 1), minor=True)
    # Gridlines based on minor ticks
    #ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontSize)

    if type(show_perc) is str and show_perc=="numbers":
        for i in range(5):
            for j in range(5):
                if vminmax == 'auto':
                    perc = round(airy2[i, j] / airyMax * 100)
                else:
                    perc = round(airy2[i, j] / vminmax[1] * 100)
                c="k"
                if perc < airyCentPerc:
                    c="w"
                plt.text(j, i, '{:.1f}'.format(airy2[i, j]), ha="center", va="center", color=c, fontsize=18)    
    elif show_perc:
        for i in range(5):
            for j in range(5):
                if vminmax == 'auto':
                    perc = round(airy2[i, j] / airyMax * 100)
                else:
                    perc = round(airy2[i, j] / vminmax[1] * 100)
                c="k"
                if perc < airyCentPerc:
                    c="w"
                plt.text(j, i, '{:.0f}%'.format(perc), ha="center", va="center", color=c, fontsize=18)

    if savefig != 0:
        plt.tight_layout()
        if savefig[-3:] == 'svg':
            plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(savefig, format=savefig[-3:])

    return airy

def plot_det_dist():
    det = []
    for i in range(25):
        det.append(distance2detelements(i, 12))
    det = np.resize(det, (5, 5))
    plt.figure()
    plt.imshow(det, cmap='viridis')