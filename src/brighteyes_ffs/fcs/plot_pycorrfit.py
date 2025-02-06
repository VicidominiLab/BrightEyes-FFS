import matplotlib.pyplot as plt
import numpy as np
import ntpath
from pathlib import Path
from os import  getcwd

from ..tools.plot_colors import plot_colors
from ..tools.csv2array import csv2array
from ..tools.find_nearest import find_nearest
from ..tools.list_files import list_files


class PyCorrfit_data():
    pass


def plot_pycorr_all(folderName=[]):
    if folderName == []:
        folderName = getcwd()
    folderName = folderName.replace("\\", "/")
    folderName = Path(folderName)
    fileList = list_files(folderName, 'csv')
    data = np.empty((0,3), float)
    # go through each file
    for file in fileList:
        if "chunk" not in file and "Gn" not in file and "sum3_average_fit_results" in file and "SPAD" in ntpath.basename(file):
            # file found
            print(ntpath.basename(file) + " found.")
            if "fit_results" in file:
                # file with fit found
                result = plot_pycorrfit(file, savefig=0)
                data = np.append(data, [[result.Gfitstart, result.tauD, result.chi2]], axis=0)
            else:
                # file with experimental G found
                plot_g_csv(file, savefig=0)
    return data
                
                
def plot_g_csv(file, info="", savefig="png"):
    """
    Plot .csv autocorrelation file

    Parameters
    ----------
    file : string
        .csv file with Nx2 matrix with tau, G.
    info : string, optional
        String with info about the measurement ('central', 'sum3', etc.)
        Needed for the color. The default is "".
    savefig : TYPE, optional
        0 to not save the figure, string with extension "png" or "eps"
        to save in this format. The default is "png".

    Returns
    -------
    data : np.array()
        Nx4 array with tau, G.

    """
    
    if "sum3" in file:
        info = "sum3"
    elif "sum5" in file:
        info = "sum5"
    elif "central" in file:
        info = "central"
    elif "ullr" in file:
        info = "ullr"
    elif "chessboard" in file:
        info = "chessboard"
        
    data = csv2array(file, ',')
    
    tau = data[:,0]
    G = data[:,1]
    taumin = np.min(tau[2:])
    
    plt.figure()
    plt.plot(tau, G, c=plot_colors(info), linewidth=1)
    plt.xscale('log')
    plt.xlim([taumin, 10])
    plt.ylim([np.min(G[2:]),np.max(G[2:])])
    plt.ylabel('G')
    plt.xlabel('tau (s)')
    
    plt.tight_layout()
    
    if savefig != 0:
        plt.savefig(file[0:-3] + savefig, format=savefig)
    
    return data


def plot_pycorrfit(file, info="", savefig=0, plot_tau=True):
    """
    Plot .asc file with PyCorrFit data

    Parameters
    ----------
    file : string
        .csv file with Nx4 matrix with tau, G, Gfit and residuals
        or object format:
            Gdata[:, 0] = tau
            Gdata[:, 1] = G
            Gdata[:, 2] = Gfit
            Gdata[:, 3] = Gres
            data.data = Gdata.
    info : string, optional
        String with info about the measurement ('central', 'sum3', etc.)
        Needed for the color. The default is "".
    savefig : int or string, optional
        0 to not save the figure
        file name with extension to save as "png" or "eps". The default is 0.
    plot_tau : boolean, optional
        Characteristic diffusion time, shown in the plot. The default is True.

    Returns
    -------
    allData : np.array
        Nx4 array with tau, G, Gfit and residuals.

    """
    
    if isinstance(file, str):
        # data given in file format
        if "sum3" in file:
            info = "sum3"
        elif "sum5" in file:
            info = "sum5"
        elif "central" in file:
            info = "central"
        elif "ullr" in file:
            info = "ullr"
        elif "chessboard" in file:
            info = "chessboard"
        allData = read_pycorrfit(file)
    else:
        # data given in object format
        allData = file
    
    tauD = allData.tauD # ms
    data = allData.data
    
    tau = data[:,0]
    G = data[:,1]
    Gfit = data[:,2]
    Gres = data[:,3]
    taumin = np.min(tau)
    taumax = np.min([0.5, np.max(tau)])
    print(taumax)
    
    # plot color
    if type(info) == str and info[0] == "C":
        pcolor = info
    else:
        pcolor = plot_colors(info)
    
    #plt.figure()
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    a0.plot(tau, Gfit, c='k', linewidth=1)
    a0.scatter(tau, G, s=1, c=pcolor)
    a0.set_xscale('log')
    a0.set_xlim([taumin, taumax])
    a0.set_ylabel('G')
    a0.set_xticks([tl for tl in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if tl > taumin and tl < taumax])
    
    # plot tau
    if plot_tau:
        ylim = a0.get_ylim()
        # if only single tau value is given, make tauD a list with 1 element
        if type(tauD) != list:
            tauD = [tauD]
        for i in range(len(tauD)):
            tauD2, idx = find_nearest(tau, 1e-3*tauD[i])
            GtauD2 = Gfit[idx]
            a0.plot([taumin, tauD2], [GtauD2, GtauD2], c='r', linewidth=0.7)
            a0.plot([tauD2, tauD2], [GtauD2, ylim[0]], c='r', linewidth=0.7)
        a0.plot([taumin, taumax], [0, 0], c='k', linewidth=0.7)
        a0.set_ylim(ylim)
    
    # plot residuals
    a1.plot(tau, Gres, c=pcolor, linewidth=1.0)
    a1.set_xlabel(r'$\tau$ (s)')
    a1.set_ylabel('residuals')
    a1.set_xscale('log')
    a1.set_xlim([taumin, taumax])
    a1.plot([taumin, taumax], [0, 0], c='k', linewidth=0.7)
    a1.set_xticks([tl for tl in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if tl > taumin and tl < taumax])
    
    if savefig != 0:
        print("saving figure")
        if savefig[-3:] == 'svg':
            plt.rcParams['svg.fonttype'] = 'none'
        plt.tight_layout()
        plt.savefig(savefig, format=savefig[-3:])
    
    return allData


def read_pycorrfit(file):
    """
    Read header and data of .csv PyCorrFit output file

    Parameters
    ----------
    file : string
        String with path to .csv file.

    Returns
    -------
    outputdata : object
        Object with the tau, G, Gfit, Gres in data field, and separate
        fields for the fitted values n, SP, offset, and chi2.

    """
    
    # create object
    outputdata = PyCorrfit_data()
    # read .csv header
    f = open(file, "r")
    if f.mode == "r":
        contents = f.read()
        start = contents.find("Parameters:")
        [n, start] = read_pycorrfit_singleparam(contents, "#   n\t", "\n", start)
        [tauD, start] = read_pycorrfit_singleparam(contents, "_diff [ms]\t", "\n", start)
        [SP, start] = read_pycorrfit_singleparam(contents, "#   SP\t", "\n", start)
        [offset, start] = read_pycorrfit_singleparam(contents, "#   offset\t", "\n", start)
        start = contents.find("Fitting:", start)
        [chi2, start] = read_pycorrfit_singleparam(contents, "\t", "\n", start)
        [Gfitstart, start] = read_pycorrfit_singleparam(contents, "#   Ival start [ms]\t", "\n", start)
        [Gfitstop, start] = read_pycorrfit_singleparam(contents, "#   Ival end [ms]\t", "\n", start)
        outputdata.n = n
        outputdata.tauD = tauD
        outputdata.SP = SP
        outputdata.offset = offset
        outputdata.chi2 = chi2
        outputdata.Gfitstart = Gfitstart
        outputdata.Gfitstop = Gfitstop
    # load .csv file
    data = csv2array(file)
    # extract data
    tau = data[:,0]
    G = data[:,1]
    Gfit = data[:,2]
    Gres = G - Gfit
    outputdata.data = np.stack((tau, G, Gfit, Gres), axis=1)
    return outputdata


def read_pycorrfit_singleparam(contents, needle_start, needle_stop, startpos=0):
    """
    Read line in .csv file header

    Parameters
    ----------
    contents : string
        .csv file contents.
    needle_start : string
        First string to search for.
    needle_stop : string
        Second string to search for.
    startpos : int, optional
        Position to start searching. The default is 0.

    Returns
    -------
    n : int
        Number with the data in between needle_start and needle_stop.
    stop : int
        Stop position.

    """
    
    start = contents.find(needle_start, startpos)
    start += len(needle_start)
    if start == -1:
        print(needle_start + "not found.")
    stop = contents.find(needle_stop, start)
    n = np.float(contents[start:stop])
    return n, stop


def plot_pycorrfit3(fileC, Ac, file3, A3, file5, A5, savefig):
    """
    Plot .asc files with PyCorrFit results for SPAD data
    Data is normalized for visualization purposes

    Parameters
    ----------
    fileC : string
        .csv file with Nx4 matrix with tau, G, Gfit and residuals
        for the central pixel data.
    Ac : float
        Normalization constants (i.e. 1/N) for the central pixel
    file3 : string
        .csv file with Nx4 matrix with tau, G, Gfit and residuals
        for the sum 3x3 data.
    A3 : float
        Normalization constants (i.e. 1/N) for sum3
    file5 : string
        .csv file with Nx4 matrix with tau, G, Gfit and residuals
                    for the sum 5x5 data.
    A5 : float
        Normalization constants (i.e. 1/N) for sum5.
    savefig : boolean
        1 to save the figure as eps, 0 to not save the figure..

    Returns
    -------
    plots.

    """
    
    # Open and plot central pixel data file
    data = csv2array(fileC)
    taumin = np.min(data[:,0])
    tau = data[:,0]
    G = data[:,1] / Ac
    Gfit = data[:,2] / Ac
    plt.scatter(tau, G, s=1, c=plot_colors('central'))
    plt.plot(tau, Gfit, c=plot_colors('central'))
    plt.xscale('log')
    plt.xlim([taumin, 10])
    plt.ylabel('G (normalized)')
    plt.xlabel(r'$\tau$ (s)')
    
    # Open and plot sum 3 data file
    data = csv2array(file3)
    taumin = np.min(data[:,0])
    tau = data[:,0]
    G = data[:,1] / A3
    Gfit = data[:,2] / A3
    plt.scatter(tau, G, s=1, c=plot_colors('sum3'))
    plt.plot(tau, Gfit, c=plot_colors('sum3'))
    
    # Open and plot sum 5 data file
    data = csv2array(file5)
    taumin = np.min(data[:,0])
    tau = data[:,0]
    G = data[:,1] / A5
    Gfit = data[:,2] / A5
    plt.scatter(tau, G, s=1, c=plot_colors('sum5'))
    plt.plot(tau, Gfit, c=plot_colors('sum5'))
    
    font = {'size' : 12}
    plt.rc('font', **font)
    plt.legend(['central', 'sum3', 'sum5'])
    plt.tight_layout()
    
    if savefig:
        plt.savefig(fileC[0:-4] + '_and_sum3_sum5.eps', format='eps')
    