from .read_bh_spc import read_spc
from .arrivaltimes2timetrace import atimes2timetrace_bh
from .fcs2corr import fcs2corrsplit, plot_fcs_correlations
from .corr2csv import corr2csv

def bh_atimes2csv(fname, bintime=500, accuracy=50, split=20):
    """
    Convert Becker and Hickl fcs data to correlation curves stored in .csv files

    Parameters
    ----------
    fname : string
        File name with BH single photon arrival times (.asc file).
    bintime : float, optional
        Bin time [ns]. The default is 500.
    accuracy : int, optional
        Accuracy calculation autocorrelation. The default is 50.
    split : int, optional
        Number of fragments in which to split the time trace.
        For each fragment the autocorrelation is calculated, as
        well as the average correlation. The default is 20.

    Returns
    -------
    .csv file with all the correlations.

    """

    # load data
    data = read_spc(fname)

    # bin data
    dataBin = atimes2timetrace_bh(data, bintime)
    
    # calculate G
    G = fcs2corrsplit(dataBin, bintime/1000, ['det0'], accuracy, split)
    
    # plot G
    plot_fcs_correlations(G, ['det0_average'])
    
    # save as .csv file
    corr2csv(G, fname[0:-4])

