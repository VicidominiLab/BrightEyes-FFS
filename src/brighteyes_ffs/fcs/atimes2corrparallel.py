import multiprocessing
from joblib import Parallel, delayed
from .atimes_2_corr import atimes_2_corr
from .fcs2corr import Correlations
import numpy as np
from .extract_spad_photon_streams import extract_spad_photon_streams

def atimes_2_corrs_parallel(data, list_of_corr, accuracy=50, taumax="auto", perform_coarsening=True, logtau=True, split=10):
    """
    Calculate correlations between several photon streams with arrival times
    stored in macrotimes, using parallel computing to speed up the process

    Parameters
    ----------
    data : TYPE
        Object having fields det0, det1, ..., det24 which contain
        the macrotimes of the photon arrivals [in a.u.].
    list_of_corr : list
        List of correlations to calculate
        e.g. [4, 12, 'sum3', 'sum5', 'x1011'].
    accuracy : float, optional
        Accuracy with which to calculate G. The default is 50.
    taumax : float or string, optional
        Maximum tau value for which to calculate G. The default is "auto".
    perform_coarsening : Boolean, optional
        Perform coarsening. The default is True.
    logtau : Boolean, optional
        Use log spaced tau values. The default is True.
    split : float, optional
        Chunks size with which to split the data. The default is 10.

    Returns
    -------
    G : object
        object with [N x 2] matrices with tau and G values

    """
    
    if taumax == "auto":
        taumax = 1 / data.macrotime
    
    G = Correlations()
    
    Ndet = 25
    calcAv = False
    if 'av' in list_of_corr:
        # calculate the correlations of all channels and calculate average
        list_of_corr.remove('av')
        list_of_corr += list(range(Ndet))
        calcAv = True
    
    for corr in list_of_corr:
        print("Calculating correlation " + str(corr))
        
        # EXTRACT DATA
        crossCorr = False
        if type(corr) == int:
            dataExtr = getattr(data, 'det' + str(corr))
            t0 = dataExtr[:, 0]
            corrname = 'det' + str(corr)
        elif corr == "sum5" or corr == "sum3":
            print("Extracting and sorting photons")
            dataExtr = extract_spad_photon_streams(data, corr)
            t0 = dataExtr[:, 0]
            corrname = corr
        elif corr[0] == 'x':
            c0 = corr[1:3] # first channel
            c1 = corr[3:5] # second channel
            print("Extracting photons channels " + c0 + " and " + c1)
            dataExtr = getattr(data, 'det' + str(int(c0)))
            t0 = dataExtr[:, 0]
            dataExtr1 = getattr(data, 'det' + str(int(c1)))
            t1 = dataExtr1[:, 0]
            corrname = corr
            crossCorr = True
        
        # CALCULATE CORRELATIONS
        duration = t0[-1] * data.macrotime
        Nchunks = int(np.floor(duration / split))
        # go over all filters
        for j in range(np.shape(dataExtr)[1] - 1):
            print("   Filter " + str(j))
            if crossCorr == False:
                if j == 0:
                    Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(parallel_g)(t0, [1], data.macrotime, j, split, accuracy, taumax, perform_coarsening, logtau, chunk) for chunk in list(range(Nchunks)))
                else:
                    w0 = dataExtr[:, j+1]
                    Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(parallel_g)(t0, w0, data.macrotime, j, split, accuracy, taumax, perform_coarsening, logtau, chunk) for chunk in list(range(Nchunks)))
            else:
                if j == 0:
                    Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(parallel_gx)(t0, [1], t1, [1], data.macrotime, j, split, accuracy, taumax, perform_coarsening, logtau, chunk) for chunk in list(range(Nchunks)))
                else:
                    w0 = dataExtr[:, j+1]
                    w1 = dataExtr1[:, j+1]
                    Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(parallel_gx)(t0, w0, t1, w1, data.macrotime, j, split, accuracy, taumax, perform_coarsening, logtau, chunk) for chunk in list(range(Nchunks)))
            
            for chunk in range(Nchunks):
                setattr(G, corrname + "F" + str(j) + '_chunk' + str(chunk), Processed_list[chunk])
           
            # average over all chunks
            listOfFields = list(G.__dict__.keys())
            listOfFields = [i for i in listOfFields if i.startswith(corrname + "F" + str(j) + "_chunk")]
            Gav = sum(getattr(G, i) for i in listOfFields) / len(listOfFields)
            setattr(G, corrname + "F" + str(j) + '_average', Gav)
    
    if calcAv:
        # calculate average correlation of all detector elements
        for f in range(np.shape(dataExtr)[1] - 1):
            # start with correlation of detector 20 (last one)
            Gav = getattr(G, 'det' + str(Ndet-1) + 'F' + str(f) + '_average')
            # add correlations detector elements 0-19
            for det in range(Ndet - 1):
                Gav += getattr(G, 'det' + str(det) + 'F' + str(f) + '_average')
            # divide by the number of detector elements to get the average
            Gav = Gav / Ndet
            # store average in G
            setattr(G, 'F' + str(f) + '_average', Gav)
    
    return G


def parallel_g(t0, w0, macrotime, filter_number, split, accuracy, taumax, perform_coarsening, logtau, chunk):
    tstart = chunk * split / macrotime
    tstop = (chunk + 1) * split / macrotime
    tchunk = t0[(t0 >= tstart) & (t0 < tstop)]
    tchunkN = tchunk - tchunk[0]
    if filter_number == 0:
        # no filter
        Gtemp = atimes_2_corr(tchunkN, tchunkN, [1], [1], macrotime, accuracy, taumax, perform_coarsening, logtau)
    else:
        # filters
        wchunk = w0[(t0 >= tstart) & (t0 < tstop)].copy()
        Gtemp = atimes_2_corr(tchunkN, tchunkN, wchunk, wchunk, macrotime, accuracy, taumax, perform_coarsening, logtau)
    return(Gtemp)


def parallel_gx(t0, w0, t1, w1, macrotime, filter_number, split, accuracy, taumax, perform_coarsening, logtau, chunk):
    tstart = chunk * split / macrotime
    tstop = (chunk + 1) * split / macrotime
    tchunk0 = t0[(t0 >= tstart) & (t0 < tstop)]
    tchunk1 = t1[(t1 >= tstart) & (t1 < tstop)]
    # normalize time by sutracting first number
    tN = np.min([tchunk0[0], tchunk1[0]])
    tchunk0 = tchunk0 - tN
    tchunk1 = tchunk1 - tN
    if filter_number == 0:
        # no filter
        Gtemp = atimes_2_corr(tchunk0, tchunk1, [1], [1], macrotime, accuracy, taumax, perform_coarsening, logtau)
    else:
        # filters
        wchunk0 = w0[(t0 >= tstart) & (t0 < tstop)].copy()
        wchunk1 = w1[(t1 >= tstart) & (t1 < tstop)].copy()
        Gtemp = atimes_2_corr(tchunk0, tchunk1, wchunk0, wchunk1, macrotime, accuracy, taumax, perform_coarsening, logtau)
    return(Gtemp)

