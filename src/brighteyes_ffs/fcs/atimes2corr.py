import numpy as np
from datetime import datetime
from .extract_spad_photon_streams import extract_spad_photon_streams


class Correlations:
    pass


def atimes_2_corrs(data, list_of_corr, accuracy=50, taumax="auto", perform_coarsening=True, logtau=True, split=10):
    """
    Calculate correlations between several photon streams with arrival times
    stored in macrotimes

    Parameters
    ----------
    data : object
        Object having fields det0, det1, ..., det24 which contain
            the macrotimes of the photon arrivals [in a.u.].
    list_of_corr : list
        List of correlations to calculate.
    accuracy : int, optional
        Accurracy with which to calulcate the correlation. The default is 50.
    taumax : float or string, optional
        Maximum tau value for which to calculate G. The default is "auto".
    perform_coarsening : Boolean, optional
        Apply coarsening. The default is True.
    logtau : TYPE, optional
        Use logarithmically spaced tau values. The default is True.
    split : float, optional
        Chunk size with which to split the data [s]. The default is 10.

    Returns
    -------
    G : object
        correlations.

    """
    
    if taumax == "auto":
        taumax = 1 / data.macrotime
    
    G = Correlations()
    
    Ndet = 21
    calcAv = False
    if 'av' in list_of_corr:
        # calculate the correlations of all channels and calculate average
        list_of_corr.remove('av')
        list_of_corr += list(range(Ndet))
        calcAv = True
    
    for corr in list_of_corr:
        # EXTRACT DATA
        if type(corr) == int:
            dataExtr = getattr(data, 'det' + str(corr))
            t0 = dataExtr[:, 0]
            corrname = 'det' + str(corr)
        elif corr == "sum5" or corr == "sum3":
            print("Extracting and sorting photons")
            dataExtr = extract_spad_photon_streams(data, corr)
            t0 = dataExtr[:, 0]
            corrname = corr
        
        # CALCULATE CORRELATIONS
        duration = t0[-1] * data.macrotime
        Nchunks = int(np.floor(duration / split))
        # go over all filters
        for j in range(np.shape(dataExtr)[1] - 1):
            for chunk in range(Nchunks):
                corrstring = corrname + "F" + str(j) + "_chunk" + str(chunk)
                print("Calculating correlation " + corrstring)
                tstart = chunk * split / data.macrotime
                tstop = (chunk + 1) * split / data.macrotime
                tchunk = t0[(t0 >= tstart) & (t0 < tstop)]
                tchunk -= tchunk[0]
                print("   Nphotons: " + str(len(tchunk)))
                if j == 0:
                    # no filter
                    Gtemp = atimes_2_corr(tchunk, tchunk, [1], [1], data.macrotime, accuracy, taumax, perform_coarsening, logtau)
                else:
                    # filters
                    w0 = dataExtr[:, j+1]
                    wchunk = w0[(t0 >= tstart) & (t0 < tstop)]
                    Gtemp = atimes_2_corr(tchunk, tchunk, wchunk, wchunk, data.macrotime, accuracy, taumax, perform_coarsening, logtau)
                setattr(G, corrstring, Gtemp)
            # average over all chunks
            print("   Calculating average correlation")
            listOfFields = list(G.__dict__.keys())
            listOfFields = [i for i in listOfFields if i.startswith(corrname + "F" + str(j) + "_chunk")]
            Gav = sum(getattr(G, i) for i in listOfFields) / len(listOfFields)
            setattr(G, corrname + "F" + str(j) + '_average', Gav)
    
    if calcAv:
        # calculate average correlation
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


def atimes_2_corr(t0, t1, w0, w1, macroTime, accuracy=50, taumax="auto", perform_coarsening=True, logtau=True):
    """
    Calculate correlation between two photon streams with arrival times t0 and t1
    Inspired by Wahl et al., Opt. Expr. 11 (26), 2003

    Parameters
    ----------
    t0 : np.array()
        Vector with arrival times channel 1 [a.u.].
    t1 : np.array()
        Vector with arrival times channel 2 [a.u.]..
    w0 : np.array()
        Vector with (filtered) weights channel 1 [a.u.].
    w1 : np.array()
        Vector with (filtered) weights channel 2 [a.u.]..
    macroTime : float
        Multiplication factor for the arrival times vectors [s].
    accuracy : int, optional
        Number of tau values for which G is calculated before delta tau is
        doubled
        E.g. accuracy = 3 yields:
            tau = [1, 2, 3, 5, 7, 9, 13, 17, 21, 29, 37, 45,...]. The default is 50.
    taumax : float or string, optional
        Maximum tau value for which to calculate the correlation
        If left empty, 1/10th of the measurement duration is used. The default is "auto".
    perform_coarsening : Boolean, optional
        Apply time trace coarsening for more efficient (and more accurate)
        correlation calculation. The default is True.
    logtau : Boolean, optional
        Use logarithmically spaced tau values. The default is True.

    Returns
    -------
    np.array()
        [N x 2] matrix with tau and G values.

    """
    
    
    # convert t0 and t1 lists to array
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    
    # check max tau value
    if taumax=="auto":
        taumax = t0[-1] / 10
    
    # generate list [tau] with logarithmically distributed tau values
    t = 0
    tau = []
    step = 1
    if logtau:
        while t <= taumax:
            for i in range(accuracy):
                t += step
                # make sure t is multiple of step (needed for coarsing)
                t = int(np.floor(t / step) * step)
                tau.append(t)
            step *= 2
        tau = [i for i in tau if i <= taumax]
        tau = [0] + tau
    else:
        tau = np.linspace(0, taumax, taumax+1, dtype=int)
    
    # create array for g and weights
    N = len(tau)
    g = np.zeros(N)
    if len(w0) == 1:
        w0 = np.ones(len(t0))
    if len(w1) == 1:
        w1 = np.ones(len(t1))
    
    # coarse factor
    c = 1

    printDateTime = False
    if len(t0) > 100000:
        printDateTime = True
    
    # calculate g for each tau value
    if printDateTime:
        print(datetime.now())
    
    
    # tau = np.asarray(tau)
    # tau_diff = tau[1:] - tau[0:-1]
    # g = []
    
    # for tau_d in list(np.sort(np.unique(tau_diff))):
    #     idx = tau_diff == tau_d
    #     idxT = np.where(idx==True)[0]
    #     tau_list_temp = list(tau[idxT + 1])
    #     if tau_d == 1:
    #         tau_list = [0] + tau_list_temp
    #     else:
    #         tau_list = tau_list_temp
        
        
        
    #     gtemp = Parallel(n_jobs=-2)(delayed(atimes_2_corr_single)(t0, t1, w0, w1, t, c) for t in tau_list)
    #     g += gtemp
    #     # perform coarsening
    #     c *= 2
    #     [t0, w0, ind] = time_coarsening(t0, w0)
    #     [t1, w1, ind] = time_coarsening(t1, w1)
    
    # g = np.transpose(np.asarray(g))
        
    for i in range(N):
        t = tau[i]
        g[i] = atimes_2_corr_single(t0, t1, w0, w1, t/c, c)
        if perform_coarsening and np.mod(i, accuracy) == 0 and i != 0:
            # change in step size: perform coarsening
            c *= 2
            [t0, w0, ind] = time_coarsening(t0, w0)
            [t1, w1, ind] = time_coarsening(t1, w1)
        
    if printDateTime:
        print(datetime.now())
    
    tau = np.asarray(tau) * macroTime
    
    return np.transpose([tau, g])


def atimes_2_corr_single(t0, t1, w0, w1, tau, c):
    """
    Calculate single correlation value between two photon streams with arrival
    times t0 and t1, weight vectors w0 and w1, and tau value tau

    Parameters
    ----------
    t0 : np.array()
        Vector with arrival times channel 1
        [multiples of minimum coarsed macrotime].
    t1 : np.array()
        Vector with arrival times channel 2
        [multiples of minimum coarsed macrotime].
    w0 : np.array()
        Vector with weight values channel 1.
    w1 : np.array()
        Vector with weight values channel 2.
    tau : np.array()
        tau value for which to calculate the correlation
        [multiples of minimum coarsed macrotime].
    c : int
        Coarsening factor (power of 2).

    Returns
    -------
    g : float
        Single value g(tau).

    """
    
    # calculate time shifted vector
    t1 = t1 + tau
    
    # find intersection between t0 and t1
    [tauDouble, idxt0, idxt1] = np.intersect1d(t0, t1, return_indices=True)
    
    # overlap time
    T = (t0[-1] - tau + 1) * c
    
    # calculate autocorrelation value
    G = np.sum(w0[idxt0] * w1[idxt1]) / T / c
    
    # normalize G
    I0 = np.sum(w0[t0 >= tau]) / T
    I1 = np.sum(w1[t1<=t0[-1]]) / T
    g = G / I0 / I1 - 1
    
    return g


def time_coarsening(t, w):
    # divide all arrival times by 2
    t = np.floor(t / 2)
    
    # check for duplicate values
    ind = np.where(np.diff(t) == 0)[0]
    
    # sum weights
    w[ind] += w[ind+1]
    
    # delete duplicates
    w = np.delete(w, ind+1)
    t = np.delete(t, ind+1)
    
    return [t, w, ind]
    