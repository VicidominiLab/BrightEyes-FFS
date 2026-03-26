import multiprocessing
import re
from joblib import Parallel, delayed
from .atimes2corr import atimes_2_corr
from .fcs2corr import Correlations, Correlations_chunks
from .atimes_data import atimes_data_2_duration, load_atimes_data, atimes_data_2_channels, atimes_data_attr_2_ch
import numpy as np
from .extract_spad_photon_streams import extract_spad_photon_streams


def atimes_file_2_corr(fname, list_of_g=['central', 'sum3', 'sum5'], accuracy=16, split=10, time_trace=False, root=0, list_of_g_out=None, averaging=None, parallel_prefer=None, min_step=10, print_info=True):
    """
    Calculate correlations between several photon streams with arrival times
    stored in an h5 or ptu TCSPC file

    Parameters
    ----------
    fname : str
        Path to the file
    list_of_g : list
        List of correlations to calculate
        e.g. [4, 12, 'sum3', 'sum5', 'x1011'] or ['crossAll']
    accuracy : float, optional
        Accuracy with which to calculate G. The default is 50.
    root : int, optional
        used for GUI only to pass progress. The default is 0.
    averaging : list of strings
        used to average cross correlations, e.g.
        averaging = ['14x12+15x3', '10x12+9x3', '12x14+3x15']
        averages the xcorr between ch14x12 and ch15x13 and saves them
        in a field with name from "list_of_g_out". THe length of averaging list
        and list_of_g_out must be the same
    split : float, optional
        Chunks size (s) with which to split the data. The default is 10.
    time_trace : boolean
        If true, return also the time trace
    list_of_g_out : list of strings, optional
        Names of the correlation curves. The default is None.
    min_step : int, optional
        Minimum correlation lag time calculated as a power of 2
        E.g. if arrival times are in ps, then min_step = 10 means
        min lag time = 2**10 ps = 1024 ps
        Used to speed up calculation

    Returns
    -------
    G : object
        object with [N x 2] matrices with tau and G values
    data : np.array(1000 x Nc)
        2D array with the photon time traces for each channel
        The full time trace is compressed into 1000 points per channel

    """
    if list_of_g_out is None and 'crossAll' not in list_of_g:
        list_of_g_out = list_of_g
    
    raw_data = load_atimes_data(fname, channels='auto', perform_calib=False)
    raw_data.macrotime = 1e-12 # raw macrotimes must be in ps
    raw_data.microtime = 1e-12 # raw microtimes must be in ps
    
    maxseg = 1000
    
    all_ch = atimes_data_2_channels(raw_data) # list of all channels, sorted
    data = np.zeros((maxseg, len(all_ch)))
    duration = atimes_data_2_duration(raw_data, macrotime=raw_data.macrotime, subtract_start_time=False) # s
    time_bins = np.linspace(0, duration, maxseg + 1)
    
    for idet, det in enumerate(all_ch):
        time = getattr(raw_data, det)[:,0]
        timeAbs = time * raw_data.macrotime
        [Itrace, timeBins] = np.histogram(timeAbs, time_bins)
        data[:, idet] = Itrace[0:] #/ (timeBins[2] - timeBins[1]) / 1e3
    
    G = atimes_2_corrs_parallel(raw_data, list_of_g, accuracy=accuracy, taumax="auto", perform_coarsening=True, logtau=True, root=root, split=split, averaging=averaging, list_of_g_out=list_of_g_out, parallel_prefer=parallel_prefer, min_step=min_step, print_info=print_info)
    G.dwell_time = 1e-12 # timeBins[2] - timeBins[1]
    
    if time_trace:
        return G, data
    
    return G
    


def atimes_2_corrs_parallel(data, list_of_g, accuracy=50, taumax="auto", root=0, averaging=None, perform_coarsening=True, logtau=True, split=10, list_of_g_out=None, parallel_prefer=None, min_step=10, print_info=True):
    """
    Calculate correlations between several photon streams with arrival times
    stored in macrotimes, using parallel computing to speed up the process

    Parameters
    ----------
    data : Correlations object
        Object having fields det0, det1, ..., det24 which contain
        the macrotimes of the photon arrivals [in a.u.].
        In addition, the object must have a field "data.macrotime" containing
        the macrotime unit in s (e.g. 1e-12 for ps)
    list_of_g : list
        List of correlations to calculate
        e.g. [4, 12, 'sum3', 'sum5', 'x1011'] or ['crossAll']
    accuracy : float, optional
        Accuracy with which to calculate G. The default is 50.
    taumax : float or string, optional
        Maximum tau value for which to calculate G. The default is "auto".
    root : int, optional
        used for GUI only to pass progress. The default is 0.
    averaging : list of strings
        used to average cross correlations, e.g.
        averaging = ['14x12+15x3', '10x12+9x3', '12x14+3x15']
        averages the xcorr between ch14x12 and ch15x13 and saves them
        in a field with name from "list_of_g_out". THe length of averaging list
        and list_of_g_out must be the same
    perform_coarsening : Boolean, optional
        Perform coarsening. The default is True.
    logtau : Boolean, optional
        Use log spaced tau values. The default is True.
    split : float, optional
        Chunks size (s) with which to split the data. The default is 10.
    list_of_g_out : list of strings, optional
        used for GUI only. The default is None.

    Returns
    -------
    G : object
        object with [N x 2] matrices with tau and G values

    """
    
    all_ch = atimes_data_2_channels(data) # number of ch
    duration = atimes_data_2_duration(data, macrotime=data.macrotime, subtract_start_time=False) # s
    
    n_ch = len(all_ch)
    if n_ch < 1:
        return None
    
    if taumax == "auto":
        taumax = split / 10 * 1e12 # s
    
    calc_all_xcorr = False
    start_ch = 0
    if "crossAll" in list_of_g:
        start_ch = int(all_ch[0][3:])
        stop_ch = int(all_ch[-1][3:])+1
        list_of_g = convert_crossall_to_list_of_g(stop_ch, start_ch)
        n_ch = stop_ch
        calc_all_xcorr = True
    elif averaging is not None:
        n_ch = int(all_ch[-1][3:])
    
    G = Correlations()
    
    calcAv = False
    if 'av' in list_of_g:
        # calculate the correlations of all channels and calculate average
        list_of_g.remove('av')
        list_of_g += list(range(n_ch))
        calcAv = True
    
    # keep track of which xcorrs are calculated in case of crossAll
    c0_all = []
    c1_all = []
    
    corrs_chunks = []
    G.list_of_g_out = []
    
    for idx_corr, corr in enumerate(list_of_g):
        if root != 0:
            root.progress = idx_corr / len(list_of_g)
        if print_info:
            print("Calculating correlation " + str(corr))
        
        if list_of_g_out is not None and not calc_all_xcorr and averaging is None:
            attrname = list_of_g_out[idx_corr]
            #idx_corr += 1
        elif calc_all_xcorr or averaging is not None:
            attrname = None
        else:
            attrname = None
        
        # extract atimes t0, t1 and some other data
        t0, t1, corrname, crossCorr, dataExtr, dataExtr1, c0, c1, duration, n_chunks, data_macrotime = extract_atimes_for_corr(data, corr, split, print_info=print_info)
        
        # CALCULATE CORRELATIONS
        # go over all filters
        num_filters = np.shape(dataExtr)[1] - 1 # including unfiltered
        for j in range(num_filters):
            if j > 0:
                if print_info:
                    print("   Filter " + str(j))
            if crossCorr == False:
                if j == 0:
                    Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1, prefer=parallel_prefer)(delayed(parallel_g)(t0, [1], data_macrotime, j, split, accuracy, taumax, perform_coarsening, logtau, min_step, chunk) for chunk in list(range(n_chunks)))
                else:
                    w0 = dataExtr[:, j+1]
                    Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1, prefer=parallel_prefer)(delayed(parallel_g)(t0, w0, data_macrotime, j, split, accuracy, taumax, perform_coarsening, logtau, min_step, chunk) for chunk in list(range(n_chunks)))
            else:
                if j == 0:
                    Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1, prefer=parallel_prefer)(delayed(parallel_gx)(t0, [1], t1, [1], data_macrotime, j, split, accuracy, taumax, perform_coarsening, logtau, min_step, chunk) for chunk in list(range(n_chunks)))
                else:
                    w0 = dataExtr[:, j+1]
                    w1 = dataExtr1[:, j+1]
                    Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1, prefer=parallel_prefer)(delayed(parallel_gx)(t0, w0, t1, w1, data_macrotime, j, split, accuracy, taumax, perform_coarsening, logtau, min_step, chunk) for chunk in list(range(n_chunks)))
            
            if calc_all_xcorr or averaging is not None:
                c0_all.append(c0)
                c1_all.append(c1)
                if idx_corr == 0:
                    n_times = len(Processed_list[0][:,0])
                    g_cross_all = np.zeros((num_filters, n_chunks, n_times, n_ch, n_ch))
                for chunk in range(n_chunks):
                    g_temp = Processed_list[chunk]
                    g_times = g_temp[:,0]
                    g_cross_all[j, chunk, :, int(c0), int(c1)] = g_temp[:,1]
                    
            else:
                for chunk in range(n_chunks):
                    g_temp = Processed_list[chunk]
                    if chunk == 0 and j == 0:
                        # make one object for each correlation, with all chunks and all filters
                        corrs_chunks.append(Correlations_chunks(np.zeros((n_chunks, np.shape(g_temp)[0], 1+num_filters))))
                        G.list_of_g_out.append(attrname)
                        
                    if not np.isnan(g_temp).any():
                        corrs_chunks[idx_corr].chunks[chunk,:,0] = g_temp[:,0] # tau
                        corrs_chunks[idx_corr].chunks[chunk,:,j+1] = g_temp[:,1] # G_unfiltered, G_filtered
                    
                G.add_corr_chunks(attrname, corrs_chunks[idx_corr].chunks)
    
    if calc_all_xcorr or averaging is not None:
        add_cross_all_to_G(G, g_cross_all, g_times, averaging, list_of_g_out, c0_all, c1_all)
    
    if calcAv:
        # calculate average correlation of all detector elements
        for f in range(num_filters):
            # start with correlation of detector 20 (last one)
            Gav = getattr(G, 'det' + str(n_ch-1) + 'F' + str(f) + '_average')
            # add correlations detector elements 0-19
            for det in range(n_ch - 1):
                Gav += getattr(G, 'det' + str(det) + 'F' + str(f) + '_average')
            # divide by the number of detector elements to get the average
            Gav = Gav / n_ch
            # store average in G
            setattr(G, 'F' + str(f) + '_average', Gav)
    
    return G


def parallel_g(t0, w0, macrotime, filter_number, split, accuracy, taumax, perform_coarsening, logtau, min_step, chunk):
    tstart = chunk * split / macrotime
    tstop = (chunk + 1) * split / macrotime
    tchunk = t0[(t0 >= tstart) & (t0 < tstop)]
    tchunkN = tchunk - tchunk[0]
    if filter_number == 0:
        # no filter
        Gtemp = atimes_2_corr(tchunkN, tchunkN, [1], [1], macrotime, accuracy, taumax, perform_coarsening, logtau, min_step=min_step)
    else:
        # filters
        wchunk = w0[(t0 >= tstart) & (t0 < tstop)].copy()
        Gtemp = atimes_2_corr(tchunkN, tchunkN, wchunk, wchunk, macrotime, accuracy, taumax, perform_coarsening, logtau, min_step=min_step)
    return(Gtemp)


def parallel_gx(t0, w0, t1, w1, macrotime, filter_number, split, accuracy, taumax, perform_coarsening, logtau, min_step, chunk):
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
        Gtemp = atimes_2_corr(tchunk0, tchunk1, [1], [1], macrotime, accuracy, taumax, perform_coarsening, logtau, min_step=min_step)
    else:
        # filters
        wchunk0 = w0[(t0 >= tstart) & (t0 < tstop)].copy()
        wchunk1 = w1[(t1 >= tstart) & (t1 < tstop)].copy()
        Gtemp = atimes_2_corr(tchunk0, tchunk1, wchunk0, wchunk1, macrotime, accuracy, taumax, perform_coarsening, logtau, min_step=min_step)
    return(Gtemp)


def convert_crossall_to_list_of_g(n_ch, start_ch=0):
    """
    Convert the string "crossAll" to a list of all crosscorrelations to calculate

    Parameters
    ----------
    n_ch : int
        Number of channels.

    Returns
    -------
    list_of_g : list of strings
        ["x0000", "x0001", ..., "x2525"].

    """
    list_of_g = []
    for i in range(n_ch):
        if i < start_ch:
            continue
        str_i = ""
        if i < 10:
            str_i += "0"
        str_i += str(i)
        for j in range(n_ch):
            if j < start_ch:
                continue
            str_j = ""
            if j < 10:
                str_j += "0"
            str_j += str(j)
            list_of_g.append("x" + str_i + str_j)
    return list_of_g


def extract_atimes_for_corr(data, corr, split, print_info=True):
    crossCorr = False
    dataExtr1 = None
    c0 = None
    c1 = None
    if type(corr) == int:
        # autocorrelation single channel
        dataExtr = getattr(data, 'det' + str(corr))
        t0 = dataExtr[:, 0]
        t1 = None
        corrname = 'det' + str(corr)
    elif corr[0] == 'x':
        # cross-correlation two channels, e.g., x0412 between 4 and 12
        c0 = corr[1:3] # first channel
        c1 = corr[3:5] # second channel
        if print_info:
            print("Extracting photons channels " + c0 + " and " + c1)
        dataExtr = getattr(data, 'det' + str(int(c0)))
        t0 = dataExtr[:, 0]
        dataExtr1 = getattr(data, 'det' + str(int(c1)))
        t1 = dataExtr1[:, 0]
        corrname = corr
        crossCorr = True
    elif corr[0] == 'C':
        # crosscorrelation custom sum of channels
        xpos = np.max([corr.find('X'), corr.find('x')])
        if xpos > -1:
            if print_info:
                print('Calculating crosscorrelation custom sum')
            dataExtr = extract_spad_photon_streams(data, corr[0:xpos])
            dataExtr1 = extract_spad_photon_streams(data, 'C' + corr[xpos+1:])
        else:
            if print_info:
                print('Calculating autocorrelation custom sum')
            dataExtr = extract_spad_photon_streams(data, corr)
            dataExtr1 = dataExtr
        t0 = dataExtr[:, 0]
        t1 = dataExtr1[:, 0]
        corrname = corr
        crossCorr = True
    else:
        if print_info:
            print("Extracting photons")
        dataExtr = extract_spad_photon_streams(data, corr)
        t0 = dataExtr[:, 0]
        t1 = None
        corrname = corr
    
    try:
        data_macrotime = data.macrotime
    except:
        data_macrotime = 1e-12
    duration = atimes_data_2_duration(data, macrotime=data_macrotime, subtract_start_time=False)
    Nchunks = int(np.floor(duration / split))
    
    return t0, t1, corrname, crossCorr, dataExtr, dataExtr1, c0, c1, duration, Nchunks, data_macrotime


def calc_average_correlation(G):
    # Get list of "root" names, i.e. without "_chunk"
    Gfields = list(G.__dict__.keys())
    t = [Gfields[i].split("_chunk")[0] for i in range(len(Gfields))]
    t = list(dict.fromkeys(t))
    for field in t:
        avList = [i for i in Gfields if i.startswith(field + '_chunk')]
        # check if all elements have same dimension
        Ntau = [len(getattr(G, i)) for i in avList]
        avList2 = [avList[i] for i in range(len(avList)) if Ntau[i] == Ntau[0]]
        
        Gtemp = getattr(G, avList2[0]) * 0
        GtempSquared = getattr(G, avList2[0])**2 * 0
        for chunk in avList2:
            Gtemp += getattr(G, chunk)
            GtempSquared += getattr(G, chunk)**2
        
        Gtemp /= len(avList2)
        Gstd = np.sqrt(np.clip(GtempSquared / len(avList2) - Gtemp**2, 0, None))
        
        Gtot = np.zeros((np.shape(Gtemp)[0], np.shape(Gtemp)[1] + 1))
        Gtot[:, 0:-1] = Gtemp # [time, average]
        Gtot[:, -1] = Gstd[:,1] # standard deviation
        
        setattr(G, str(field) + '_average', Gtot)
        
    return G


def add_cross_all_to_G(G, g_cross_all, tau, averaging, list_of_g_out, c0_all=None, c1_all=None):
    # g_cross_all = [Nf, Nc, G, c0, c1]
    averaging_list = []
    if averaging is None:
        for i in range(len(c0_all)):
            averaging_list.append([[int(c0_all[i]), int(c1_all[i])]])
    else:
        for av in averaging:
            singleAv = [int(ch_nr) for ch_nr in re.findall(r'\d+', av)]
            single_av_list = [[singleAv[i], singleAv[i + 1]] for i in range(0, len(singleAv), 2)]
            averaging_list.append(single_av_list)
    
    if list_of_g_out is None:
        list_of_g_out = []
        for av in averaging_list:
            corr_av_single_str = 'det'
            for j in range(len(av)):
                corr_av_single_str += str(av[j][0]) + 'x' +  str(av[j][1]) + '+'
            list_of_g_out.append(corr_av_single_str[:-1])
    
    # which curves to average over is defined now and their names are defined
    for i in range(len(averaging_list)):
        list_of_av_single_corr = averaging_list[i]
        c0_idx = []
        c1_idx = []
        for j in range(len(list_of_av_single_corr)):
            c0_idx.append(list_of_av_single_corr[j][0])
            c1_idx.append(list_of_av_single_corr[j][1])
        g_single_mean = g_cross_all[:, :, :, c0_idx, c1_idx].mean(axis=-1)
        
        # move Nf axis to the end → (Nc, Ng, Nf)
        g_single_mean_reordered = np.transpose(g_single_mean, (1, 2, 0))
        tau_expanded = np.broadcast_to(tau, (g_single_mean.shape[1], g_single_mean.shape[2]))[..., None]
        g_single_mean_final = np.concatenate((tau_expanded, g_single_mean_reordered), axis=-1)
        
        G.add_corr_chunks(list_of_g_out[i], g_single_mean_final)
        G.list_of_g_out.append(list_of_g_out[i])
            