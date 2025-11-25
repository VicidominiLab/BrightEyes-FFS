from ..fcs.atimes_data import atimes_data_2_duration, load_atimes_data, atimes_data_2_channels
from ..fcs.fcs2corr import Correlations
import numpy as np
from ..fcs.extract_spad_photon_streams import extract_spad_photon_streams


def atimes_file_2_pch(fname, list_of_pch=['central', 'sum3', 'sum5'], split=10, bin_time=1e-6, normalize=True, time_trace=False, list_of_pch_out=None, sysclk_MHz=240, perform_calib=False, max_k=30):
    """
    Read TCSPC data from h5 file and make PCH

    Parameters
    ----------
    fname : string
        Path to the h5 file. The file should have fields det0, det1, etc.
        with each field a 2D np.array() with the macrotime and microtime in the
        first and second column, respectively. Units of ps.
    list_of_pch : list
        List of histograms to calculate. E.g. ['central', 'sum3', 'sum5']
        for the usual 3 spot-variation histograms
    split : float
        Time of each chunk in s.
        Separate histograms are calculated for each chunk of data
    bin_time : float
        Time (s) with which to bin the arrival times before calculating the histogram.
    normalize : boolean
        Normalize the histogram to sum=1.
    list_of_pch_out : list, optional
        Names of the output histograms. The default is None, meaning equal to
        list_of_pch.
    sysclk_MHz : float, optional
        System clock in MHz. The default is 240. Only needed when perform_calib=True
    perform_calib : boolean, optional
        Perform calibration. Only needed for raw TTM data. The default is False.
    max_k : int, optional
        Maximum counts per bin for the histograms. The default is 30.

    Returns
    -------
    G : Object
        Similar to FCS data, object with data.central_chunk0 the PCH for the
        central element - chunk 0, etc.

    """
    
    if list_of_pch_out is None:
        list_of_pch_out = list_of_pch
    
    raw_data = load_atimes_data(fname, channels='auto', perform_calib=False)
    raw_data.macrotime = 1e-12 # raw macrotimes must be in ps
    raw_data.microtime = 1e-12 # raw microtimes must be in ps
    
    maxseg = 1000
    all_ch = atimes_data_2_channels(raw_data)
    data = np.zeros((maxseg, len(all_ch)))
    duration = atimes_data_2_duration(raw_data, macrotime=raw_data.macrotime, subtract_start_time=False)
    time_bins = np.linspace(0, duration, maxseg + 1)
    
    for idet, det in enumerate(all_ch):
        time = getattr(raw_data, det)[:,0]
        timeAbs = time * raw_data.macrotime
        [Itrace, timeBins] = np.histogram(timeAbs, time_bins)
        data[:, idet] = Itrace[0:] #/ (timeBins[2] - timeBins[1]) / 1e3
    
    G = atimes_2_pch_all(raw_data, list_of_pch, split, bin_time, normalize=normalize, list_of_pch_out=list_of_pch_out, max_k=max_k)
    
    if time_trace:
        return G, data
    return G


def atimes_2_pch_all(data, list_of_pch, split, bin_time, normalize=True, list_of_pch_out=None, max_k=30):
   
    G = Correlations()
    
    # CALCULATE CORRELATIONS
    try:
        data_macrotime = data.macrotime
    except:
        data_macrotime = 1e-12
    duration = atimes_data_2_duration(data, macrotime=data_macrotime, subtract_start_time=False) # in s
    n_chunks = int(np.floor(duration / split))
    
    for idx_corr, corr in enumerate(list_of_pch):
    
        # EXTRACT DATA
        if type(corr) == int:
            dataExtr = getattr(data, 'det' + str(corr))
            t0 = dataExtr[:, 0]
            corrname = 'det' + str(corr)
        elif type(corr) == str and corr[0] == 'x':
            dataExtr = getattr(data, 'det' + str(corr[1:3]))
            t0 = dataExtr[:, 0]
            corrname = 'det' + str(corr[1:3])
        else:
            print("Extracting and sorting photons")
            dataExtr = extract_spad_photon_streams(data, corr)
            t0 = dataExtr[:, 0]
            corrname = corr
        
        t0 *= data_macrotime # s
        
        for chunk in range(n_chunks):
            hist = arrival_times_to_pch(t0, bin_time, chunk, split, normalize, max_k)
            if list_of_pch_out is None:
                corrname_out = corrname + "F" + str(idx_corr)
            else:
                corrname_out = list_of_pch_out[idx_corr]
                
            setattr(G, corrname_out + '_chunk' + str(chunk), hist)
   
        # average over all chunks
        listOfFields = list(G.__dict__.keys())
        listOfFields = [i for i in listOfFields if i.startswith(corrname_out + "_chunk")]
        Gav = sum(getattr(G, i) for i in listOfFields) / len(listOfFields)
        setattr(G, corrname_out + '_average', Gav)
    
    return G
    
def arrival_times_to_pch(t0, bin_time, chunk, split, normalize, max_k=30):
    """
    Convert photon arrival times to a photon counting histogram (PCH).

    Parameters
    ----------
    t0 : array-like, shape (N,)
        Photon arrival times (e.g., in seconds). Must be 1-D and nonnegative.
    bin_time : float
        Width of each time bin (Δt). Must be > 0 (same units as t).
    t_start : float, optional
        Start time of the first bin. If None, uses min(t) or 0 if all t >= 0.
    t_end : float, optional
        End time (exclusive) of the last bin. If None, uses max(t) rounded up to a full bin.
    normalize : bool, default False
        If True, returns the histogram as probabilities summing to 1.

    Returns
    -------
    k : ndarray, shape (K,)
        Count values (0, 1, 2, ...).
    n_k : ndarray, shape (K,)
        Histogram of count occurrences: number of time bins that contained k photons.
        If normalize=True, n_k is the probability mass function P(K = k).
    counts_per_bin : ndarray, shape (M,)
        The raw photon counts in each time bin.

    Notes
    -----
    - This computes the classic PCH: discretize time into bins of width Δt,
      count photons per bin, then histogram those counts across bins.
    - Arrival times exactly equal to t_end are excluded (half-open interval).
    """
    
    tstart = chunk * split
    tstop = (chunk + 1) * split
    tchunk = t0[(t0 >= tstart) & (t0 < tstop)]
    
    tchunk -= tchunk[0]
    duration = tchunk[-1] - tchunk[0]
    n_bins = int(np.ceil(duration / bin_time))
    
    # Map each photon to its bin index
    idx = np.floor(tchunk / bin_time).astype(np.int64)
    # Guard against boundary rounding nudges
    idx = idx[(idx >= 0) & (idx < n_bins)]
    # Count photons per bin
    counts_per_bin = np.bincount(idx, minlength=n_bins).astype(np.int64)

    # Build the PCH: how many bins had k photons?
    # np.bincount over counts_per_bin gives n_k at k = 0,1,2,...
    n_k = np.bincount(counts_per_bin)
    
    hist_out = np.zeros((max_k, 2)) # [bin number, bin height]
    if len(n_k) < max_k:
        hist_out[0:len(n_k),1] = n_k
    else:
        hist_out[:,1] = n_k[0:max_k]
    
    hist_out[:,0] = np.arange(max_k, dtype=int)

    if normalize and n_bins > 0 and np.sum(hist_out[:,1]) > 0:
        hist_out[:,1] = hist_out[:,1] / np.sum(hist_out[:,1])

    return hist_out
