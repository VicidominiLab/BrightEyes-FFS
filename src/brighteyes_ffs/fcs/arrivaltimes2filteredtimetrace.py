import numpy as np


class ATimesData:
    pass


def arrival_times_2_filtered_time_trace(data, filter_function, microbin_length, macrobin_length):
    """
    Convert a list of arrivalTimes to an intensity time trace I(t)

    Parameters
    ----------
    data : object
        Object with for each detector element field with a 2D int array, e.g.:
            data.det0 = np.array(N x 2)
                with N the number of photons
                First column: macro arrival times [a.u.]
                Second column: micro arrival times [a.u.]
            data.macrotime: macroTime [s]
            data.microtime: microTime [s].
    filter_function : np.array()
        np.array(M x Nf) with the filter functions
            M: number of lifetime bins
            Nf: number of filters, tyically 2 (1 fluor, 1 afterpulse)
                sum of each row = 1
        For multiple detector elements, and thus multiple filters,
        this variable is np.array(Ndet x M x Nf).
    microbin_length : float
        Duration of each bin of the lifetime histogram [s].
    macrobin_length : float
        Duration of each bin of the final intensity trace [s].

    Returns
    -------
    timeTraceF
        Object with for each filter function a np.array(K x L) with
            K the number of photon bins of duration binLength
            L the number of detector elements.

    """
    
    # get list of detector fields
    listOfFields = list(data.__dict__.keys())
    listOfFields = [i for i in listOfFields if i.startswith('det')]
    Ndet = len(listOfFields)
    
    # get total measurement time
    Mtime = 0
    for i in range(Ndet):
        dataSingleDet = getattr(data, listOfFields[i])
        Mtime = np.max((Mtime, dataSingleDet[-1, 0]))
    Mtime *= data.macrotime # [s]
    
    # number of time bins for binned and filtered intensity traces
    Nbins = int(np.ceil(Mtime / macrobin_length))
    
    # make sure filter_function is a 3D array
    if len(np.shape(filter_function)) == 2:
        filter_function = np.expand_dims(filter_function, 0)
    
    # create empty arrays to store output traces
    Nfilt = np.shape(filter_function)[-1]
    timeTraces = np.zeros((Ndet, Nbins, Nfilt), dtype=float)
    
    # go through each channel and create filtered intensity trace
    for det in range(Ndet):
        print("Calculating filtered intensity trace " + listOfFields[det])
        dataSingleDet = getattr(data, listOfFields[det])
        # calculate for each photon in which bin it should be
        photonMacroBins = np.int64(dataSingleDet[:,0] * data.macrotime / macrobin_length)
        photonMicroBins = np.int64(dataSingleDet[:,1] * data.microtime / microbin_length)
        for i in range(len(photonMacroBins)):
            for filt in range(Nfilt):
                timeTraces[det, photonMacroBins[i], filt] += filter_function[det, photonMicroBins[i], filt]

    return np.squeeze(timeTraces)


def atimes_filtered(data, filter_function, micro_bin=False, verbose=False):
    """
    Filter a list of arrivalTimes

    Parameters
    ----------
    data : object
        Object with for each detector element field with a 2D int array, e.g.:
            data.det0 = np.array(N x 2)
                with N the number of photons
                First column: macro arrival times [a.u.]
                Second column: micro arrival times [*]
            data.macrotime: macro time [s]
            data.microtime: micro time [s]
            data.microbintime: micro bin time [s].
    filter_function : np.array()
        np.array(M x Nf) with the filter functions
                M: number of lifetime bins
                Nf: number of filters, tyically 2 (1 fluor, 1 afterpulse)
                    sum of each row = 1
            For multiple detector elements, and thus multiple filters,
            this variable is np.array(Ndet x M x Nf).
    micro_bin : Boolean, optional
        True if micro arrival times [*] are in bin numbers
        False if micro arrival times [*] are in [a.u.]
            In this case, the bin numbers are calculated as
            bin = t * data.microtime / data.microbintime
            with data.microtime the microtime unit in s
            and data.microbintime the bin time in s. The default is False.

    Returns
    -------
    Data object is modified in-place. Nothing is returned.
    but data.det0 is now np.array(N x 2+Nf)
        For every detector element, Nf columns are added with 
        the filtered weights for the arrival times.

    """
    
    # get list of detector fields
    listOfFields = list(data.__dict__.keys())
    listOfFields = [i for i in listOfFields if i.startswith('det')]
    Ndet = len(listOfFields)
    
    # make sure filter_function is a 3D array (detector, microbin, filter function)
    if len(np.shape(filter_function)) == 2:
        filter_function = np.expand_dims(filter_function, 0)
    
    # number of filters
    Nf = np.shape(filter_function)[2]
    
    # number of time bins
    M = np.shape(filter_function)[1]
    
    # micro times normalization factor
    microN = 1
    if not micro_bin:
        microN = data.microtime / data.microbintime
    
    # go through each channel and create filtered intensity trace
    for det in range(Ndet):
        if verbose:
            print("Calculating filtered photon streams " + listOfFields[det])
        # get photon streams single detector element
        dataSingleDet = getattr(data, listOfFields[det])
        # remove exessive columns which may already contain filtered photon streams
        dataSingleDet = dataSingleDet[:, 0:2]
        # calculate for each photon the filtered values
        photonMicroBins = np.int64(np.floor(dataSingleDet[:,1] * microN))
        photonMicroBins = np.clip(photonMicroBins, 0, M-1)
        for filt in range(Nf):
            filteredValues = np.expand_dims(np.take(np.squeeze(filter_function[det, :, filt]), photonMicroBins), 1)
            dataSingleDet = np.concatenate((dataSingleDet, filteredValues), axis=1)
        setattr(data, listOfFields[det], dataSingleDet)