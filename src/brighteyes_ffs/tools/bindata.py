import numpy as np

def bindata(data, binsize, interleaved_binning=False):
    """
    Sum elements in "data" in groups of size "binsize", creating a new vector
    with summed data

    Parameters
    ----------
    data : np.array
        Vector with data.
    binsize : int
        Number of data points per bin.
    interleaved_binning : boolean
        True for binning consecutive data points, false for interleaved binning

    Returns
    -------
    bdata : np.array
        Binned data.

    """
    
    if interleaved_binning:
        T = data.shape[0]
        spacing = T // binsize # number of complete groups
    
        # Reshape into (n_bins, m, ...)
        new_shape = (spacing, binsize) + data.shape[1:]
        binned_array = np.zeros(new_shape, dtype=data.dtype)
    
        for i in range(binsize):
            binned_array[:, i] = data[spacing*i : spacing*(i+1)]
        
        # Sum over the m points in each bin
        bdata = binned_array.sum(axis=1)
    
    else:
        squeeze = False
        if data.ndim == 1:
            data = data[:, np.newaxis] 
            squeeze = True
        cumdata = np.array(np.zeros([1, np.size(data, 1)]), dtype="uint32")
        cumdata = np.append(cumdata, np.uint32(np.cumsum(data, axis=0)), axis=0)
        bdata = cumdata[0::binsize, :]
        
        for i in range(np.size(data, 1)):
            bdata[0:-1, i] = np.ediff1d(bdata[:, i])
            
        bdata = bdata[0:-1]
        
        if squeeze:
            bdata = np.squeeze(bdata)
            
    return bdata


def bindata_chunks(data, binsize, printChunkNr=False, interleaved_binning=False):
    squeeze = False
    if data.ndim == 1:
        data = data[:, np.newaxis] 
        squeeze = True
    bdata = np.empty((0, np.size(data, 1)), int)
    N = np.size(data, 0)
    chunksize = np.min([int(binsize * np.floor(10e6 / binsize)), N])
    binsize = np.min((binsize, chunksize))
    Nchunks = int(np.floor(N/chunksize))
    for i in range(Nchunks):
        if printChunkNr:
            print("Chunk " + str(i+1) + " of " + str(Nchunks))
        newbindata = bindata(data[i*chunksize:(i+1)*chunksize, :], binsize, interleaved_binning)
        np.size(newbindata)
        bdata = np.append(bdata, newbindata, axis=0)
    if squeeze:
        bdata = np.squeeze(bdata)
    return bdata