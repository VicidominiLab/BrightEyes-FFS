import numpy as np

def filter_g_obj(G, filt='sum5', f_acc=0.66):
    """
    Check for bad chunks in autocorrelations

    Parameters
    ----------
    G : Correlations object
        Object with fields for correlation curves, e.g. G.sum5_chunk0.
    filt : str, optional
        Choose which correlation type to use for filtering. The default is 'sum5'.
    f_acc : float, optional
        Acceptance ration. The default is 0.66, meaning that the 1/3 worst
        correlation curves are rejected.

    Returns
    -------
    chunks_on : np.array()
        1D boolean array with the same length as the number of data chunks
        Values are True (1) for good chunks and False (0) for bad chunks.

    """
    [indsorted, _, _] = sort_g_obj(G, filt)
    if indsorted is None:
        return None
    n_chunks = len(indsorted)
    chunks_on = np.zeros((n_chunks))
    indsorted = indsorted[0:int(np.round(f_acc*n_chunks))]
    chunks_on[indsorted] = 1
    return chunks_on
    

def sort_g_obj(G, filt):
    keylist = list(G.__dict__.keys())
    good_keys = []
    for ind, key in enumerate(keylist):
        if key.startswith(filt + '_chunk'):
            good_keys.append(key)
    
    if len(good_keys) == 0:
        return None, None, None
    
    g_shape = np.shape(getattr(G, good_keys[0]))
    Garray = np.zeros((g_shape[0],len(good_keys)))
    for i in range(len(good_keys)):
        Garray[:,i] = getattr(G, filt + '_chunk' + str(i))[:,1]
    
    [indsorted, Gsorted, dGmArray] = automatic_suppression_g(Garray)
    
    return indsorted, Gsorted, dGmArray
    

def automatic_suppression_g(G):
    """
    Automated suppression of artifacts in FCS data
    based on the article by Ries et al., Opt. Express, 2010.
    
    Parameters:
    G : np.ndarray
        2D array (m x n) with each column containing an autocorrelation curve
        (m lag time points, n curves).
        
    Returns:
    indsorted : np.ndarray
        List of the indices from best to worst correlation.
    Gsorted : np.ndarray
        Sorted autocorrelation array from low to high deviation.
    dGmArray : np.ndarray
        Sorted autocorrelation deviations from low to high.
    
    """
    
    Gsize = G.shape
    n = Gsize[1]  # number of autocorrelation curves
    ind = np.arange(n)  # keeps track of the indices of the images
    indsorted = np.zeros(n, dtype=int)
    Gsorted = np.zeros(Gsize)
    dGmArray = np.zeros(n)
    
    if n == 2:
        Gsorted = G
        return indsorted, Gsorted, dGmArray
    
    for step in range(n-1, 0, -1):
        dG = np.zeros(step)
        
        # Calculate deviation for each curve
        for k in range(step):
            g_mean = np.mean(np.delete(G, k, axis=1), axis=1)
            dG[k] = np.mean((G[1:, k] - g_mean[1:]) ** 2)
        
        dGm = np.max(dG)
        m = np.argmax(dG)
        
        dGmArray[step] = dGm
        Gsorted[:, step] = G[:, m]
        indsorted[step] = ind[m]
        
        # Remove the worst autocorrelation function
        G = np.delete(G, m, axis=1)
        ind = np.delete(ind, m)
    
    Gsorted[:, 0] = G[:, 0]
    indsorted[0] = ind[0]

    return indsorted, Gsorted, dGmArray
