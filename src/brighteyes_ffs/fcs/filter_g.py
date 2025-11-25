import numpy as np


def good_chunks_from_g_obj(G, filt='sum5', f_acc=0.66):
    """
    Check for bad chunks in autocorrelations object

    Parameters
    ----------
    G : Correlations object
        Object with fields for correlation curves, e.g. G.sum5_chunk0.
    filt : str, optional
        Choose which correlation type to use for filtering. The default is 'sum5'.
    f_acc : float, optional
        Acceptance ratio. The default is 0.66, meaning that the 1/3 worst
        correlation curves are rejected. Quality is defined as how close to the
        mean a curve is

    Returns
    -------
    idx : list
        List with the indices of the good chunks

    """
    # make boolean list with True for good chunks, False for bad chunks
    chunks_on = check_chunks_from_g_obj(G, filt, f_acc)
    # return index of good chunks
    idx = list(np.nonzero(chunks_on)[0])
    return idx


def check_chunks_from_g_obj(G, filt='sum5', f_acc=0.66):
    """
    Check for bad chunks in autocorrelations

    Parameters
    ----------
    G : Correlations object
        Object with fields for correlation curves, e.g. G.sum5_chunk0.
    filt : str, optional
        Choose which correlation type to use for filtering. The default is 'sum5'.
    f_acc : float, optional
        Acceptance ratio. The default is 0.66, meaning that the 1/3 worst
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
    chunks_on = np.zeros((n_chunks), dtype=int)
    indsorted = indsorted[0:int(np.round(f_acc*n_chunks))]
    chunks_on[indsorted] = 1
    return chunks_on
    

def sort_g_obj(G, filt):
    Garray, tau = G.get_corrs(filt)
    [indsorted, Gsorted, dGmArray] = sort_g_array(Garray)
    
    return indsorted, Gsorted, dGmArray
    

def sort_g_array(G, optimize=True):
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
    ind = np.arange(n)  # keeps track of the indices of the curves
    indsorted = np.zeros(n, dtype=int)
    Gsorted = np.zeros(Gsize)
    dGmArray = np.zeros(n)
    
    if n == 2:
        Gsorted = G
        return indsorted, Gsorted, dGmArray
    
    if optimize:
        while n > 250:
            # Global mean over all curves (no exclusion)
            g_mean = np.mean(G, axis=1)
            # Compute deviations of each curve vs global mean, ignoring lag 0 as in original
            diffs = G[1:, :] - g_mean[1:, None]
            dG = np.mean(diffs**2, axis=0)
            
            # Sort curves from best (smallest deviation) to worst (largest)
            m_all = np.argsort(dG)
            n_disc = 1 #max(1, int(0.001 * n))
            # indices of the curves that need to be removed
            m = m_all[-n_disc:]
           
            Gsorted[:,n-n_disc:n] = G[:, m]
            indsorted[n-n_disc:n] = ind[m]
            dGmArray[n-n_disc:n] = dG[n-n_disc:n]
            
            # Remove the worst autocorrelation function
            G = np.delete(G, m, axis=1)
            ind = np.delete(ind, m)
            
            n -= n_disc
    
    for step in range(n-1, 0, -1):
        dG = np.zeros(step)
        
        # Calculate deviation for each curve
        g_mean = np.sum(G, 1)
        g_mean = np.tile(g_mean[:, None], 5)
        
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
