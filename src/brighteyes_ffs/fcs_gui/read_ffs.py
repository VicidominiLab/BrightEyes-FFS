from .restore_session import restorelib
import numpy as np


def read_ffs_file(libfile, read='active', returnObj='G'):
    """
    Read an .ffs libfile and return as an object

    Parameters
    ----------
    libfile : str
        path to the .ffs file.
    read : either 'active' or list of 4 numbers
        list indicating which image, ffs file, and correlation should be
        returned
    returnObj : str, optional
        Either 'im', 'file', 'analysis', or 'G'. The default is 'G'.
        Choose to return the image object, the ffs file object, the analysis
        object, or the correlation object, specified in 'read'

    Returns
    -------
    G
        Correlation object (by default) or image, ffs file, or analysis object.

    Example
    -------
    To return the 4rd ffs analysis from the 2nd ffs file that belongs to image 1:
        analysis = read_ffs_file(libfile, read=[0, 1, 3], returnObj='analysis')
    """
    readlist = False
    if read != 'active':
        readlist = True
    
    lib = restorelib(libfile)
    if returnObj=='ffs':
        return lib
    
    # get active image
    nIm = lib.num_images
    if readlist:
        imnr = read[0]
    else:
        imnr = lib.active_image
    if nIm < 1 or nIm <= imnr:
        return None
    im = lib.lib[imnr]
    if returnObj=='im':
        return im
    
    # get active file
    Nfiles = im.num_files
    if readlist:
        filenr = read[1]
    else:
        filenr = im.active_ffs
    if Nfiles < 1 or Nfiles <= filenr:
        return None
    file = im.ffs_list[filenr]
    if returnObj=='file':
        return file
    
    # get active analysis
    Nanal = file.num_analyses
    if readlist:
        analnr = read[2]
    else:
        analnr = file.active_analysis
    if Nanal < 1 or Nanal <= analnr:
        return None
    analysis = file.analysis_list[analnr]
    if returnObj=='analysis':
        return analysis
    
    if returnObj == 'G':
        G = analysis.corrs
        return G
    
    # get active fit
    Nfits = analysis.num_fits
    if readlist:
        fitnr = read[3]
    else:
        fitnr = analysis.active_fit
    if Nfits < 1 or Nfits <= fitnr:
        return None
    fit = analysis.fits[fitnr]

    return fit


def read_fitresults_from_ffs(libfile, read='active'):
    """
    Read fit results from a saved ffs session

    Parameters
    ----------
    libfile : path
        Path to .ffs file.
    read : str or list, optional
        Either 'active' to read the active file.
        Otherwise list with four elemetns. The default is 'active'.

    Returns
    -------
    fitresults : np.array()
        2D array [fitted values x fitted curve].

    """
    fitObj = read_ffs_file(libfile, read=read, returnObj='fit')
    num_curves = len(fitObj.fit_all_curves)
    if num_curves < 1 or num_curves is None:
        return None
    fitresults = []
    for i in range(num_curves):
        is_fitted = fitObj.fit_all_curves[i].fitarray
        fitObj.fitresults_mem
        fitfun_label = fitObj.fit_all_curves[i].fitfunction_label
        if fitfun_label == 'Maximum entropy method free diffusion':
            analysis = read_ffs_file(libfile, read=read, returnObj='analysis')
            Gsingle = analysis.get_corr()
            tau = Gsingle[:,0]
            [fitresults_mem, tauD, _, _] = fitObj.fitresults_mem(tau, 7)
            return np.transpose(fitresults_mem), tauD
        else:
            fitresults.append((fitObj.fit_all_curves[i].startvalues[is_fitted[0:-1]]))
    fitresults = np.asarray(fitresults)
    return fitresults


def read_g_from_ffs(libfile, read='active'):
    """
    Read an .ffs libfile and return the active correlation curves and fits

    Parameters
    ----------
    libfile : str
        path to the .ffs file.

    Returns
    -------
    G : np.array()
        2D array with for each column a correlation curve.
    tau : np.array()
        1D array with the corresponding tau values.
    Gfit : np.array()
        2D array with for each column the fitted correlation curve.
    taufit : np.array()
        1D array with the corresponding tau values.

    Note: tau and taufit may differ because the fit range is not always
    the full data range

    """
    
    analysis = read_ffs_file(libfile, read=read, returnObj='analysis')
    elements = analysis.settings.elements # central, sum3x3, sum5x5
    Ncurves = len(elements)
    
    Gsingle = analysis.get_corr(elements[0])
    N = len(Gsingle)
    
    G = np.zeros((N, Ncurves))
    Gfit = None
    tau = None
    taufit = None
    fitFound = False
    
    for i in range(Ncurves):
        Gsingle = analysis.get_corr(elements[i])
        tau = analysis.get_corr()[:,0]
        if 'crossCenterAv' in elements:
            G = Gsingle
        else:       
            G[:,i] = Gsingle[:,1]
            
        
        if len(analysis.fits) > 0 and analysis.active_fit is not None:
        
            if read == 'active':
                fitnr = analysis.active_fit
            else:
                fitnr = read[3]
            fits = analysis.fits[fitnr]
        
            fitrange = fits.fit_all_curves[0].fitrange
            start = fitrange[0]
            stop = fitrange[1]
            
            if fitFound == False:
                Gfit = np.zeros((stop-start, Ncurves))
                taufit = tau[start:stop]
                fitFound = True
        
            fitres = fits.fit_all_curves[i].fitresult
            if len(fitres) == len(Gsingle[start:stop,1]):
                Gfit[:, i] = Gsingle[start:stop,1] - fitres
        
    return G, tau, Gfit, taufit