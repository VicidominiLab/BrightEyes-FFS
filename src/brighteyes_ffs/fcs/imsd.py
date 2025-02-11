from ..tools.fit_gauss_2d import fit_gauss_2d
from ..tools.fit_curve import fit_curve
import numpy as np

def fcs2imsd(G, tau, fitparam=[1, 1, 0, 0], startvalues=[1,1,1,1], lbounds=[-1e9, -1e9, -1e9, -1e9], ubounds=[1e9, 1e9, 1e9, 1e9], remove_outliers=False, remove_afterpulsing=True):
    """
    Perform iMSD analysis on FFS data.
    Note that this assumes that all detector elements have a similar PSF.
    This assumption limits the accuracy of the analysis.

    Parameters
    ----------
    G : np.array(tau, xi, psi)
        3D array with G vs (tau, shift_x, shift_y).
    tau : np.array()
        Time points.
    fitparam : list with 4 elements, optional
        The first two elements describe the slope and offset of the curve.
        The other two elements are currently not used.
        The default is [1, 1, 0, 0].
    startvalues : list with 4 elements, optional
        Start values for the fit parameters.
    lbounds : list with 4 elements, optional
        DESCRIPTION. The default is [-1e9, -1e9, -1e9, -1e9].
    ubounds : list with 4 elements, optional
        DESCRIPTION. The default is [1e9, 1e9, 1e9, 1e9].
    remove_outliers : boolean, optional
        Remove outliers before fitting the iMSD curve. The default is False.
    remove_afterpulsing : boolean, optional
        Set weights of G(tau, 0, 0) to zero. The default is True.

    Returns
    -------
    var : np.array()
        Variance of the fitted Gaussians for each tau point.
    taunew : np.array()
        Corresponding tau values.
    fitresult : list
        List of fit results.

    """
    # G is 3D array with G(tau, xi, psi)
    # param = [D, offset, smoothing, pixel size]
        
    smoothing = int(startvalues[2])
    newlen = int(len(tau)//smoothing)
    var = np.zeros((newlen))
    taunew = np.zeros((newlen))
    
    # convert D to slope
    D = startvalues[0]
    rho = 1e-3 * startvalues[3] # µm
    slope = 2 / rho**2 * D
    startvalues[0] = slope
    
    # find sigma as a function of tau
    for i in range(int(len(tau)//smoothing)):
        # [x0, y0, A, sigma, offset]
        Gsingle = np.sum(G[i*smoothing:(i+1)*smoothing,:,:],0) / smoothing
        # remove central afterpulsing peak
        weights = 1
        if remove_afterpulsing:
            weights = np.ones(Gsingle.shape)
            weights[4, 4] = 0
        fitres = fit_gauss_2d(Gsingle, [1,1,1,1,1], [1,1,np.max((G[i,1,1]+1e-5, 0)),1,0])
        var[i] = fitres.x[3]**2
        taunew[i] = np.mean(tau[i*smoothing:(i+1)*smoothing])
    
    # remove outliers
    if remove_outliers:
        median_var = np.median(var)
        mask = var < 3 * median_var
        var = var[mask]
        taunew = taunew[mask]
        
    # fit linear curve
    fitres = fit_curve(var, taunew, 'linear', fitparam[0:2], startvalues[0:2], lbounds[0:2], ubounds[0:2], savefig=0)
    
    fitresult = startvalues
    fitresult[fitparam] = fitres.x
    
    # convert slope to D
    fitresult[0] *= (1e-3*startvalues[-1])**2 / 2
    
    return var, taunew, fitresult