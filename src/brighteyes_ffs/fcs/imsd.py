from ..tools.fit_gauss_2d import fit_gauss_2d
from ..tools.fit_curve import fit_curve
import numpy as np

def fcs2imsd(G, tau, fitparam=[1, 1, 0, 0], startvalues=[1,1,1,1], lbounds=[-1e9, -1e9, -1e9, -1e9], ubounds=[1e9, 1e9, 1e9, 1e9], remove_outliers=False, remove_afterpulsing=False):
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
        Gsingle = np.sum(G[i*smoothing:(i+1)*smoothing,:,:],0)
        # remove central afterpulsing peak
        if remove_afterpulsing:
            Gsingle[4,4] = 0
            Gsingle[4,4] = np.max(Gsingle[3:6,3:6])
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