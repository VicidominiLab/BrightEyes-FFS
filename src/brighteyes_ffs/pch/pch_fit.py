import numpy as np
from scipy.optimize import least_squares
from .simulate_pch import simulate_pch_1c, simulate_pch_1c_mc_ntimes
from .generate_psf import generate_3d_gaussian

def fit_pch(hist, fit_info, param, psf, lBounds=[-10,-10,-10,-10], uBounds=[10,10,10,10], weights=1, n_bins=1e5, fitfun='fitfun_pch'):
    """
    Fit PCH to the FIDA model

    Parameters
    ----------
    hist : 1D np.array()
        Photon counting histogram, from 0 to N-1 in steps of 1.
    fit_info : 1D np.array
        np.array boolean vector with always 4 elements
        [concentration, brightness, time, voxel_volume]
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit concentration and brightness, this becomes [1, 1, 0, 0]
    param : 1D np.array
        np.array vector with always 4 elements containing the starting values
        for the fit, same order as fit_info
    psf : 3D np.array
        3D array with psf values, normalized to np.max(psf) = 1.
        Alternatively a list of two values [w0, z0] with the beam waist
        (1/exp(-2) values) assuming a Gaussian focal volume
    lBounds : 1D np.array
        log10(lower bounds) for ALL 4 parameters for the fit.
    uBounds : 1D np.array
        log10(upper bounds) for ALL 4 parameters for the fit.
    weights : 1D np.array, optional
        Same dimensions as hist, weights for the fit. The default is 1.

    Returns
    -------
    fitresult : object
        Fit result, output from least_squares.

    """
    
    # check psf
    if type(psf)==list:
        # assume Gaussian
        w0 = psf[0] # nm
        z0 = psf[1] * w0 # nm
        psf = generate_3d_gaussian((200,200,200), w0, z0, px_xy=10.0, px_z=20.0)
    
    # normalize psf
    psf /= np.max(psf)
    
    fit_info = np.asarray(fit_info)
    param = np.asarray(param)
    lBounds = np.asarray(lBounds)
    uBounds = np.asarray(uBounds)
    
    param[0] = np.log10(param[0]) # use log10 of concentration for fitting
    param[1] = np.log10(param[1]) # use log10 of brightness for fitting
    
    fitparam_start = param[fit_info==1]
    fixed_param = param[fit_info==0]
    lowerBounds = lBounds[fit_info==1]
    upperBounds = uBounds[fit_info==1]
    
    # reshape 3D psf to 1D with weights for faster calculation
    bins = np.linspace(0, 1, int(n_bins))
    psf_reshaped = np.reshape(psf, psf.size)
    psf_hist = np.histogram(psf_reshaped, bins)
    psf_compressed = psf_hist[1][1:]
    psf_weights = psf_hist[0]
    
    if fitfun == 'fitfun_pch':
        fitfun = fitfun_pch
    
    fitresult = least_squares(fitfun, fitparam_start, args=(fixed_param, fit_info, hist/np.sum(hist), psf_compressed/np.max(psf_compressed), psf_weights, weights), bounds=(lowerBounds, upperBounds)) #, xtol=1e-12
    fitresult.fun /= weights
    
    # go back from log10 scale to original scale
    j = 0
    for i in range(2):
        if fit_info[i]:
            fitresult.x[j] = 10**fitresult.x[j]
            j += 1
  
    if type(psf)==list:
        psf = [w0, z0/w0]
  
    return fitresult


def fitfun_pch(fitparam, fixedparam, fit_info, hist, psf, psf_weights=1, weights=1):
    """
    fcs free diffusion fit function
    
    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [log10(concentration), log10(brightness), time, voxel_volume]
        E.g. if only concentration and brightness are fitted, this becomes a two
        element vector [-2, -3].
    fixedparam : 1D np.array
        List with values for the fixed parameters:
        order: [log10(concentration), log10(brightness), time, voxel_volume]
        same principle as fitparamStart.
    fit_info : 1D np.array
        np.array boolean vector with always 4 elements
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit concentration and brightness, this becomes [1, 1, 0, 0]
    hist : 1D np.array
        Vector with pch values (normalized to sum=1).
    psf : 3D np.array
        3D array with psf values, normalized to np.max(psf) = 1.
    weights : 1D np.array, optional
        Vector with pch weights. The default is 1.

    Returns
    -------
    res : 1D np.array
        Weighted residuals.

    """
    
    all_param = np.float64(np.zeros(4))
    all_param[fit_info==1] = fitparam
    all_param[fit_info==0] = fixedparam
    
    concentration = 10**all_param[0]
    brightness = 10**all_param[1]
    T = all_param[2]
    dV0 = all_param[3]

    # calculate theoretical autocorrelation function
    pch_theo = simulate_pch_1c(psf, dV=psf_weights, k_max=len(hist), c=concentration, q=brightness, T=T, dV0=dV0)
    
    # calculate residuals
    res = hist - pch_theo
    
    # calculate weighted residuals
    res *= weights
    
    return res

def fitfun_pch_mc(fitparam, fixedparam, fit_info, hist, psf, weights=1):
    """
    fcs free diffusion fit function
    
    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [N, tauD, SP, offset, A, B]
        E.g. if only N and tauD are fitted, this becomes a two
        element vector [1, 1e-3].
    fixedparam : 1D np.array
        List with values for the fixed parameters:
        order: [N, tauD, SP, offset, 1e6*A, B]
        same principle as fitparamStart.
    fit_info : 1D np.array
        np.array boolean vector with always 6 elements
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit N and tau D this becomes [1, 1, 0, 0, 0, 0]
        order: [N, tauD, SP, offset, 1e6*A, B].
    tau : 1D np.array
        Vector with tau values.
    yexp : 1D np.array
        Vector with experimental autocorrelation.
    weights : 1D np.array, optional
        Vector with weights. The default is 1.

    Returns
    -------
    res : 1D np.array
        Residuals.

    """
    
    all_param = np.float64(np.zeros(6))
    all_param[fit_info==1] = fitparam
    all_param[fit_info==0] = fixedparam
    
    concentration = all_param[0]
    brightness = all_param[1]
    n_samples = int(all_param[2])
    n_hist_max = int(all_param[3])
    max_bin = int(all_param[4])
    err = all_param[5]

    # calculate theoretical autocorrelation function    
    pch_theo, _, _, _, _ = simulate_pch_1c_mc_ntimes(psf, concentration, brightness, n_samples, n_hist_max, max_bin, err)
    
    # calculate residuals
    res = hist - pch_theo
    
    # calculate weighted residuals
    res *= weights
    
    return res