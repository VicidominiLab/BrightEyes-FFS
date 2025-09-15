import numpy as np
from scipy.optimize import least_squares, minimize
from .simulate_pch import simulate_pch_1c, simulate_pch_1c_mc_ntimes, simulate_pch_nc
from .generate_psf import generate_3d_gaussian

def fit_pch(hist, fit_info, param, psf, lBounds=[1e-10,1e-10,0,0], uBounds=[1e10,1e10,1e10,1e10], weights=1, n_bins=1e5, fitfun='fitfun_pch', minimization='relative'):
    """
    Fit PCH to the FIDA model

    Parameters
    ----------
    hist : 1D np.array()
        Photon counting histogram, from 0 to N-1 in steps of 1.
        Will be normalized to np.sum(hist) == 1 before fitting
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
    
    # reshape 3D psf to 1D with weights for faster calculation
    bins = np.linspace(0, 1, int(n_bins))
    psf_reshaped = np.reshape(psf, psf.size)
    psf_hist = np.histogram(psf_reshaped, bins)
    psf_compressed = psf_hist[1][1:]
    psf_weights = psf_hist[0]
    
    fit_info = np.asarray(fit_info)
    param = np.asarray(param)
    lBounds = np.asarray(lBounds)
    uBounds = np.asarray(uBounds)
    
    if fitfun == 'fitfun_pch' or fitfun == fitfun_pch:
        fitfun = fitfun_pch
        n_comp = 1
    else:
        fitfun = fitfun_pch_nc
        n_comp = int((len(fit_info) - 3) / 2)
    
    if minimization == 'relative' or minimization == 'absolute':
        param[0:2*n_comp] = np.log10(np.clip(param[0:2*n_comp], 1e-10, None)) # use log10 of concentration and brightness for fitting
        lBounds[0:2*n_comp] = np.log10(np.clip(lBounds[0:2*n_comp], 1e-10, None))
        uBounds[0:2*n_comp] = np.log10(np.clip(uBounds[0:2*n_comp], 1e-10, None))
        
        fitparam_start = param[fit_info==1]
        fixed_param = param[fit_info==0]
        lowerBounds = lBounds[fit_info==1]
        upperBounds = uBounds[fit_info==1]
        
        # use least squares
        fitresult = least_squares(fitfun, fitparam_start, args=(fixed_param, fit_info, hist/np.sum(hist), psf_compressed/np.max(psf_compressed), psf_weights, weights, minimization), bounds=(lowerBounds, upperBounds)) #, xtol=1e-12
        fitresult.fun /= weights
        if minimization == 'relative':
            fitresult.fun *= hist/np.sum(hist)
        
        # go back from log10 scale to original scale
        param[fit_info==1] = fitresult.x
        param[0:2*n_comp] = 10**param[0:2*n_comp]
        fitresult.x = param[fit_info==1]
    else:
        # do MLE
        fitparam_start = param[fit_info==1]
        fixed_param = param[fit_info==0]
        lowerBounds = lBounds[fit_info==1]
        upperBounds = uBounds[fit_info==1]
        # do fit with MLE
        fitresult = minimize(fitfun_pch_nc, x0=fitparam_start, args=(fixed_param, fit_info, hist/np.sum(hist), psf_compressed/np.max(psf_compressed), psf_weights, weights, minimization), bounds=list(zip(lowerBounds, upperBounds)))
        param[fit_info==1] = fitresult.x
        # calculate residuals
        concentration = list(param[0:n_comp])
        brightness = list(param[n_comp:2*n_comp])
        bg = param[2*n_comp]
        T = param[2*n_comp+1]
        dV0 = param[2*n_comp+2]
        yfit = simulate_pch_nc(psf_compressed/np.max(psf_compressed), dV=psf_weights, k_max=len(hist), c=concentration, q=brightness, T=T, dV0=dV0, b=bg)
        fitresult.fun = hist/np.sum(hist) - yfit
        fitresult.x = param[fit_info==1]
    
    if type(psf)==list:
        psf = [w0, z0/w0]
  
    return fitresult


def fitfun_pch(fitparam, fixedparam, fit_info, hist, psf, psf_weights=1, weights=1, minimize_rel_error=False):
    """
    pch fit function
    
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
    
    # calculate absolute residuals
    res = hist - pch_theo
    
    # calculate relative residuals
    if minimize_rel_error:
        res /= (hist + 1e-100)
        res[hist==0] = 0
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_pch_nc(fitparam, fixedparam, fit_info, hist, psf, psf_weights=1, weights=1, minimization='absolute'):
    """
    pch fit function n components
    
    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [log10(all concentrations), log10(all brightness), time, voxel_volume, bg]
        E.g. if only concentration and brightness are fitted, this becomes a two
        element vector [-2, -3].
    fixedparam : 1D np.array
        List with values for the fixed parameters
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
    
    n_comp = int((len(fit_info) - 3) / 2)
    
    all_param = np.float64(np.zeros(len(fit_info)))
    all_param[fit_info==1] = fitparam
    all_param[fit_info==0] = fixedparam
    
    if minimization == 'absolute' or minimization == 'relative':
        concentration = list(10**all_param[0:n_comp])
        brightness = list(10**all_param[n_comp:2*n_comp])
    else:
        concentration = list(all_param[0:n_comp])
        brightness = list(all_param[n_comp:2*n_comp])
    bg = all_param[2*n_comp]
    T = all_param[2*n_comp+1]
    dV0 = all_param[2*n_comp+2]
    
    # calculate theoretical autocorrelation function
    pch_theo = simulate_pch_nc(psf, dV=psf_weights, k_max=len(hist), c=concentration, q=brightness, T=T, dV0=dV0, b=bg)
    
    if minimization == 'absolute' or minimization == 'relative':
        # calculate absolute residuals
        res = hist - pch_theo
    
        if minimization == 'relative':
            # calculate relative residuals
            res /= (hist + 1e-100)
            res[hist==0] = 0
    
        # calculate weighted residuals
        res *= weights
    
        return res
    
    # use mle -> calculate negative log likelihood
    pch_theo = np.maximum(pch_theo, 1e-300)
    pch_theo = pch_theo / np.sum(pch_theo)
    nll = -np.sum(hist * np.log(pch_theo))
    
    return nll


def fitfun_pch_mc(fitparam, fixedparam, fit_info, hist, psf, weights=1):
    """
    pch free diffusion fit function using MC simulation
    
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