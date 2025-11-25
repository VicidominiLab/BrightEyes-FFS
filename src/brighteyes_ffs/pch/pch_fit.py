import numpy as np
from scipy.optimize import least_squares, minimize
from .simulate_pch import simulate_pch_1c, simulate_pch_1c_mc_ntimes, simulate_pch_nc
from .generate_psf import generate_3d_gaussian


def fit_pch(hist, fit_info, param, psf, lBounds=[1e-10,1e-10,0,0], uBounds=[1e10,1e10,1e10,1e10], weights=1, n_draws=1, n_bins=1e5, global_param=None, fitfun='fitfun_pch', minimization='relative'):
    """
    Fit photon counting histogram to the FIDA model

    Parameters
    ----------
    hist : 1D or 2D np.array()
        Photon counting histogram, from 0 to N-1 in steps of 1.
        Will be normalized to np.sum(hist) == 1 before fitting
        In case of a global fit, 2D np.array(K x L) with L the number of
        histograms
    fit_info : 1D np.array
        np.array boolean vector with always 4 or 5 elements
        [concentration, brightness, time, voxel_volume] for fitfun_pch
        [concentration, brightness, background, time, voxel_volume] for fitfun_pch_nc and fitfun_pch_nc_global
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit concentration and brightness, this becomes [1, 1, 0, 0]
        or [1, 1, 0, 0, 0], depending on the chosen fit function
        When using a multispecies fit, the array has always 2*n+3 elements
        with n the number of species, e.g. for n = 2:
        [c1, c2, q1, q2, bg, t, V] with c and q conc and brightness
        When using a global fit, the array is [2*n+3, n_hist] with n_hist
        the number of histograms to fit
    param : 1D np.array
        np.array vector with always 4, 5, or 2*n+3 elements containing the
        starting values for the fit, same order as fit_info.
    psf : 3D np.array
        3D array with psf values, normalized to np.max(psf) = 1.
        Alternatively a list of two values [w0, z0] with the beam waist
        (1/exp(-2) values) assuming a Gaussian focal volume
        For multiple histograms in a global fit:
            psf = [w0, z0, w1, z1, ..., wn, zn]
            or psf = np.array(n_hist, n_z, n_y, n_z)
    lBounds : 1D np.array
        log10(lower bounds) for ALL parameters for the fit (also the fixed ones).
        For multiple hisograms, the same bounds are taken.
    uBounds : 1D np.array
        log10(upper bounds) for ALL parameters for the fit.
    weights : 1D np.array, optional
        Same dimensions as hist, weights for the fit. The default is 1.
    n_bins : int
        Number of bins used in the histogram of the PSF. The higher this number,
        the more accurate the result
    global_param : boolean list
        Only used with fitfun_pch_nc_global    
        List of length equal to fit_info length
        True for parameters that must be identical for all curves in a global fit
        False for parameters that can vary for each curve.
        E.g. Use [True, False, False, False, False] for a 1 component fit
        with the concentration identical for all curves, while the brightness
        may vary

    Returns
    -------
    fitresult : object
        Fit result, output from least_squares.
        fitresult.fun contains the residuals
        fitresult.x contains all parameters, fitted and fixed

    """
    
    # check which fit function, number of species, and number of histograms
    fitfun, n_comp, n_hist, k_max = parse_fitfun_and_fitinfo(fitfun, fit_info, hist)
    
    # generate and/or compress PSFs
    psf_compressed, psf_weights = generate_psfs(psf, n_hist, n_bins)
    
    fit_info = np.asarray(fit_info)
    param = np.asarray(param)
    lBounds = np.asarray(lBounds)
    uBounds = np.asarray(uBounds)
    
    # ----- least squares -----
    if minimization == 'relative' or minimization == 'absolute':
        param[0:2*n_comp] = np.log10(np.clip(param[0:2*n_comp], 1e-10, None)) # use log10 of concentration and brightness for fitting
        lBounds[0:2*n_comp] = np.log10(np.clip(lBounds[0:2*n_comp], 1e-10, None))
        uBounds[0:2*n_comp] = np.log10(np.clip(uBounds[0:2*n_comp], 1e-10, None))
        
        # global
        if fitfun == fitfun_pch_nc_global:
            fitparam_start, fixed_param, lowerBounds, upperBounds = make_fit_parameters_global_fit(param, fit_info, n_hist, global_param, lBounds, uBounds)
            fitresult = least_squares(fitfun, fitparam_start, args=(fixed_param, fit_info, hist, psf_compressed, global_param, psf_weights, weights, minimization), bounds=(lowerBounds, upperBounds)) #, xtol=1e-12
            fitresult.x = global_fit_result_to_parameters(param, fit_info, n_hist, global_param, fitresult.x)
            fitresult.x[0:2*n_comp,:] = 10**fitresult.x[0:2*n_comp,:]
            fitresult.fun = global_fit_result_to_residuals(fitresult.fun, hist, n_hist, weights, minimization)
        
        else:
            # single fit
            fitparam_start = param[fit_info==1]
            fixed_param = param[fit_info==0]
            lowerBounds = lBounds[fit_info==1]
            upperBounds = uBounds[fit_info==1]
            
            fitresult = least_squares(fitfun, fitparam_start, args=(fixed_param, fit_info, hist, psf_compressed, psf_weights, weights, n_draws, minimization), bounds=(lowerBounds, upperBounds)) #, xtol=1e-12
            fitresult.fun /= weights
            if minimization == 'relative':
                fitresult.fun *= hist/np.sum(hist)
            # go back from log10 scale to original scale
            param[fit_info==1] = fitresult.x
            param[0:2*n_comp] = 10**param[0:2*n_comp]
            fitresult.x = param
        
        lBounds[0:2*n_comp] = 10**lBounds[0:2*n_comp]
        uBounds[0:2*n_comp] = 10**uBounds[0:2*n_comp]
        
        return fitresult
    
    # ----- MLE -----
    # global
    if fitfun == fitfun_pch_nc_global:
        fitparam_start, fixed_param, lowerBounds, upperBounds = make_fit_parameters_global_fit(param, fit_info, n_hist, global_param, lBounds, uBounds)
        fitresult = minimize(fitfun, x0=fitparam_start, args=(fixed_param, fit_info, hist, psf_compressed, global_param, psf_weights, weights, minimization), bounds=list(zip(lowerBounds, upperBounds)))
        fitresult.x = global_fit_result_to_parameters(param, fit_info, n_hist, global_param, fitresult.x)
        # mle does not return residuals, calculate them now
        fit_info = [False for i in range(len(fitresult.x))]
        fitparam_start, fixed_param, _, _ = make_fit_parameters_global_fit(fitresult.x, fit_info, n_hist, global_param, lBounds, uBounds)
        fitresult.fun = fitfun_pch_nc_global(fitparam_start, fixed_param, fit_info, hist, psf_compressed, global_param, psf_weights, weights=weights, minimization='mle', return_residuals=True)
        fitresult.fun = global_fit_result_to_residuals(fitresult.fun, hist, n_hist, weights, minimization)
        return fitresult
        
    # single fit
    fitparam_start = param[fit_info==1]
    fixed_param = param[fit_info==0]
    lowerBounds = lBounds[fit_info==1]
    upperBounds = uBounds[fit_info==1]
    fitresult = minimize(fitfun_pch_nc, x0=fitparam_start, args=(fixed_param, fit_info, hist/np.sum(hist), psf_compressed/np.max(psf_compressed), psf_weights, weights, n_draws, minimization), bounds=list(zip(lowerBounds, upperBounds)))
    param[fit_info==1] = fitresult.x
    # calculate residuals
    concentration = list(param[0:n_comp])
    brightness = list(param[n_comp:2*n_comp])
    bg = param[2*n_comp]
    T = param[2*n_comp+1]
    dV0 = param[2*n_comp+2]
    yfit = simulate_pch_nc(psf_compressed/np.max(psf_compressed), dV=psf_weights, k_max=len(hist), c=concentration, q=brightness, T=T, dV0=dV0, bg=bg)
    fitresult.fun = hist/np.sum(hist) - yfit
    fitresult.x = param
    return fitresult


def fitfun_pch(fitparam, fixedparam, fit_info, hist, psf, psf_weights=1, weights=1, n_draws=1, minimization='absolute'):
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
    if minimization == 'relative':
        res /= (hist + 1e-100)
        res[hist==0] = 0
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_pch_nc(fitparam, fixedparam, fit_info, hist, psf, psf_weights=1, weights=1, n_draws=1, minimization='absolute', return_residuals=False):
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
    return_residuals : boolean
        Only needed for MLE analysis
        If True, returns the residuals between theoretical and experimental
        pch;
        if False, returns the negative log likelihood (default)

    Returns
    -------
    res : 1D np.array
        Weighted residuals or negative log-likelihood.

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
    pch_theo = simulate_pch_nc(psf, dV=psf_weights, k_max=len(hist), c=concentration, q=brightness, T=T, dV0=dV0, bg=bg, n_draws=n_draws)
    
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
    
    if return_residuals:
        return hist - pch_theo
    
    return nll


def fitfun_pch_nc_global(fitparam, fixedparam, fit_info, hist, psf, global_param=None, psf_weights=1, weights=1, minimization='absolute', return_residuals=False):
    """
    pch fit function n components with global fit parameters
    
    Parameters
    ----------
    fitparamStart : 1D np.array
        Array with starting values for the fit parameters:
        order: [log10(concentrations), log10(brightness), time, voxel_volume, bg]
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
    global_param : list with 
    weights : 1D np.array, optional
        Vector with pch weights. The default is 1.
    return_residuals : boolean
        Only needed for MLE analysis
        If True, returns the residuals between theoretical and experimental
        pch;
        if False, returns the negative log likelihood (default)

    Returns
    -------
    res : 1D np.array
        Weighted residuals.

    """
   
    # number of parameters (fitted and fixed combined)
    n_param = len(fit_info)
    
    # number of species
    n_comp = int((len(fit_info) - 3) / 2)
    
    # number of histogram curves to fit and number of bins in each histogram
    n_hist, k_max = hist_param(hist)
    
    # if global_param is None, assume none of the parameters is globally fitted
    if global_param is None:
        global_param = [False for i in range(n_param)]
    
    # build 2D array with all parameters for all curves
    all_param = make_2D_fit_parameter_array_global_fit(fitparam, fixedparam, fit_info, global_param, n_param, n_hist)
    
    if minimization == 'absolute' or minimization == 'relative':
        # use log10 for more robust fitting of c and q
        all_param[0:2*n_comp,:] = 10**all_param[0:2*n_comp,:]
    
    # calculate theoretical pch function
    pch_theo = np.concatenate([simulate_pch_nc(psf[i], dV=psf_weights[i], k_max=k_max, c=list(all_param[0:n_comp,i]), q=list(all_param[n_comp:2*n_comp,i]), T=all_param[2*n_comp+1,i], dV0=all_param[2*n_comp+2,i], bg=all_param[2*n_comp,i]) for i in range(n_hist)])
    pch_exp = np.concatenate([hist[:,i] for i in range(n_hist)])
    
    if minimization == 'absolute' or minimization == 'relative':
        # set values back to log10
        all_param[0:2*n_comp,:] = np.log10(all_param[0:2*n_comp,:])
        
        # calculate absolute residuals
        res = pch_exp - pch_theo
    
        if minimization == 'relative':
            # calculate relative residuals
            res /= (pch_exp + 1e-100)
            res[pch_exp==0] = 0
    
        # calculate weighted residuals
        res *= weights
    
        return res
    
    # use mle -> calculate negative log likelihood
    pch_theo = np.maximum(pch_theo, 1e-300)
    pch_theo = pch_theo / np.sum(pch_theo) * n_hist
    nll = -np.sum(pch_exp * np.log(pch_theo))
    
    if return_residuals:
        return pch_exp - pch_theo
    
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


def hist_param(hist):
    # number of histogram curves to fit
    if hist.ndim == 1:
        n_hist = 1
        k_max = len(hist)
    else:
        n_hist = hist.shape[1]
        k_max = hist.shape[0]
    return n_hist, k_max


def make_fit_parameters_global_fit(param, fit_info, n_hist, global_param, lBounds, uBounds):
    """
    Turn a 2D fit array with the parameter values for each curve into a 1D array
    of parameters for conventional fitting

    Parameters
    ----------
    param : 2D np.array()
        array of size [K x n_hist] with K the number of parameters per histogram
        and n_hist the number of histograms. For 1 species, K = 5
    fit_info : 1D boolean np.array()
        array of size length K with 1s for fit parameters and 0s for fixed parameters.
    n_hist : int
        Number of histograms to fit in the global fit.
    global_param : 1D boolean np.array()
        array of size length K with 1s for global parameters and 0s for individual parameters.
    lBounds : 1D np.array(K)
        Lower bounds for all fitted and fixed parameters.
    uBounds : 1D np.array(K)
        Upper bounds for all fitted and fixed parameters.

    Returns
    -------
    fitparam_start : 1D np.array(M)
        Start values for the fitted parameters.
    fixed_param : 1D np.array(N)
        Start values for the fixed parameters.
    lowerBounds : 1D np.array(M)
        Lower bounds for the fit parameters.
    upperBounds : 1D np.array(M)
        Upper bounds for the fit parameters.

    """
    fitparam_start = []
    fixed_param = []
    lowerBounds = []
    upperBounds = []
    
    for i in range(len(fit_info)):
        if fit_info[i]:
            # parameter is fitted
            if global_param[i]:
                # parameter is global
                fitparam_start.append(param[i, 0])
                lowerBounds.append(lBounds[i])
                upperBounds.append(uBounds[i])
            else:
                # parameter is unique for each histogram
                for j in range(n_hist):
                    fitparam_start.append(param[i, j])
                    lowerBounds.append(lBounds[i])
                    upperBounds.append(uBounds[i])
        else:
            # parameter is fixed
            if global_param[i]:
                # parameter is global
                fixed_param.append(param[i, 0])
            else:
                # parameter is unique for each histogram
                for j in range(n_hist):
                    fixed_param.append(param[i, j])
    
    return fitparam_start, fixed_param, lowerBounds, upperBounds


def make_2D_fit_parameter_array_global_fit(fitparam, fixedparam, fit_info, global_param, n_param, n_hist):
    """
    Generate a 2D array with all parameter values for all histograms.
    Needed to convert fitted and fixed parameters to a 2D parameter array
    for global fitting
    """
    all_param = np.float64(np.zeros((n_param, n_hist)))
    fit_param_idx = 0
    fixed_param_idx = 0
    for i in range(n_param):
        if fit_info[i]:
            # parameter is fitted
            if global_param[i]:
                # parameter must be identical for all curves
                for j in range(n_hist):
                    all_param[i, j] = fitparam[fit_param_idx]
                fit_param_idx += 1
            else:
                # parameter is different for each curve
                for j in range(n_hist):
                    all_param[i, j] = fitparam[fit_param_idx]
                    fit_param_idx += 1
        else:
            # parameter is fixed
            if global_param[i]:
                # parameter must be identical for all curves
                for j in range(n_hist):
                    all_param[i, j] = fixedparam[fixed_param_idx]
                fixed_param_idx += 1
            else:
                # parameter is different for each curve
                for j in range(n_hist):
                    all_param[i, j] = fixedparam[fixed_param_idx]
                    fixed_param_idx += 1
    return all_param


def global_fit_result_to_parameters(param, fit_info, n_hist, global_param, fitresult_x):
    """
    Turn global fit result ouput into a 2D array with all the parameter values
    for each histogram
    """
    fitresult_idx = 0
    for i in range(len(fit_info)):
        if fit_info[i]:
            # parameter is fitted
            if global_param[i]:
                # parameter is global
                param[i,:] = fitresult_x[fitresult_idx]
                fitresult_idx += 1
            else:
                # parameter is unique for each histogram
                for j in range(n_hist):
                    param[i,j] = fitresult_x[fitresult_idx]
                    fitresult_idx += 1
    
    return param
                

def global_fit_result_to_residuals(fitresult_fun, hist, n_hist, weights, minimization):
    """
    Turn global fit result ouput into a 2D array with all the residual values
    for each histogram
    """
    
    len_hist = len(fitresult_fun) // n_hist
    
    fitresult_fun_out = np.zeros((len_hist, n_hist))
    
    for i in range(n_hist):
        fitresult_fun_out[:, i] = fitresult_fun[i*len_hist:(i+1)*len_hist]
    
    fitresult_fun_out /= weights
    
    if minimization == 'relative':
        for i in range(n_hist):
            fitresult_fun_out[:, i] *= hist[:,i]
        
    return fitresult_fun_out


def parse_fitfun_and_fitinfo(fitfun, fit_info, hist):
    if fitfun == 'fitfun_pch' or fitfun == fitfun_pch:
        fitfun = fitfun_pch
        n_comp = 1
        n_hist = 1
        k_max = None
    elif fitfun == 'fitfun_pch_nc_global' or fitfun == fitfun_pch_nc_global:
        fitfun = fitfun_pch_nc_global
        n_comp = int((len(fit_info) - 3) / 2)
        # number of histogram curves to fit
        n_hist, k_max = hist_param(hist)
    else:
        fitfun = fitfun_pch_nc
        n_comp = int((len(fit_info) - 3) / 2)
        n_hist = 1
        k_max = None
    
    return fitfun, n_comp, n_hist, k_max


def generate_psfs(psf, n_hist, n_bins):
    """
    Generate all PSFs needed for a (global) fit analysis.
    Each PSF is compressed by calculating a histogram of all the PSF values
    in the 3D array, and returning the bin values and corresponding counts

    Parameters
    ----------
    psf : list or np.array()
        If list: [w0, z0, w1, z1, ..., wn, zn]
            with w0, w1, ..., wn the gaussian lateral beam waist (in nm)
            with z0, z1, ..., zn the gaussian axial beam height (in nm)
            n PSFs can be given (e.g. n=3 for central, sum3, sum5)
        If np.array()
            Either 3D array for a single 3D psf (Ny x Nx x Nz)
            Or 4D array for multiple PSF (Nn x Ny x Nx x Nz) for n PSFs
    n_hist : int
        Number of histograms to fit (n_hist = n)
    n_bins : int
        Number of bins used to calculate the compressed PSF.

    Returns
    -------
    psf_compressed : 1D or 2D np.array()
        If n_hist == 1: 1D array with the bin numbers of the compressed histogram.
        If n_hist > 1: psf_compressed[i] is 1D array for i from 0 to n
    psf_weights : 1D or 2D np.array()
        PSF counts for the compressed histogram, same syntax as psf_compressed.

    """
    psf_compressed = np.zeros((n_hist, int(n_bins)))
    psf_weights = np.zeros((n_hist, int(n_bins)))
    if type(psf)==list:
        # assume Gaussian
        for i in range(n_hist):
            w0 = psf[2*i] # nm
            z0 = psf[2*i+1] * w0 # nm
            psf_sim = generate_3d_gaussian((200,200,200), w0, z0, px_xy=10.0, px_z=20.0)
            
            # normalize psf
            psf_sim /= np.max(psf_sim)
            
            # reshape 3D psf to 1D with weights for faster calculation
            bins = np.linspace(0, 1, int(n_bins+1))
            psf_reshaped = np.reshape(psf_sim, psf_sim.size)
            psf_hist = np.histogram(psf_reshaped, bins)
            psf_compressed[i] = psf_hist[1][1:]
            psf_weights[i] = psf_hist[0]
        if n_hist == 1:
            psf_compressed = psf_compressed[0]
            psf_weights = psf_weights[0]
    else:
        bins = np.linspace(0, 1, int(n_bins+1))
        if n_hist == 1:
            # normalize psf
            psf /= np.max(psf)
            # reshape 3D psf to 1D with weights for faster calculation
            psf_reshaped = np.reshape(psf, psf.size)
            psf_hist = np.histogram(psf_reshaped, bins)
            psf_compressed = psf_hist[1][1:]
            psf_weights = psf_hist[0]
        else:
            for i in range(n_hist):
                psf_reshaped = np.reshape(psf[i], psf[i].size)
                psf_hist = np.histogram(psf_reshaped, bins)
                psf_compressed[i] = psf_hist[1][1:]
                psf_weights[i] = psf_hist[0]
    
    # compress further by removing zeros
    if n_hist == 1:
        nonzero_idx = psf_weights != 0
        psf_compressed = psf_compressed[nonzero_idx]
        psf_weights = psf_weights[nonzero_idx]
    else:
        zero_idx = np.sum(np.abs(psf_weights), 0) == 0
        nonzero_idx = zero_idx == False
        psf_compressed = psf_compressed[:, nonzero_idx]
        psf_weights = psf_weights[:, nonzero_idx]
    
    return psf_compressed, psf_weights
