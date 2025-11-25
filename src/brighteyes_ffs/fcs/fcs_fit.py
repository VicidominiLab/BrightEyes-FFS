import numpy as np
from scipy.optimize import least_squares
from .fcs_analytical import fcs_analytical, fcs_dualfocus, fcs_2c_analytical, fcs_2c_2d_analytical, fcs_analytical_2c_anomalous, nanosecond_fcs_analytical, uncoupled_reaction_diffusion
from .plot_pycorrfit import plot_pycorrfit, PyCorrfit_data
from .mem_fit import mem_fit_free_diffusion
from ..pch.pch_fit import make_fit_parameters_global_fit, make_2D_fit_parameter_array_global_fit, hist_param, global_fit_result_to_parameters, global_fit_result_to_residuals


def fcs_fit(Gexp, tau, fitfun, fit_info, param, lBounds, uBounds, plotInfo, savefig=0, plotTau=True, weights=1, global_param=None):
    """
    Fit experimental fcs data to the analytical model
    Assuming 3D diffusion in a Gaussian focal volume
    No triplet state assumed

    Parameters
    ----------
    Gexp : 1D numpy array
        Vector with experimental autocorrelation curve (G).
    tau : 1D numpy array
        Vector with lag times [s].
    fitfun : string or function
        Fit function (either string or the actual function)
        'fitfun_2c'           Fit two components
        'fitfun_dualfocus'    Fit two-focus fcs
        'fitfun_circfcs'      Circular scanning-fcs.
        'fitfun_free_diffusion_2d'
        'fcs_analytical_2c_anomalous'
    fit_info : 1D numpy array
        np.array boolean vector with [N, tauD, SP, offset, 1e6*A, B, vx, vy,...]
        1 if this value has to be fitted
        0 if this value is fixed during the fit.
    param : 1D numpy array
        np.array vector with start values for [N, tauD, SP, offset, A, B, vx, vy]
        A ~ 1e6*2.5e-8
        B ~ -1.05.
    lBounds : 1D numpy array
        Lower bounds for ALL parameters.
    uBounds : 1D numpy array
        Upper bounds for ALL parameters.
    plotInfo : string
        "central", "sum3", or "sum5". Determines only the color of the plot
        -1 to not plot the figure.
    savefig : scalar or string, optional
        0 to not plot but not save the figure
        file name with extension to save as "png" or "eps". The default is 0.
    plotTau : boolean, optional
        Plot option. The default is True.
    weights : int or 1D np.array(), optional
        Weights for the fit. The default is 1.

    Returns
    -------
    fitresult : object
        Output of least_squares
        fitresult.x: parameter results (both fitted and fixed)
            tauD in [ms].
        fitresult.fun: residuals

    """
    
    if fitfun != 'mem_fit_free_diffusion' and fitfun != mem_fit_free_diffusion and fitfun != fcs_fit_dualfocus:
        fitparamStart = param[fit_info==1]
        fixedparam = param[fit_info==0]
        lowerBounds = lBounds[fit_info==1]
        upperBounds = uBounds[fit_info==1]
    
    if type(fitfun) == str:
        if fitfun == 'fitfun_2c':
            fitresult = least_squares(fitfun_2c, fitparamStart, args=(fixedparam, fit_info, tau, Gexp, weights), bounds=(lowerBounds, upperBounds))
        if fitfun == 'fitfun_an':
            fitresult = least_squares(fitfun_an, fitparamStart, args=(fixedparam, fit_info, tau, Gexp, weights), bounds=(lowerBounds, upperBounds))
        elif fitfun == 'fitfun_dualfocus':
            fitresult = least_squares(fitfun_dualfocus, fitparamStart, args=(fixedparam, fit_info, global_param, tau, Gexp, weights), bounds=(lowerBounds, upperBounds))
        elif fitfun == 'fitfun_circfcs':
            fitresult = least_squares(fitfun_circfcs, fitparamStart, args=(fixedparam, fit_info, tau, Gexp, weights), bounds=(lowerBounds, upperBounds))
        elif fitfun == 'mem_fit_free_diffusion':
            fitresult = mem_fit_free_diffusion(param, tau, Gexp, weights)
        elif fitfun == 'fitfun_free_diffusion_2d':
            fitresult = least_squares(fitfun_free_diffusion_2d, fitparamStart, args=(fixedparam, fit_info, tau, Gexp, weights), bounds=(lowerBounds, upperBounds))
        elif fitfun == 'fcs_analytical_2c_anomalous':
            fitresult = least_squares(fcs_analytical_2c_anomalous, fitparamStart, args=(fixedparam, fit_info, tau, Gexp, weights), bounds=(lowerBounds, upperBounds))
    else:
        if fitfun == fitfun_dualfocus:
            fitresult = least_squares(fitfun_dualfocus, fitparamStart, args=(fixedparam, fit_info, global_param, tau, Gexp, weights), bounds=(lowerBounds, upperBounds))
        elif fitfun == mem_fit_free_diffusion:
            fitresult = mem_fit_free_diffusion(param, tau, Gexp, weights)
        elif fitfun == fcs_fit_dualfocus:
            fitresult = fcs_fit_dualfocus(Gexp, tau, fit_info, param, weights=weights, global_param=global_param)
        else:
            fitresult = least_squares(fitfun, fitparamStart, args=(fixedparam, fit_info, tau, Gexp, weights), bounds=(lowerBounds, upperBounds))

    n_corr, _ = hist_param(Gexp)
    
    if n_corr > 1:
        # multiple curves were fitted simultaneously
        fitresult.x = global_fit_result_to_parameters(param, fit_info, n_corr, global_param, fitresult.x)
        fitresult.fun = global_fit_result_to_residuals(fitresult.fun, Gexp, n_corr, weights, minimization='absolute')
    else:
        # only one curve fitted
        if fitfun != 'mem_fit_free_diffusion' and fitfun != mem_fit_free_diffusion:
            fitresult.fun /= weights
            param[fit_info==1] = fitresult.x
            fitresult.x = param
    
    if plotInfo != -1:
        plot_fit(tau, Gexp, param, fit_info, fitresult, plotInfo, savefig, plotTau)
    
    return fitresult


def fcs_fit_dualfocus(Gexp, tau, fit_info, param, weights=1, global_param=None):
    """
    Fit experimental fcs data to the analytical model
    Assuming 3D diffusion in a Gaussian focal volume
    No triplet state assumed

    Parameters
    ----------
    Gexp : 2D np.array
        2D matrix with correlation curves
        Each column is a correlation curve.
    tau : 1D np.array
        Vector with tau values for a single autocorrelation curve [s].
    fit_info : 1D np.array
        np.array boolean vector with
        [c, D, w0, SF, rhox, rhoy, vx, vy, dc]
        1 if this value has to be fitted
        0 if this value is fixed during the fit.
    param : 2D np.array
        vector with start values for all the parameters.
        Each column corresponds to a different correlation curve.
    plotResults : boolean, optional
        Plot results. The default is True.
    weights : int or 1D np.array()
        Weights for the fit
    global_param : list or None
        Boolean list of length len(fit_info)
        True for global parameters, False for individual parameters

    Returns
    -------
    fitresult : object
        Output of least_squares.

    """
    
    n_corr, _ = hist_param(Gexp)
    n_param = len(fit_info)
    
    if global_param is None:
        global_param = [False for i in range(n_param)]
    
    dummy_bounds = [0 for i in range(len(fit_info))]
    
    if n_corr > 1:
        fitparam_start, fixed_param, _, _ = make_fit_parameters_global_fit(param, fit_info, n_corr, global_param, dummy_bounds, dummy_bounds)
    else:
        fitparam_start = param[fit_info==1]
        fixed_param = param[fit_info==0]
    
    fitresult = least_squares(fitfun_dualfocus, fitparam_start, args=(fixed_param, fit_info, global_param, tau, Gexp, weights))

    return fitresult


def fitfun(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
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
    
    fitparam = np.float64(np.zeros(6))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    N = fitparam[0]
    tauD = fitparam[1]
    SF = fitparam[2]
    offset = fitparam[3]
    A = fitparam[4]
    B = -fitparam[5]

    # calculate theoretical autocorrelation function    
    FCStheo = fcs_analytical(tau, N, tauD, SF, offset, 1e-6*A, B)
    
    # calculate residuals
    res = yexp - FCStheo
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_an(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
    """
    fcs fit anomalous diffusion function

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [N, tauD, SP, offset, alpha]
        E.g. if only N and tauD are fitted, this becomes a two
        element vector [1, 1e-3].
    fixedparam : 1D np.array
        List with values for the fixed parameters:
        order: [N, tauD, SP, offset, alpha]
        same principle as fitparamStart.
    fit_info : 1D np.array
        np.array boolean vector with always 5 elements
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit N and tau D this becomes [1, 1, 0, 0, 0]
        order: [N, tauD, SP, offset, alpha].
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
    
    fitparam = np.float64(np.zeros(5))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    N = fitparam[0]
    tauD = fitparam[1]
    SF = fitparam[2]
    offset = fitparam[3]
    alpha = fitparam[4]

    # calculate theoretical autocorrelation function    
    FCStheo = fcs_analytical(tau, N, tauD, SF, offset, 0, 0, alpha)
    
    # calculate residuals
    res = yexp - FCStheo
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_an_2c(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
    """
    fcs fit anomalous diffusion function

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [N, tauD1, tauD2, alpha1, alpha2, F, T, tautrip, SP, offset]
        E.g. if only N and tauD are fitted, this becomes a two
        element vector [1, 1e-3].
    fixedparam : 1D np.array
        List with values for the fixed parameters:
        order: [N, tauD, SP, offset, alpha]
        same principle as fitparamStart.
    fit_info : 1D np.array
        np.array boolean vector with always 5 elements
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit N and tau D this becomes [1, 1, 0, 0, 0]
        order: [N, tauD, SP, offset, alpha].
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
    
    fitparam = np.float64(np.zeros(11))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    N = fitparam[0]
    tauD1 = fitparam[1]
    tauD2 = fitparam[2]
    alpha1 = fitparam[3]
    alpha2 = fitparam[4]
    F = fitparam[5]
    brightness = fitparam[6]
    T =fitparam[7]
    tau_triplet = fitparam[8]
    SF = fitparam[9]
    offset = fitparam[10]
    
    # calculate theoretical autocorrelation function    
    FCStheo = fcs_analytical_2c_anomalous(tau, N, tauD1, tauD2, alpha1, alpha2, F, T, tau_triplet, SF, offset, brightness)
    
    # calculate residuals
    res = yexp - FCStheo
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_2c(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
    """
    fcs fit function two free components

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [N, tauD1, tauD2, F, alpha, T, tautrip, SF, offset, A, B]
        E.g. if only N and tauD1 are fitted, this becomes a two
        element vector [1, 1e-3].
        parameters  N       number of particles in focal volume [dim.less]
                    tauD1   diffusion time species 1 [ms]
                    tauD2   diffusion time species 2 [ms]
                    F       fraction of species 1 [dim.less]
                    alpha   relative molecular brightness [dim.less]
                    T       fraction in triplet [dim.less]
                    tautrip residence time in triplet state [µs]
                    SF      shape factor [dim.less]
                    offset  [dim.less]
                    A, B    afterpulsing properties
    fixedparam : 1D np.array
        List with values for the fixed parameters:
        order: [N, tauD1, tauD2, F, alpha, T, tautrip, SF, offset, A, B]
        same principle as fitparamStart.
    fit_info : 1D np.array
        np.array boolean vector with always 11 elements
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit N and tau D this becomes [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        order: [N, tauD1, tauD2, F, alpha, T, tautrip, SF, offset, A, B].
    tau : 1D np.array
        Vector with tau values.
    yexp : 1D np.array
        Vector with experimental autocorrelation.
    weights : 1D np.array, optional
        Weights. The default is 1.

    Returns
    -------
    res : 1D np.array
        Residuals.

    """
    
    fitparam = np.float64(np.zeros(11))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    # convert to SI
    N = fitparam[0]
    tauD1 = 1e-3 * fitparam[1] # ms to s
    tauD2 = 1e-3 * fitparam[2] # ms to s
    F = fitparam[3]
    alpha = fitparam[4]
    T = fitparam[5]
    tautrip = 1e-6 * fitparam[6] # s
    SF = fitparam[7]
    offset = fitparam[8]
    A = 1e-6 * fitparam[9]
    B = -fitparam[10]

    # calculate theoretical autocorrelation function
    FCStheo = fcs_2c_analytical(tau, N, tauD1, tauD2, F, alpha, T, tautrip, SF, offset, A, B)
    
    # calculate residuals
    res = yexp - FCStheo
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_dualfocus(fitparamStart, fixedparam, fit_info, global_param, tau, yexp, weights=1):
    """
    Dual focus fcs fit function

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [c, D (um^2/s), w2 for all (nm), SF for all, rhox for all (nm), rhoy for all (nm), vx (um/s), vy (um/s), offset for all]
        E.g. if only c and D are fitted, this becomes a two
        element vector [1, 1e-3].
    fixedparam : 1D np.array
        List with values for the fixed parameters:
        order: [N, tauD, SF, offset]
        same principle as fitparamStart.
    fit_info : 1D np.array
        np.array boolean vector with always 4 elements
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit N and tau D this becomes [1, 1, 0, 0]
        order: [N, tauD, SF, offset].
    tau : 1D np.array
        Vector with tau values
        All curves are concatenated.
    yexp : 1D np.array
        Vector with experimental autocorrelation.
    useSingleW : 1D np.array, optional
        Use the same w0 value for all cross-correlation curves
        If True, "fitparamStart", "fixedparam" and "fit_info"
        remain as before, bu only the first w0 value is used.
        The default is False.
    weights : 1D np.array, optional
        Vector with weights. The default is 1.

    Returns
    -------
    res : 1D np.array
        Fit residuals.

    """
    
    # number of parameters (fitted and fixed combined)
    n_param = len(fit_info)
    
    # number of histogram curves to fit and number of bins in each histogram
    n_corr, _ = hist_param(yexp)
    
    # if global_param is None, assume none of the parameters is globally fitted
    if global_param is None:
        global_param = [False for i in range(n_param)]
    
    all_param = make_2D_fit_parameter_array_global_fit(fitparamStart, fixedparam, fit_info, global_param, n_param, n_corr)
    
    if global_param[0]:
        # concentration is global fit parameter
        # --> convert N to concentration for first curve
        # --> then convert concentration to N for all other curves
        # These N's may be different if the psf is different
        N0 = all_param[0, 0]
        w0 = all_param[2, 0]
        SF0 = all_param[3, 0]
        
        # get particle concentration
        c = N0 / np.pi**(3/2) / w0**3 / SF0
        
        # calculate N for all curves
        N = np.asarray([np.pi**(3/2) * c * all_param[2, i]**3 * all_param[3, i] for i in range(n_corr)])
        all_param[0,:] = N
    
    
    # get diffusion coefficient
    #D = w[0]**2 / 4 / tauD0
    
    yModel = np.concatenate([fcs_dualfocus(tau,
                                           all_param[0, i], # N
                                           1e-12*all_param[1, i], # D
                                           1e-9*all_param[2, i], # w0 (m)
                                           all_param[3, i], # SP
                                           1e-9*all_param[4, i], # rho x (m)
                                           1e-9*all_param[5, i], # rho y (m)
                                           all_param[8, i], # offset
                                           1e-6*all_param[6, i], # vx (m/s)
                                           1e-6*all_param[7, i] # vy (m/s)
                                           ) for i in range(n_corr)])

    if n_corr > 1:
        yexp = np.concatenate([yexp[:,i] for i in range(n_corr)])
        if not np.isscalar(weights):
            weights = np.concatenate([weights[:,i] for i in range(n_corr)])
    
    res = yexp - yModel
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_circfcs(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
    """
    Dual focus fcs fit function

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [N, tauD, w, SF, Rcirc, Tcirc, offset]
        E.g. if only N and tauD are fitted, this becomes a two
        element vector [1, 1e-3].
    fixedparam : 1D np.array
        List with values for the fixed parameters
        same principle as fitparamStart.
    fit_info : 1D np.array
        np.array boolean vector with always 7 elements
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit N and tau D this becomes [1, 1, 0, 0, 0, 0, 0].
    tau : 1D np.array
        Vector with tau values.
    yexp : 1D np.array
        Vector with experimental autocorrelation.
    weights : 1D np.array, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    res : 1D np.array
        Fit residuals.

    """
    
    fitparam = np.float64(np.zeros(7))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    N = fitparam[0]
    tauD0 = 1e-3 * fitparam[1] # ms -> s
    w = fitparam[2]
    SF = fitparam[3]
    Rcirc = fitparam[4]
    Tcirc = fitparam[5]
    alpha = 2 * np.pi / Tcirc * tau
    rho = Rcirc * np.sqrt(2-2*np.cos(alpha)) # = 2 * Rcirc * abs(sin(alpha/2))
    dc = fitparam[6]
    
    # get diffusion coefficient
    D = w**2 / 4 / tauD0

    yModel = fcs_dualfocus(tau, N, D, w, SF, rho, 0, dc, 0, 0)

    res = yexp - yModel
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_finitelength(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
    """
    Fit function finite length, 3D free diffusion
    Based on Kohler et al., Biophys. J., 2023

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [N, Tau (ms), Shape parameter, T (s), Tsampling (ms)]
    fixedparam : 1D np.array
        List with values for the fixed parameters:
        same order as fitparamStart
    fit_info : 1D np.array
        np.array boolean vector with always 6 elements
        1 for a fitted parameter, 0 for a fixed parameter.
    tau : 1D np.array
        Vector with tau values.
    yexp : 1D np.array
        Vector with experimental autocorrelation.
    weights : 1D np.array or 1, optional
        Vector with weights. The default is 1.

    Returns
    -------
    res : 1D np.array
        Residuals.

    """
    fitparam = np.float64(np.zeros(6))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    
    #
    N = fitparam[0]
    tauD = 1e-3 * fitparam[1] # ms -> s
    SF = fitparam[2]
    T = fitparam[3] # segment length
    Tsampling = 1e-3 * fitparam[4] # sampling time (ms -> s)
    brightness = fitparam[5]
    
    FCStheo = fcs_analytical(tau, N, tauD, SF, 0, 0, 0, 1)
    gamma = compute_gamma(tau, N, tauD, T, Tsampling, gamma_factor=1, SP=SF, brightness=brightness)
    yModel = FCStheo + gamma
    
    res = yexp - yModel
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_free_diffusion_2d(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
    """
    fcs fit function two components 2D

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters:
        order: [N, tauD1, tauD2, F, alpha, T, tautrip, offset, A, B]
        E.g. if only N and tauD1 are fitted, this becomes a two
        element vector [1, 1e-3].
        parameters  N       number of particles in focal volume [dim.less]
                    tauD1   diffusion time species 1 [ms]
                    tauD2   diffusion time species 2 [ms]
                    F       fraction of species 1 [dim.less]
                    alpha   relative molecular brightness [dim.less]
                    T       fraction in triplet [dim.less]
                    tautrip residence time in triplet state [µs]
                    offset  [dim.less]
                    A, B    Afterpulsing parameters
    fixedparam : 1D np.array
        List with values for the fixed parameters:
        order: [N, tauD1, tauD2, F, alpha, T, tautrip, offset, A, B]
        same principle as fitparamStart.
    fit_info : 1D np.array
        np.array boolean vector with always 11 elements
        1 for a fitted parameter, 0 for a fixed parameter
        E.g. to fit N and tau D this becomes [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        order: [N, tauD1, tauD2, F, alpha, T, tautrip, offset, A, B].
    tau : 1D np.array
        Vector with tau values.
    yexp : 1D np.array
        Vector with experimental autocorrelation.
    weights : 1D np.array, optional
        Weights. The default is 1.

    Returns
    -------
    res : 1D np.array
        Residuals.

    """
    
    fitparam = np.float64(np.zeros(10))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    # convert to SI
    N = fitparam[0]
    tauD1 = 1e-3 * fitparam[1] # ms to s
    tauD2 = 1e-3 * fitparam[2] # ms to s
    F = fitparam[3]
    alpha = fitparam[4]
    T = fitparam[5]
    tautrip = 1e-6 * fitparam[6] # s
    offset = fitparam[7]
    A = fitparam[8]
    B = fitparam[9]
    
    # calculate theoretical autocorrelation function
    FCStheo = fcs_2c_2d_analytical(tau, N, tauD1, tauD2, F, alpha, T, tautrip, offset, A, B)
    
    # calculate residuals
    res = yexp - FCStheo
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_nanosecond_fcs(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
    """
    Fit with nanosecond fcs fit model
    See Galvanetto et al., Nature, 2023 (original formula from elsewhere)

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters
    fixedparam : 1D np.array
        List with values for the fixed parameters
    fit_info : 1D np.array
        np.array boolean vector with always 11 elements
        1 for a fitted parameter, 0 for a fixed parameter
    tau : 1D np.array
        Vector with tau values.
    yexp : 1D np.array
        Vector with experimental autocorrelation.
    weights : 1D np.array, optional
        Weights. The default is 1.

    Returns
    -------
    res : 1D np.array
        Residuals.

    """
    
    fitparam = np.float64(np.zeros(11))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    # convert to SI
    A        = fitparam[0] # amplitude
    c_ab     = fitparam[1] # antibunching amplitude
    tau_ab   = fitparam[2] # antibunching time
    c_conf   = fitparam[3] # conformational dynamics
    tau_conf = fitparam[4]
    c_rot    = fitparam[5] # rotational dynamics
    tau_rot  = fitparam[6]
    c_trip   = fitparam[7] # triplet
    tau_trip = fitparam[8]
    tauD     = fitparam[9] # diffusion
    SP       = fitparam[10] # shape parameter
    
    
    # calculate theoretical autocorrelation function
    FCStheo = nanosecond_fcs_analytical(tau, A, c_ab, tau_ab, c_conf, tau_conf, c_rot, tau_rot, c_trip, tau_trip, tauD, SP)
    
    # calculate residuals
    res = yexp - FCStheo
    
    # calculate weighted residuals
    res *= weights
    
    return res


def fitfun_uncoupled_reaction_diffusion(fitparamStart, fixedparam, fit_info, tau, yexp, weights=1):
    """
    Fit with uncoupled reaction and diffusion model
    Assumes that tauD << 1/k_on
    See Mazza et al., ch 12, Monitoring Dynamic Binding of Chromatin Proteins
    In Vivo by Fluorescence Correlation Spectroscopy
    and Temporal Image Correlation Spectroscopy

    Parameters
    ----------
    fitparamStart : 1D np.array
        List with starting values for the fit parameters
    fixedparam : 1D np.array
        List with values for the fixed parameters
    fit_info : 1D np.array
        np.array boolean vector with always 5 elements
        1 for a fitted parameter, 0 for a fixed parameter
    tau : 1D np.array
        Vector with tau values.
    yexp : 1D np.array
        Vector with experimental autocorrelation.
    weights : 1D np.array, optional
        Weights. The default is 1.

    Returns
    -------
    res : 1D np.array
        Residuals.

    """
    
    fitparam = np.float64(np.zeros(5))
    fitparam[fit_info==1] = fitparamStart
    fitparam[fit_info==0] = fixedparam
    
    # convert to SI
    A        = fitparam[0] # amplitude
    tauD     = fitparam[1] # diffusion (s)
    SP       = fitparam[2] # shape parameter
    f_eq     = fitparam[3] # fraction of free molecules at equilibrium
    k_off    = fitparam[4] # dissociation rate (/s)
    
    
    # calculate theoretical autocorrelation function
    FCStheo = uncoupled_reaction_diffusion(tau, A, tauD, SP, f_eq, k_off)
    
    # calculate residuals
    res = yexp - FCStheo
    
    # calculate weighted residuals
    res *= weights
    
    return res


def compute_gamma(tau, Nav, tau_D, T=10, T_s=10e-6, gamma_factor=0.51, SP=3, brightness=100):
    """
    Function needed for the finite length fit function   

    """
    # code by Lisa Cuneo, adapted to 3D by Eli (formula from Kohler et al, Eq. 26)
    
    k_medio = Nav * brightness * T_s
    
    # for 2D
    # B = lambda t, tau : 2 * (tau)**2 * ( (1+t) * np.log(1+t) - t)
    
    r = SP
    s = np.sqrt(r**2-1)
    
    # B2 (Eq. 26 Muller Biophys J. (86) 2004)
    B = lambda x, tD : 4 * r * tD**2 / s * (r*s - s*np.sqrt(r**2+x) - (1+x) * np.log((r-s)*(s+np.sqrt(r**2+x))/np.sqrt(1+x)))
    
    # Eq. S23
    kappa = lambda t : gamma_factor * brightness**2 * Nav * B(t / tau_D, tau_D)
    
    # Eq. S21
    Gamma_C = - T_s**2/(T-tau)**2 * ( kappa(T) + kappa(abs(T-2*tau)) - 2*kappa(tau) ) / ( 2*(k_medio**2) )
    
    #` Eq. 8 main text
    Gamma_S = - ( T_s * (T - 2 * tau) ) / ( (T - tau)**2 * k_medio )
    Gamma_S = np.where(T - 2*tau > 0, Gamma_S, 0 )
    
    Gamma = Gamma_C + Gamma_S
    
    return Gamma


def plot_fit(tau, yexp, param, fit_info, fitResult, plotInfo="", savefig=0, plotTau=True):  
    # final parameters
    param[fit_info==1] = fitResult.x
    # param now contains the final values for [N, tauD, SF, offset]
    # store in object
    data = PyCorrfit_data()
    if len(param) < 7:
        data.tauD = 1e3 * param[1] # in ms
    else:
        data.tauD = [param[1], param[2]] # in ms, 2 components
    
    G = yexp
    Gres = fitResult.fun
    Gfit = G - Gres
    
    Gdata = np.zeros([len(tau), 4])
    Gdata[:, 0] = tau
    Gdata[:, 1] = G
    Gdata[:, 2] = Gfit
    Gdata[:, 3] = Gres
    
    data.data = Gdata
    
    # plot
    plot_pycorrfit(data, plotInfo, savefig, plotTau)


def make_fit_info_global_fit(G, list_of_g, lbounds, ubounds, fitarray, startvalues, rho_x, rho_y):
    # depricated
    tau = getattr(G, list_of_g[0] + '_averageX')[:,0]
    n_tau = len(tau)
    n_traces = len(list_of_g)
    
    G_all = np.zeros((n_tau, n_traces))
    fit_info_all = np.zeros((4 + 5*n_traces, 4)) # fitinfo, startv, minb, maxb
    
    for f, corr in enumerate(list_of_g):
        G_all[:,f] = getattr(G, corr + '_averageX')[:,1]
        minb = lbounds
        maxb = ubounds
        fitarraytemp = fitarray
        fitstartvtemp = np.concatenate([startvalues[0:4], [rho_x[f]], [rho_y[f]], startvalues[4:7]])
        
        for idxp, fitp in enumerate([fitarraytemp, fitstartvtemp, minb, maxb]):
            fit_info_all[0:2, idxp] = fitp[0:2] # c, D
            fit_info_all[2+f, idxp] = fitp[2] # w for all
            fit_info_all[2+n_traces+f, idxp] = fitp[3] # SF for all
            fit_info_all[2+2*n_traces+f, idxp] = fitp[4] # rhox for all
            fit_info_all[2+3*n_traces+f, idxp] = fitp[5] # rhoy for all
            fit_info_all[2+4*n_traces, idxp] = fitp[6] # vx
            fit_info_all[3+4*n_traces, idxp] = fitp[7] # vy
            fit_info_all[4+4*n_traces+f, idxp] = fitp[8] # dc for all
    return G_all, tau, fit_info_all

def read_global_fit_result(fitresult, list_of_g, fitarray, startvalues, rho_x, rho_y):
    n_traces = len(list_of_g)
    fitOut = np.zeros((9, n_traces))
    for f in range(n_traces):
        fitOut[:,f] = np.concatenate([startvalues[0:4], [rho_x[f]], [rho_y[f]], startvalues[4:7]])
    j = 0
    if fitarray[0]:
        fitOut[0,:] = fitresult.x[j] # c was fitted
        j += 1
    if fitarray[1]:
        fitOut[1,:] = fitresult.x[j] # D was fitted
        j += 1
    for p in range(4):
        if fitarray[p+2]:
            fitOut[p+2,:] = fitresult.x[j:j+n_traces] # w, SF, rhox, rhoy were fitted
            j += n_traces
    if fitarray[6]:
        fitOut[6,:] = fitresult.x[j] # vx was fitted
        j += 1
    if fitarray[7]:
        fitOut[7,:] = fitresult.x[j] # vy was fitted
        j += 1
    if fitarray[8]:
        fitOut[8,:] = fitresult.x[j:j+n_traces] # dc was fitted
    # beam waist should be the same for all curves
    fitOut[2,:] = fitOut[2,0]
    
    return np.concatenate([fitOut[0:4,0], fitOut[6:9,0]])

def g2global_fit_struct(G, listOfFields=['spatialCorr'], start=0, N=5):
    """
    Create all variables needed for fcs fit based on correlation object G

    Parameters
    ----------
    G : object
        Correlation object.
    listOfFields : list, optional
        Type of analysis, e.g. "spatialCorr", "crossCenterAv",...
        The default is ['spatialCorr'].
    start : scalar, optional
        Start index for G and tau. The default is 0.
    N : scalar, optional
        Number of correlations to use
        For "spatialCorr", N=3 means the central 3x3 square of G
        For "crossCenterAv", N=3 means rho=1, rho=sqrt(2), and
            rho=2 are used. The default is 5.

    Returns
    -------
    Gout : 1D np.array
        Vector with all G's appended to one another.
    tau : 1D np.array
        Vector with all tau's appended to one another.
    split : 1D np.array
        Vector with the boundary indices for Gout and tau.
    rhox : 1D np.array
        Vector with rhox values for all Gout values.
    rhoy : 1D np.array
        Vector with rhoy values for all Gout values.
    Gcrop : 2D np.array
        2D matrix with all G's in different columns.

    """
    
    if listOfFields == '':
        listOfFields = list(G.__dict__.keys())
    pxdwelltime = G.dwellTime
    tau = []
    Gout = []
    rhox = []
    rhoy = []
    if 'spatialCorr' in listOfFields:
        Gcrop = G.spatialCorr[:, :, start:]
        Gshape = np.shape(Gcrop)
        GNy = Gshape[0]
        GNx = Gshape[1]
        GNt = Gshape[2]
        Gcolumns = np.zeros((GNt, N**2))
        Centerx = int(np.floor(GNx/2))
        Centery = int(np.floor(GNy/2))
        GxStart = int(-np.floor(N/2))
        for i in range(N):
            for j in range(N):
                Gcolumns[:, i*N+j] = Gcrop[GxStart+i+Centery, GxStart+j+Centerx, :]
                rhox = np.append(rhox, GxStart+j)
                rhoy = np.append(rhoy, GxStart+i)
                tau = np.append(tau, np.linspace(start*pxdwelltime, pxdwelltime*(GNt-1), GNt-start))
                Gout = np.append(Gout, Gcrop[i, j, :])
        Gcrop = Gcolumns
        split = np.linspace(0, GNt*N*N, N*N+1)
    elif 'crossCenterAv' in listOfFields:
        # average all pair-correlations between the center and the other pixels
        # that are equally far from the center
        Gcrop = G.crossCenterAv[start:,:]
        Gshape = np.shape(Gcrop)
        GNt = Gshape[0]
        for i in range(6):
            Gout = np.append(Gout, Gcrop[:, i])
            tau = np.append(tau, np.linspace(start*pxdwelltime, pxdwelltime*(GNt-1), GNt-start))
        rhox = np.array([0, 1, np.sqrt(2), 2, np.sqrt(5), np.sqrt(8)])
        rhoy = np.array([0, 0, 0, 0, 0, 0])
        split = np.linspace(0, GNt*6, 7)
    elif 'crossCenterAvN' in listOfFields:
        # similar to crossCenterAv, but without the central autocorrelation
        # only the N G's closest to the center are calculated
        Gcrop = G.crossCenterAv[start:,1:N+1]
        Gshape = np.shape(Gcrop)
        GNt = Gshape[0]
        for i in range(N):
            Gout = np.append(Gout, Gcrop[:, i])
            tau = np.append(tau, np.linspace(start*pxdwelltime, pxdwelltime*(GNt-1), GNt-start))
        rhox = np.array([1, np.sqrt(2), 2, np.sqrt(5), np.sqrt(8)])
        rhox = rhox[0:N]
        rhoy = np.array([0, 0, 0, 0, 0])
        rhoy = rhoy[0:N]
        split = np.linspace(0, GNt*N, N+1)
    elif 'twofocus5' in listOfFields:
        # average all pair-correlations between the center and the other pixels
        # that are equally far from the center
        Nt = len(G.twofocusTB_average[start:,0])
        Gcrop = np.zeros((Nt, 6))
        # cross-correlation top and bottom
        Gcrop[:,0] = (G.twofocusTB_average[start:,1] + G.twofocusBT_average[start:,1]) / 2
        # cross-correlation left and right
        Gcrop[:,1] = (G.twofocusLR_average[start:,1] + G.twofocusRL_average[start:,1]) / 2
        # cross-correlation left and top
        Gcrop[:,2] = (G.twofocusLT_average[start:,1] + G.twofocusTL_average[start:,1]) / 2
        # cross-correlation bottom and right
        Gcrop[:,3] = (G.twofocusBR_average[start:,1] + G.twofocusRB_average[start:,1]) / 2
        # cross-correlation left and bottom
        Gcrop[:,4] = (G.twofocusLB_average[start:,1] + G.twofocusBL_average[start:,1]) / 2
        # cross-correlation top and right
        Gcrop[:,5] = (G.twofocusTR_average[start:,1] + G.twofocusRT_average[start:,1]) / 2
        for i in range(6):
            Gout = np.append(Gout, Gcrop[:, i])
            tau = np.append(tau, np.linspace(start*pxdwelltime, pxdwelltime*(Nt-1), Nt-start))
        rhox = np.array([2, 2, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)])
        rhoy = np.array([0, 0, 0, 0, 0, 0])
        split = np.linspace(0, Nt*6, 7)
    elif 'twofocus5Av' in listOfFields:
        # average all pair-correlations between the center and the other pixels
        # that are equally far from the center
        Nt = len(G.twofocusTB_average[start:,0])
        Gcrop = np.zeros((Nt, 2))
        # cross-correlation long distance
        Gcrop[:,0] = G.twofocusTB_average[start:,1] + G.twofocusBT_average[start:,1]
        Gcrop[:,0] += G.twofocusLR_average[start:,1] + G.twofocusRL_average[start:,1]
        Gcrop[:,0] /= 4
        # cross-correlation short distance
        Gcrop[:,1] = G.twofocusLT_average[start:,1] + G.twofocusTL_average[start:,1]
        Gcrop[:,1] += G.twofocusBR_average[start:,1] + G.twofocusRB_average[start:,1]
        Gcrop[:,1] += G.twofocusLB_average[start:,1] + G.twofocusBL_average[start:,1]
        Gcrop[:,1] += G.twofocusTR_average[start:,1] + G.twofocusRT_average[start:,1]
        Gcrop[:,1] /= 8
        for i in range(2):
            Gout = np.append(Gout, Gcrop[:, i])
            tau = np.append(tau, np.linspace(start*pxdwelltime, pxdwelltime*(Nt-1), Nt-start))
        rhox = np.array([2, np.sqrt(2)])
        rhoy = np.array([0, 0])
        split = np.linspace(0, Nt*2, 3)
    return Gout, tau, split, rhox, rhoy, Gcrop


def chi_2_fcs_fit(tau, Gexp, fit_info, fitResult):
    # calculate the chi2 value for the fcs fit
    Gres = fitResult.fun
    Gfit = Gexp - Gres
    dof = len(tau) - np.sum(fit_info) - 1 # degrees of freedom
    chi2 = np.sum((Gexp-Gfit)**2 / np.abs(Gfit)) / dof
    return chi2


def stddev_2_weights(std, clipmin=None, clipmax=None):
    # convert standard deviation to weights = 1/sigma^2
    # regularize
    std[std==0] = np.min(std[std!=0])
    std /= np.min(std)
    # weights
    weights = 1/std**2
    # normalize
    weights /= np.max(weights)
    weights = np.clip(weights, clipmin, clipmax)
    return weights
    