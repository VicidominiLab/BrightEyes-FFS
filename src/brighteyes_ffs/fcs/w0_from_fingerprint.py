import numpy as np
from scipy.optimize import least_squares
import scipy.signal as scisig


def get_w0_from_fingerprint(y, fit_info, param, N=200, weights=1, lower_bounds=[-1e3, -1e3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3], upper_bounds = [1e3, 1e3, 1e4, 1e4, 1e3, 1e4, 1e4]):
    """
    Extract excitation and emission PSFs from SPAD finger print

    Parameters
    ----------
    y : np.array()
        Matrix with SPAD finger print.
    fit_info : np.array
        np.array boolean vector with always 7 elements
        1 for a fitted parameter, 0 for a fixed parameter
        order: [x0, y0, w0Ex, M, lambdaEmEx, pxsize, pxdist]
        E.g. to fit sigma and offset this becomes [0, 0, 0, 1, 1].
    param : list
        List with starting values for all parameters:
        order: [x0, y0, A, sigma, offset].
    N : int, optional
        Resolution of the calculation. The default is 200.
    weights : np.array or int, optional
        Matrix with weight values, 1 for unweighted fit. The default is 1.
    lower_bounds : np.array, optional
        lower bounds for the fit.
        The default is [-1e3, -1e3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3].
    upper_bounds : np.array, optional
        upper bounds for the fit.
        The default is [1e3, 1e3, 1e4, 1e4, 1e3, 1e4, 1e4].

    Returns
    -------
    fitresult : fitresult
        Result of the nonlinear least squares fit

    """
    
    fit_info = np.asarray(fit_info)
    param = np.asarray(param)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    
    fitparamStart = param[fit_info==1]
    fixedparam = param[fit_info==0]
    lower_bounds = lower_bounds[fit_info==1]
    upper_bounds = upper_bounds[fit_info==1]
    
    fitresult = least_squares(fitfun_w0_from_fingerprint, fitparamStart, args=(fixedparam, fit_info, y, N, weights), bounds=(lower_bounds, upper_bounds))
    
    return fitresult
    

def fitfun_w0_from_fingerprint(fitparamStart, fixedparam, fit_info, y, n_datapoints=200, weights=1):
    """
    Caculate residuals to fit excitation and emission PSFs from SPAD finger print

    Parameters
    ----------
    fitparamStart : list
        List with starting values for the fit parameters:
        order: [x0, y0, w0Ex, M, lambdaEmEx, pxsize, pxdist]
        with    x0, y0      position of the Gaussian center
                            in the image plane [µm]
                w0Ex        1/e^2 radius excitation PSF [µm]
                M           magnification of the system
                            typically 500
                lambdaEmEx  ratio excitation / emission wavelength
                pxsize      SPAD pixel size [µm], typically 50 µm
                pxdist      SPAD distance between pixels [µm],
                            typically 20 µm.
    fixedparam : list
        List with values for the fixed parameters.
    fit_info : list
        Boolean list of always 7 elements
        1   fit this parameters
        0   keep this parameter fixed.
    y : np.array
        Matrix with experimental function values.
    n_datapoints : int, optional
        Resolution of the calculation. The default is 200.
    weights : np.array, optional
        Matrix with fit weights, 1 for unweighted fit. The default is 1.

    Returns
    -------
    res : np.array
        Vector with residuals: f(param) - y.

    """
    
    # extract parameters
    param = np.float64(np.zeros(7))
    param[fit_info==1] = fitparamStart
    param[fit_info==0] = fixedparam
    
    Ny = 5
    Nx = 5
    
    Iairy = calc_fingerprint(param, n_datapoints)
    
    res = np.reshape(weights * (Iairy - y), [Nx*Ny, 1])
    res = np.ravel(res)

    return res


def calc_fingerprint(param, n_datapoints):
    """
    Caculate SPAD finger print

    Parameters
    ----------
    param : list
        List with all parameters:
        order: [x0, y0, w0Ex, M, lambdaEmEx, pxsize, pxdist]
        with    x0, y0      position of the Gaussian center
                            in the image plane [µm]
                w0Ex        1/e^2 radius excitation PSF [µm]
                M           magnification of the system,
                            typically 500
                lambdaEmEx  ratio emission / excitation wavelength
                pxsize      SPAD pixel size [µm], typically 50 µm
                pxdist      SPAD distance between pixels [µm],
                            typically 20 µm.
    n_datapoints : int
        Resolution of the calculation, typically 200.

    Returns
    -------
    Iairy : np.array
        Matrix with SPAD finger print.

    """
    
    x0 = 1e-6 * param[0]
    y0 = 1e-6 * param[1]
    w0Ex = 1e-6 * param[2]
    M = param[3]
    lambdaEmEx = param[4]
    pxsize = 1e-6 * param[5]
    pxdist = 1e-6 * param[6]

    print("Trying w0 = " + str(w0Ex))

    # calculate additional parameters
    w0EffEx = M * w0Ex
    w0EffEm = lambdaEmEx * w0EffEx
    
    # generate grid
    Ny = 5
    Nx = 5
    x = np.linspace(-5 * w0EffEx, 5 * w0EffEx, 2 * n_datapoints + 1)
    y = np.linspace(-5 * w0EffEx, 5 * w0EffEx, 2 * n_datapoints + 1)
    xv, yv = np.meshgrid(x, y)
    dx = x[1] - x[0]
    
    # calculate Gaussian
    Iex = np.exp(-2 * (xv**2 + yv**2) / w0EffEx**2)
    Iem = np.exp(-2 * ((xv - x0)**2 + (yv - y0)**2) / w0EffEm**2)
    
    Itot = scisig.convolve2d(Iex, Iem, mode='same')
    Itot /= np.sum(Itot)
    
    # add squares for SPAD elements
    detCoordinates = np.zeros((Nx * Ny, 4), dtype=int) # store coordinates of all detector elements
    cc = n_datapoints # index center
    for i in range(Nx):
        centerSingleX = cc + (i - np.floor(Nx/2)) * (pxsize + pxdist) / dx
        Lx = int(centerSingleX - pxsize / dx / 2 + 1) # x coordinate left border
        Rx = int(centerSingleX + pxsize / dx / 2 + 1) # x coordinate right border
        for j in range(Ny):
            centerSingleY = cc + (j - np.floor(Ny/2)) * (pxsize + pxdist) / dx
            Ty = int(centerSingleY - pxsize / dx / 2 + 1) # y coordinate top border
            By = int(centerSingleY + pxsize / dx / 2 + 1) # y coordinate bottom border
            detCoordinates[i*Ny + j] = [Lx, Rx, Ty, By]
    
    Iairy = np.zeros((Ny, Nx))
    for i in range(Nx):
        for j in range(Ny):
            [Lx, Rx, Ty, By] = detCoordinates[i*Ny + j]
            Iairy[j, i] = np.sum((Itot[Ty:By, Lx:Rx]))
    
    # normalize
    Iairy /= np.max(Iairy)
    
    return Iairy