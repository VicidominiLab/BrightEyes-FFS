import numpy as np
from scipy.optimize import least_squares


def fit_gauss_2d(y, fitInfo, param, weights=1):
    """
    Fit 2D Gaussian function for STICS analysis

    Parameters
    ----------
    y : np.array
        Matrix with function values.
    fitInfo : np.array
        np.array boolean vector with always 5 elements
        1 for a fitted parameter, 0 for a fixed parameter
        order: [x0, y0, A, sigma, offset]
        E.g. to fit sigma and offset this becomes [0, 0, 0, 1, 1].
    param : np.array
        List with starting values for all parameters:
        order: [x0, y0, A, sigma, offset].
    weights : np.array, optional
        Matrix with weight values 1 for unweighted fit. The default is 1.

    Returns
    -------
    fitresult : np.array
        Result of the nonlinear least squares fit.

    """
    
    fitInfo = np.array(fitInfo)
    param = np.array(param)
    lowerBounds = np.array([0, 0, 0, 1e-10, -100])
    upperBounds = np.array([1e6, 1e6, 1e6, 1e6, 100])
    
    fitparamStart = param[fitInfo==1]
    fixedparam = param[fitInfo==0]
    lowerBounds = lowerBounds[fitInfo==1]
    upperBounds = upperBounds[fitInfo==1]
    
    fitresult = least_squares(fitfun_gauss2d, fitparamStart, args=(fixedparam, fitInfo, y, weights), bounds=(lowerBounds, upperBounds))
    
    return fitresult
    

def fitfun_gauss2d(fitparamStart, fixedparam, fitInfo, y, weights=1):
    """
    Calculate residuals 2D Gaussian function for STICS analysis

    Parameters
    ----------
    fitparamStart : np.array
        List with starting values for the fit parameters:
        order: [x0, y0, A, sigma, offset]
        with    x0, y0      position of the Gaussian center
                A           amplitude of the Gaussian
                sigma       Standard deviation of the Gaussian
                            w0 = 2 * sigma
                            with w0 1/e^2 radius
                offset      dc offset.
    fixedparam : np.array
        List with values for the fixed parameters.
    fitInfo : np.array
        Boolean list of always 5 elements
        1   fit this parameters
        0   keep this parameter fixed.
    y : np.array
        Matrix with experimental function values.
    weights : np.array, optional
        Matrix with fit weights 1 for unweighted fit. The default is 1.

    Returns
    -------
    res : np.array
        Vector with residuals: f(param) - y.

    """
    
    param = np.float64(np.zeros(5))
    param[fitInfo==1] = fitparamStart
    param[fitInfo==0] = fixedparam
    
    x0 = param[0]
    y0 = param[1]
    A = param[2]
    sigma = param[3]
    offset = param[4]
    Ny = np.shape(y)[0]
    Nx = np.shape(y)[1]
    
    res = np.reshape(weights * (normal2d(Nx, Ny, x0, y0, A, sigma, offset) - y), [Nx*Ny, 1])
    res = np.ravel(res)

    return res


def normal2d(Nx, Ny, x0, y0, A, sigma, offset):
    """
    Calculate normal 2D distribution

    Parameters
    ----------
    Nx : int
        Number of columns of the data matrix.
    Ny : int
        Number of rows of the data matrix.
    x0 : int
        Column and row number of the center of the Gaussian function.
    y0 : int
        Column and row number of the center of the Gaussian function.
    A : float
        Amplitude.
    sigma : float
        Standard deviation of the Gaussian function.
    offset : float
        DC component.

    Returns
    -------
    out : 2D np.array()
        Matrix with normal distribution
        y = exp(- ((x-x0)^2 + (y-y0)^2) / (2*sigma^2)).

    """
    
    meshgrid = np.meshgrid(np.linspace(0, Nx-1, Nx), np.linspace(0, Ny-1, Ny))
    x = meshgrid[0]
    y = meshgrid[1]
    out = A * np.exp(-((x-x0)**2 + (y-y0)**2) / 2 / sigma**2) + offset
    return out