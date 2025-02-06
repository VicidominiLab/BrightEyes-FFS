import numpy as np
from scipy.optimize import least_squares

def fit_polynomial2d(z, ox, oy, weights=1):
    """
    Fit 2D polynomial function

    Parameters
    ----------
    z : TYPE
        Matrix with function values.
    ox : int
        polynomial order in x.
    oy : int
        polynomial order in y.
    weights : int or np.array, optional
        Matrix with fit weights
        1 for unweighted fit. The default is 1.

    Returns
    -------
    fitresult : TYPE
        Result of the nonlinear least squares fit.

    """
    
    Ncoeff = ox * oy + 2
    
    fitparamStart = np.zeros((Ncoeff))
    
    lowerBounds = fitparamStart - 1e6
    upperBounds = fitparamStart + 1e6
    
    fitparamStart += 1
    Ny = np.shape(z)[0]
    Nx = np.shape(z)[1]
    fitparamStart[0] = Ny / 2
    fitparamStart[1] = Nx / 2
    
    fitresult = least_squares(fitfun_polynomial2d, fitparamStart, args=(ox, oy, z, weights), bounds=(lowerBounds, upperBounds))
    
    return fitresult
    


def fitfun_polynomial2d(fitparamStart, ox, oy, z, weights=1):
    """
    Calculate residuals 2D polynomial function

    Parameters
    ----------
    fitparamStart : np.array()
        List with starting values for the fit parameters:
        order: [x0, y0, A, sigma, offset]
        with    x0, y0      position of the Gaussian center
                A           amplitude of the Gaussian
                sigma       Standard deviation of the Gaussian
                            w0 = 2 * sigma
                            with w0 1/e^2 radius
                offset      dc offset.
    ox : int
        polynomial order in x.
    oy : int
        polynomial order in y.
    z : 2D np.array()
        Matrix with function values.
    weights : int or np.array, optional
        Matrix with fit weights
        1 for unweighted fit. The default is 1.

    Returns
    -------
    res : vector
        Vector with residuals: f(param) - y.

    """
    
    Ny = np.shape(z)[0]
    Nx = np.shape(z)[1]
    
    res = np.reshape(weights * (polynomial2d(z, ox, oy, fitparamStart) - z), [Nx*Ny, 1])
    res = np.ravel(res)

    return res

def polynomial2d(z, ox, oy, coeffs):
    """
    Calculate 2D polynomial function

    Parameters
    ----------
    z : 2D np.array()
        Matrix with function values.
    ox : int
        polynomial order in x.
    oy : int
        polynomial order in y.
    coeffs : list
        list of coefficients.

    Returns
    -------
    out : np.array()
        matrix with 2D polynomial.

    """
    
    
    Ny = np.shape(z)[0]
    Nx = np.shape(z)[1]
    
    meshgrid = np.meshgrid(np.linspace(0, Nx-1, Nx), np.linspace(0, Ny-1, Ny))
    x = meshgrid[0]
    y = meshgrid[1]
    
    x0 = coeffs[0]
    y0 = coeffs[1]
    c = coeffs[2:].reshape((ox, oy))
    
    out = np.polynomial.polynomial.polyval2d(x-x0, y-y0, c)
    
    return out

