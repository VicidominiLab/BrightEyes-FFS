import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def fit_and_plot(y, x, fitfun, fitInfo, param, lBounds=-1, uBounds=-1, printResults=1, plot_fit=1):
    """
    Fit 1D curve with any of the implemented fit functions
    See fit_data for the meaning of the parameters
    Results are plotted
    """
    
    # number of total parameters
    Nparam = len(fitInfo)
    
    # check bounds
    if lBounds == -1:
        lBounds = np.zeros(Nparam)
    if uBounds == -1:
        uBounds = np.zeros(Nparam) + 1e4
    
    # perform fit
    fitresult = fit_data(y, x, fitfun, fitInfo, param, lBounds, uBounds)    
    
    # print results
    if printResults:
        fittedParam = fitresult.x
        for i in range(len(fittedParam)):
            print(str(fittedParam[i]))
    
    # plot results
    if plot_fit:
        plt.rcParams.update({'font.size': 20})
        plt.figure()
        plt.scatter(x, y, s=1)
        plt.plot(x, y-fitresult.fun)
    
    return fitresult


def fit_data(y, x, fitfun, fitInfo, param, lBounds, uBounds):
    """
    Fit 1D curve with any of the implemented fit functions

    Parameters
    ----------
    y : np.array()
        Vector with y values.
    x : np.array()
        Vector with x values.
    fitfun : function
        Fit function
        'exp'                 A1 * exp(-alpha1 * x) +
                            + A2 * exp(-alpha2 * x) +
                            + ...
                            + An * exp(-alphan * x) +
                            + B.
    fitInfo : np.array()
        np.array boolean vector with [A1, alpha1,..., alphan, C]
        1 if this value has to be fitted
        0 if this value is fixed during the fit.
    param : np.array
        np.array vector with start values for [A, B, C]
        A ~ 1e6*2.5e-8
        B ~ -1.05.
    lBounds : np.array
        Vector with lower bounds for all parameters.
    uBounds : np.array
        Vector with upper bounds for all parameters.

    Returns
    -------
    fitresult : fitresult
        Output of least_squares
        fitresult.x = fit results.

    """
    
    # make sure that all variables are np.arrays
    fitInfo = np.asarray(fitInfo)
    param = np.asarray(param)
    lBounds = np.asarray(lBounds)
    uBounds = np.asarray(uBounds)

    # parse fit and fixed parameters
    fitparamStart = param[fitInfo==1]
    fixedparam = param[fitInfo==0]
    lowerBounds = lBounds[fitInfo==1]
    upperBounds = uBounds[fitInfo==1]    

    # perform fit
    if fitfun == 'exp':
        fitresult = least_squares(fitfun_exp_n, fitparamStart, args=(fixedparam, fitInfo, x, y), bounds=(lowerBounds, upperBounds))

    return fitresult


def fitfun_exp_n(fitparamStart, fixedparam, fitInfo, x, y):
    """
    Exponential fit function (sum of multiple exponentials is possible)
    y = A * exp(-alpha * x) + B
    (alpha = 1 / tau with tau the lifetime)

    Parameters
    ----------
    fitparamStart : np.array
        List with starting values for the fit parameters:
        order: [A, alpha, B]
        E.g. if only A and B are fitted, this becomes a two
        element vector [1e-4, 0.2].
    fixedparam : np.array
        List with values for the fixed parameters:
        order: [A, alpha, B]
        same principle as fitparamStart.
    fitInfo : np.array
        np.array boolean vector with always 3 elements
        1 for a fitted parameter, 0 for a fixed parameter.
    x : np.array
        Vector with x values.
    y : np.array
        Vector with experimental y values.

    Returns
    -------
    res : np.array
        Residuals.

    """
    
    fitparam = np.float64(np.zeros(len(fitInfo)))
    fitparam[fitInfo==1] = fitparamStart
    fitparam[fitInfo==0] = fixedparam
    
    # calculate theoretical exponential function
    ytheo = 0
    for i in range(int((len(fitparam) - 1) / 2)):
        A = fitparam[2*i]
        alpha = fitparam[2*i+1]
        ytheo += A * np.exp(-alpha * x)
    B = fitparam[-1]
    ytheo += B
    
    # calculate residuals
    res = y - ytheo
    
    return res
