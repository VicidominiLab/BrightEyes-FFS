import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


"""
Fit curves to different fit functions
"""

def fit_power_law_g(G, start=0, plotIndv=True, fitInfo=np.array([1, 1, 0]), param=np.array([3.1, -1, 0])):
    # remove dwellTime field from G list
    Gkeys = list(G.__dict__.keys())
    if "dwellTime" in Gkeys:
        Gkeys.remove("dwellTime")
    
    # number of curves
    Nc = len(Gkeys)
        
    # create empty array to store fitreseults
    fitresults = np.zeros((Nc, np.sum(fitInfo)))
    
    # check how to plot
    if Nc == 25 and plotIndv == False:
        plt.rcParams.update({'font.size': 6})
        fig, axs = plt.subplots(5, 5, figsize = (9, 6))
        fig.subplots_adjust(hspace = 1, wspace=.1)
        axs = axs.ravel()
    
    for i in range(len(Gkeys)):
        GName = Gkeys[i]
        Gtemp = getattr(G, GName)
        y = Gtemp[start:,1]
        x = Gtemp[start:,0]
        fitresult = fit_curve(y, x, 'powerlaw', fitInfo, param, np.array([0, -10, -10]), np.array([100, 0, 100]), 0)
        if len(Gkeys) > 10 and 'det' in GName:
            detNr = int(GName[3:-8])
        else:
            detNr = i
        fitresults[detNr, :] = np.array(fitresult.x)
        if Nc == 25 and plotIndv == False:
            axs[detNr].plot(x, y-fitresult.fun, c='k')
            axs[detNr].scatter(x, y, s=3)
            axs[detNr].set_xscale('log')
            axs[detNr].set_title(GName)
        elif plotIndv:
            plt.rcParams.update({'font.size': 20})
            plt.figure()
            plt.scatter(x, y, s=1)
            plt.plot(x, y-fitresult.fun)
            plt.xscale('log')
            plt.title(GName)
    return fitresults


def fit_curve(y, x, fitfun, fitInfo, param, lBounds, uBounds, savefig=0):
    """
    Fit 1D curve with a power law y = A * B^x + C or other function

    Parameters
    ----------
    y : np.array()
        Vector with y values.
    x : np.array()
        Vector with x values.
    fitfun : str
        Fit function
        'powerlaw'           A * B^x + C
        'exp'                A * exp(-alpha * x) + B
        'linear'             A * x + B.
    fitInfo : np.array()
        np.array boolean vector with [A, B, C]
        1 if this value has to be fitted
        0 if this value is fixed during the fit.
    param : np.array()
        np.array vector with start values for [A, B, C]
        A ~ 1e6*2.5e-8
        B ~ -1.05.
    lBounds : np.array()
        Vector with lower bounds for the parameters.
    uBounds : np.array()
        Vector with upper bounds for the parameters.
    savefig : boolean or str, optional
        0 to not save the figure
        file name with extension to save as "png" or "eps".
        The default is 0.

    Returns
    -------
    fitresult : object
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
    if fitfun == 'powerlaw':
        fitresult = least_squares(fitfun_powerlaw, fitparamStart, args=(fixedparam, fitInfo, x, y), bounds=(lowerBounds, upperBounds))
    elif fitfun == 'exp':
        fitresult = least_squares(fitfun_exp_n, fitparamStart, args=(fixedparam, fitInfo, x, y), bounds=(lowerBounds, upperBounds))
    elif fitfun == 'linear':
        fitresult = least_squares(fitfun_linear, fitparamStart, args=(fixedparam, fitInfo, x, y), bounds=(lowerBounds, upperBounds))
        
    #plot_fit(x, y, param, fitInfo, fitresult, savefig)

    return fitresult


def fitfun_powerlaw(fitparamStart, fixedparam, fitInfo, x, y):
    """
    Power law fit function
    y = A * B^x + C

    Parameters
    ----------
    fitparamStart : np.array()
        List with starting values for the fit parameters:
        order: [A, B, C]
        E.g. if only A and B are fitted, this becomes a two
        element vector [1e-4, 0.2].
    fixedparam : np.array()
        List with values for the fixed parameters:
        order: [A, B, C]
        same principle as fitparamStart.
    fitInfo : np.array()
        np.array boolean vector with always 3 elements
        1 for a fitted parameter, 0 for a fixed parameter.
    x : np.array()
        Vector with x values.
    y : np.array()
        Vector with experimental y values.

    Returns
    -------
    res : np.array()
        Residuals.

    """
    
    fitparam = np.float64(np.zeros(len(fitInfo)))
    fitparam[fitInfo==1] = fitparamStart
    fitparam[fitInfo==0] = fixedparam
    
    # get parameters
    A = fitparam[0]
    B = fitparam[1]
    C = fitparam[2]

    # calculate theoretical power law function
    ytheo = A * x**B + C
    
    # calcualte residuals
    res = y - ytheo
    
    return res


def fitfun_exp(fitparamStart, fixedparam, fitInfo, x, y):
    """
    Exponential fit function
    y = A * exp(-alpha * (x-dx)) + B
    (alpha = 1 / tau with tau the lifetime)

    Parameters
    ----------
    fitparamStart : np.array()
        List with starting values for the fit parameters:
        order: [A, alpha, dx, B]
        E.g. if only A and B are fitted, this becomes a two
        element vector [1e-4, 0.2].
    fixedparam : np.array()
        List with values for the fixed parameters:
        order: [A, alpha, B]
        same principle as fitparamStart.
    fitInfo : np.array()
        np.array boolean vector with always 3 elements
        1 for a fitted parameter, 0 for a fixed parameter.
    x : np.array()
        Vector with x values.
    y : np.array()
        Vector with experimental y values.

    Returns
    -------
    res : np.array()
        Residuals.

    """
    
    fitparam = np.float64(np.zeros(len(fitInfo)))
    fitparam[fitInfo==1] = fitparamStart
    fitparam[fitInfo==0] = fixedparam
    
    # get parameters
    A = fitparam[0]
    alpha = fitparam[1]
    dx = fitparam[2]
    B = fitparam[3]
    
    # calculate theoretical curve
    ytheo = A * np.exp(-alpha * (x-dx)) + B
    
    # calculate residuals
    res = y - ytheo
    
    return res


def fitfun_exp_n(fitparamStart, fixedparam, fitInfo, x, y):
    """
    Sum of exponentials fit function
    y = sum(A_i * exp(-alpha_i * (x-dx))) + offset
    Same dx assumed for each exponential
    (alpha = 1 / tau with tau the lifetime)

    Parameters
    ----------
    fitparamStart : np.array
        List with starting values for the fit parameters:
        order: [A1, alpha1, A2, alpha2, ...,dx, offset]
        E.g. if only A and B are fitted, this becomes a two
        element vector [1e-4, 0.2].
    fixedparam : np.array
        List with values for the fixed parameters:
        order: [A, alpha, B, ..., dx]
        same principle as fitparamStart.
    fitInfo : np.array
        np.array boolean vector with always 2*n+2 elements with n the number of
        components
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
    ytheo = fun_exp_n(x, fitparam)
    
    # calculate residuals
    res = y - ytheo
    
    return res


def fun_exp_n(x, param):
    """
    Calculate theoretical exponential function, multiple components allowed

    Parameters
    ----------
    x : np.array()
        1D array with x values
    param : list
        [A1, alpha1, A2, alpha2, ...,dx, offset].

    Returns
    -------
    ytheo : np.array()
        1D array with y values.

    """
    # number of components
    n_comp = int((len(param) - 2) / 2)
    
    # global values
    dx = param[-2]
    offset = param[-1]
    ytheo = 0
    for i in range(n_comp):
        A = param[2*i]
        alpha = param[2*i+1]
        ytheo += A * np.exp(-alpha * (x-dx))
    ytheo += offset
    return ytheo


def fitfun_linear(fitparamStart, fixedparam, fitInfo, x, y):
    """
    Linear fit function
    y = A * x + B

    Parameters
    ----------
    fitparamStart : np.array()
        List with starting values for the fit parameters:
        order: [A, B]
        E.g. if only A is fitted, this becomes a one
        element vector [1e-4].
    fixedparam : np.array()
        List with values for the fixed parameters:
        order: [A, B]
        same principle as fitparamStart.
    fitInfo : np.array()
        np.array boolean vector with always 2 elements
        1 for a fitted parameter, 0 for a fixed parameter.
    x : np.array()
        Vector with x values.
    y : np.array()
        DESCRIPTION.

    Returns
    -------
    res : np.array()
        Vector with experimental y values.

    """
    
    fitparam = np.float64(np.zeros(len(fitInfo)))
    fitparam[fitInfo==1] = fitparamStart
    fitparam[fitInfo==0] = fixedparam
    
    # get parameters
    A = fitparam[0]
    B = fitparam[1]

    # calculate theoretical curve
    ytheo = A * x + B
    
    # calculate residuals
    res = y - ytheo
    
    return res
