import numpy as np

def calc_bic(yfit, y, sigmay, n_params):
    """
    Calculate the Bayesian Information Criterion (BIC) for a model with
    heteroscedastic Gaussian errors.

    Parameters
    ----------
    yfit : np.ndarray
        Model predictions.
    y : np.ndarray
        Observed data.
    sigmay : float or np.ndarray
        Standard deviation of the noise (can be scalar or per-point).
    n_params : int
        Number of free parameters in the model.

    Returns
    -------
    float
        The BIC value.
    """
    y = np.asarray(y)
    yfit = np.asarray(yfit)
    sigmay = np.asarray(sigmay)
    
    # Log-likelihood for Gaussian errors
    ln_L = -0.5 * np.sum(np.log(2 * np.pi * sigmay**2) + ((y - yfit)**2) / sigmay**2)
    
    # BIC
    n = len(y)
    bic = n_params * np.log(n) - 2 * ln_L
    return bic
