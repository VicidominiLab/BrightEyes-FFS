import numpy as np
from .fcs_analytical import fcs_analytical
import copy
import multiprocessing
from joblib import Parallel, delayed

class Fitresultclass():
    pass


def mem_fit(fitparam, tau, yexp, fitfun=fcs_analytical, weights=1, startdistr='default', multiprocess=False):
    """
    Fit fcs curve with Maximum Entropy model

    Parameters
    ----------
    fitparam : list of scalars
        [n, Niter, eps2, xmu, anomD, corrVentr, shape_param] with
        n               number of diffusion components
        Niter           number of iterations
        eps2            Iteration stop criteria.
        xmu             iteration increment
        anomD           anomalous diffusion parameter
        corrVentr       weight correlation vs. entropy
        shape_param     height / width of focal volume
    tau : 1D np.array()
        tau values.
    yexp : 1D np.array()
        autocorrelation values.
    fitfun : function, optional
        fit function. The default is fcs_analytical.
    weights : 1D np.array() or scalar 1, optional
        weights for fitting. The default is 1.

    Returns
    -------
    f_fit : 1D np.array()
        fitted parameters.
    tauD : 1D np.array()
        same as input.
    G_fit : 1D np.array()
        fitted G values.

    """
    
    
    # fixedparam = [200, 10000, 5e-6, 2e-4, 1, 20]
    n_tauD      = int(fitparam[0])   # number of diffusion components
    Niter_f     = int(fitparam[1])  # number of iterations
    eps2        = fitparam[2]          # Iteration stop criteria.
    x_mu        = fitparam[3]          # iteration increment
    anomD       = fitparam[4]         # anomalous diffusion parameter
    corrVentr   = fitparam[5]     # weight correlation vs. entropy
    shape_param = fitparam[6]     # height / width of focal volume
    
    mu = copy.deepcopy(yexp)
    normc = mu[0]
    mu /= normc
    M = len(yexp)
    sigma = weights
    sigma_square = sigma**2
    
    n_tau = len(tau)
    taumin = np.log10(np.min(tau))
    taumax = np.log10(np.max(tau))
    tauD = np.logspace(taumin, taumax, n_tauD)
    
    dtauD = np.zeros(n_tauD)
    dtauD[0:n_tauD-1] = np.diff(tauD)
    dtauD[n_tauD-1] = dtauD[n_tauD-2]
    
    C0 = np.zeros((Niter_f))
    S = np.zeros((Niter_f))
    alpha = np.zeros((Niter_f))

    f_fit = startdistr
    if type(startdistr) == str:
        f_fit = np.zeros((n_tauD)) + 1 / n_tauD # start distribution
    
    fin_fit = np.zeros((n_tauD, n_tau))
    
    if multiprocess:
        Processed_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(fitfun)(tau, 1, tauD[i], shape_param, 0, A=0, B=0, alpha=anomD) for i in list(range(n_tauD)))
        for i in range(n_tauD):
            fin_fit[i,:] = Processed_list[i] #fitfun(tau, 1, tauD[i], 3, 0, A=0, B=0, alpha=anomD)
    else:
        for i in range(n_tauD):
            fin_fit[i,:] = fitfun(tau, 1, tauD[i], shape_param, 0, A=0, B=0, alpha=anomD)
    
    G_fit = np.zeros((n_tau))
    
    iter_f = 0
    fitFound = False
    while iter_f < Niter_f and not fitFound:
        exp2 = np.sum(f_fit) # Summation of all fractions.
        f_fit = f_fit / exp2 # Normalizing fractions, so G(0)=1.
        # for i in range(n_tau):
        #     G_fit[i] = np.sum(fin_fit[:,i] * f_fit)
        G_fit = np.einsum('ni,n->i', fin_fit, f_fit)
        G_fit = G_fit / G_fit[0] # Normalize G_fit.
        G_fit = G_fit * np.mean(mu[1:2]) # scaled to average amplitude of first two fcs points.
        
        Diff_C_E = G_fit - mu # Residuals (G(tau)_calculated minus G(tau)_experimental).
        r = Diff_C_E / sigma # Weighted residuals.
        r_square = r**2
   
        C0[iter_f] = np.sum(r_square) / n_tau # Record weighted least-squares
        S[iter_f] = -np.sum(f_fit * np.log(f_fit)) # Entropy of each iteration.
    
        if iter_f > 1:
            if np.abs((C0[iter_f] - C0[iter_f-1]) / C0[iter_f-1]) < eps2:
                # termination at variation of chi square is less than 1e-5.
                fitFound = True
                
        # First order derivative of least-squares
        # DC = np.zeros((n_tauD))
        # for i in range(n_tauD):
        #     DC[i] = np.sum(2 * Diff_C_E * fin_fit[i,:] / sigma_square) / M
        DC = np.sum(2 * Diff_C_E * fin_fit / sigma_square, axis=1) / M
    
        # First order derivative of entropy
        # DS = np.zeros((n_tauD))
        # for i in range(n_tauD):
        #     DS[i] = -1 - np.log10(f_fit[i])
        DS = -1 - np.log10(f_fit)
        
        # Scaling factor for balancing entropy and least squares in determining search direction.
        alpha[iter_f] = np.linalg.norm(DC) / np.linalg.norm(DS) / corrVentr
        # Search direction construct.
        e_mu = f_fit * (alpha[iter_f] * DS - DC)
        e_mu = e_mu / np.linalg.norm(e_mu)
        # Update f_fit's using search direction.
        f_fit=f_fit+ e_mu*x_mu
        for i in range(n_tauD):
            if f_fit[i]<0:
                f_fit[i] = 0.0001
            elif f_fit[i] == 0:
                f_fit[i] = 0.0001
    
        iter_f += 1
    
    return f_fit, tauD, G_fit*normc


def mem_fit_free_diffusion(param, tau, yexp, weights=1):
    if len(param) < 6:
        # add default shape parameter to parameter list in case not given as input
        param.append(3)
    [f_fit, tauD, G_fit] = mem_fit(param, tau, yexp, fitfun=fcs_analytical, weights=weights)
    # maxind = np.r_[True, f_fit[1:] > f_fit[:-1]] & np.r_[f_fit[:-1] > f_fit[1:], True]
    # Npeaks = np.sum(maxind == True)
    # print(Npeaks)
    # if Npeaks > 0:
    #     tauD = np.asarray(maxind.astype(int)) * np.asarray(f_fit) + 1e-7
    #     startdistr = tauD / np.sum(tauD)
    #     print(startdistr)
    #     [f_fit, tauD, G_fit] = mem_fit(param, tau, yexp, fitfun=fcs_analytical, weights=weights, startdistr=startdistr)
    
    fitresult = Fitresultclass()
    fitresult.x = f_fit
    fitresult.fun = yexp - G_fit
    fitresult.fun /= weights
    fitresult.tauD = tauD
    return fitresult

