import numpy as np
from .fcs2corr import Correlations


def two_focus_fcs(tau, rhox, rhoy, rhoz, c, D, w0, w1, z0, z1):
    """
    Calculate the fcs cross correlation between two PSFs

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s].
    rhox : scalar
        Spatial shift in the x direction between two detectors [m].
        Corresponds to the spatial shifts in the sample space
    rhoy : scalar
        Spatial shift in the y direction between two detectors [m].
        Corresponds to the spatial shifts in the sample space.
    rhoz : scalar
        Spatial shift in the x direction between two detectors [m].
        Corresponds to the spatial shifts in the sample space. Usually 0.
    c : scalar
        Concentration of fluorophores/particles [/m^3].
    D : scalar
        Diffusion coefficient of the fluorophores/particles [µm^2/s].
    w0 : scalar
        Lateral 1/e^2 radius of the first PSF.
    w1 : scalar
        Lateral 1/e^2 radius of the second PSF.
    z0 : scalar
        Axial 1/e^2 value of the first PSF.
    z1 : scalar
        Axial 1/e^2 value of the second PSF.

    Returns
    -------
    G : correlations object
        Object with all autocorrelations.
    Garray : 2D numpy array
        Array with all autocorrelations..

    """

    factorW = 8 * D * tau + w0**2 + w1**2
    factorZ = 8 * D * tau + z0**2 + z1**2
    
    G0 = 2 * np.sqrt(2) / np.pi**(3/2) / c / factorW / np.sqrt(factorZ)
    
    expTerm = 16 * D * tau * (rhox**2 + rhoy**2)
    expTerm = expTerm + 2 * (z0**2 + z1**2) * (rhox**2 + rhoy**2)
    expTerm = expTerm + 2 * (w0**2 + w1**2) * rhoz**2
    expTerm = -1 * expTerm / factorW / factorZ
    expTerm = np.exp(expTerm)
    
    Gy = G0 * expTerm
    
    if type(Gy) == np.float64:
        Garray = np.zeros((1, 2))
    else:
        Garray = np.zeros((np.size(Gy, 0), 2))
    Garray[:, 0] = tau
    Garray[:, 1] = Gy

    G = Correlations()
    setattr(G, 'theory', Garray)

    return G, Garray


def simulate_two_focus_cross_center(tau, c, D, w, z, shift):
    G = Correlations()
    N = np.size(w, 0)
    for i in range(N):
        rhoy = np.floor(i / 5)
        rhox = np.mod(i, 5)
        [xx, Gtemp] = two_focus_fcs(tau, (rhox-2) * shift, (rhoy-2) * shift, 0, c, D, w[12], w[i], z[12], z[i])
        setattr(G, 'det12x' + str(i), Gtemp)
    return G


def simulate_spatialcorr(tau, rho, c, D, w0, z0):
    w0 = np.resize(w0, (5, 5))
    z0 = np.resize(z0, (5, 5))
    G = Correlations()
    if type(tau) == float:
        tau = [tau]
    Nt = len(tau)
    Gall = np.zeros((9, 9, Nt))
    for i in range(Nt):
        Gsinglet = np.zeros((9, 9))
        for shifty in np.arange(-4, 5):
            print('shifty: ' + str(shifty))
            for shiftx in np.arange(-4, 5):
                print(' shiftx: ' + str(shiftx))
                # go through all detector elements
                n = 0  # number of overlapping detector elements
                Gtemp = 0
                for detx in np.arange(np.max((0, shiftx)), np.min((5, 5+shiftx))):
                    print('  detx: ' + str(detx))
                    for dety in np.arange(np.max((0, shifty)), np.min((5, 5+shifty))):
                        print('   dety: ' + str(dety))
                        dummy, Gout = two_focus_fcs(tau[i], shiftx*rho, shifty*rho, 0, c, D, w0[dety, detx], w0[dety-shifty, detx-shiftx], z0[dety, detx], z0[dety-shifty, detx-shiftx])
                        Gtemp += Gout[0, 1]
                        n += 1
                Gtemp /= n
                Gsinglet[shifty+4, shiftx+4] = Gtemp
        Gall[:,:, i] = Gsinglet
    G.autoSpatial = Gall
    G.dwellTime = tau[0]
    return G

