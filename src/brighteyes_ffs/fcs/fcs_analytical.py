import numpy as np


def fcs_analytical(tau, N, tauD, SF, offset, A=0, B=0, alpha=1):
    """
    Calculate the analytical fcs autocorrelation function assuming 3D Gaussian
    diffusion without triplet state

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s]
    N : scalar
        Number of particles on average in the focal volume [dimensionsless]
        N = w0^2 * z0 * c * pi^(3/2)
        with c the average particle concentration
    tauD : scalar
        Diffusion coefficient of the fluorophores/particles [µm^2/s]
    SF : scalar
        Shape factor of the PSF.
    offset : scalar
        DESCRIPTION.
    A : scalar, optional
        Afterpulsing characteristics. Power law assumed: G = A * tau^B (with B < 0).
        The default is 0.
    B : scalar, optional
        Afterpulsing characteristics. The default is 0.
    alpha : scalar, optional
        Anomalous diffusion parameter (alpha = 1 for free diffusion). The default is 1.

    Returns
    -------
    Gy : 1D numpy array
        Vector with the autocorrelation G(tau).

    """

    # standard autocorrelation function
    Gy = 1 / N / (1 + (tau/tauD)**alpha) # lateral correlation
    Gy /= np.sqrt(1 + tau**alpha / (SF**2 * tauD**alpha)) # axial correlation
    Gy += offset # offset
    # power law component to take into account afterpulsing (see e.g. Buchholz, Biophys J., 2018)
    Gy += A * tau**B
    
    if type(Gy) == np.float64:
        Garray = np.zeros((1, 2))
    else:
        Garray = np.zeros((np.size(Gy, 0), 2))
    Garray[:, 0] = tau
    Garray[:, 1] = Gy

    return Gy


def fcs_2c_analytical(tau, N, tauD1, tauD2, F, alpha=1, T=0, tautrip=1e-6, SF=5, offset=0, A=0, B=0):
    """
    Calculate the analytical fcs autocorrelation function assuming 3D Gaussian
    diffusion with triplet state, afterpulsing and 2 components

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    N : scalar
        Number of particles on average in the focal volume [dimensionsless]
        N = w0^2 * z0 * c * pi^(3/2).
        with c the average particle concentration
    tauD1 : scalar
        Diffusion time species 1 [s].
    tauD2 : scalar
        Diffusion time species 2 [s].
    F : scalar
        Fraction of species 1.
    alpha : scalar, optional
        Relative molecular brightness q2/q1. The default is 1.
    T : scalar, optional
        Fraction in triplet. The default is 0.
    tautrip : scalar, optional
        Residence time in triplet state [s]. The default is 1e-6.
    SF : scalar, optional
        Shape factor of the PSF. The default is 5.
    offset : scalar, optional
        Offset. The default is 0.
    A : scalar, optional
        Afterpulsing characteristics. The default is 0.
        Power law assumed: G = A * tau^B (with B < 0)
    B : scalar, optional
        Afterpulsing characteristics. The default is 0.

    Returns
    -------
    Gy : 1D numpy array
        Vector with the autocorrelation G(tau).

    """

    # amplitude
    Gy = N * (F + alpha*(1-F))**2
    Gy = 1 / Gy
    
    # triplet
    Gy *= (1 + (T * np.exp(-tau / tautrip)) / (1 - T))
    
    # diffusion
    Gy *= F / (1 + tau/tauD1) / np.sqrt(1 + tau/SF**2/tauD1) + alpha**2 * (1-F) / (1 + tau/tauD2) / np.sqrt(1 + tau/SF**2/tauD2)

    # offset
    Gy += offset

    # afterpulsing (see e.g. Buchholz, Biophys J., 2018)
    Gy += A * tau**B

    return Gy


def fcs_analytical_2c_anomalous(tau, N, tauD1, tauD2, alpha1, alpha2, F, T, tau_triplet, SF, offset, brightness):
    # amplitude
    Gy = 1 / N
    
    # brightness
    Gy /= (F + brightness * (1-F))**2
    
    # triplet fraction
    Gy *= (1 + T / (1 - T) * np.exp(-tau/tau_triplet))
    
    # two anomalous components
    Gcomp1 = F * (1 + (tau/tauD1)**alpha1)**(-1) * (1 + (tau/tauD1)**alpha1/SF**2)**(-1/2)
    Gcomp2 = brightness**2 * (1 - F) * (1 + (tau/tauD2)**alpha2)**(-1) * (1 + (tau/tauD2)**alpha2/SF**2)**(-1/2)
    
    # total
    Gy *= (Gcomp1 + Gcomp2)
    Gy += offset
    
    return Gy


def fcs_dualfocus(tau, N, D, w, SF, rhox, rhoy, offset, vx=0, vy=0):
    """
    Calculate the analytical fcs crosscorrelation function for dual focus fcs
    assuming 3D Gaussian and diffusion without triplet state
    Equation from Scipioni, Nat. Comm., 2018 and consistent with own Maple
    calculations

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    N : scalar
        Number of particles on average in the focal volume [dimensionsless]
        N = w0^2 * z0 * c * pi^(3/2)
        with c the average particle concentration
        and w0 the effective focal volume (w0^2 + w1^2) / 2.
    D : scalar
        Diffusion coefficient of the fluorophores/particles [m^2/s].
    w : scalar
        Radius of the effective PSF, i.e. sqrt((w0^2 + w1^2) / 2)
        with w0 and w1 the 1/e^2 radii of the two PSFs. [m]
    SF : scalar
        Shape factor of the PSF.
    rhox : scalar
        Distance between the two detector elements in the horizontal direction [m].
    rhoy : scalar
        Distance between the two detector elements in the vertical direction [m].
    offset : scalar
        DC component of G.
    vx : scalar, optional
        Velocity in x direction. The default is 0.
    vy : scalar, optional
        Velocity in y direction. The default is 0.

    Returns
    -------
    G : 1D numpy array
        Vector with the autocorrelation G(tau).

    """
    
    tauD = w**2 / 4 / D
    G = N * (1 + tau/tauD) * np.sqrt(1 + tau/(tauD*SF**2))
    G = 1 / G
    G = G * np.exp(-((rhox - vx*tau)**2 + (rhoy - vy*tau)**2) / w**2 / (1 + tau/tauD))
    G += offset
    
    return G


def fcs_2c_2d_analytical(tau, N, tauD1, tauD2, F, alpha=1, T=0, tautrip=1e-6, offset=0, A=0, B=0):
    # amplitude
    Gy = N * (F + alpha*(1-F))**2
    Gy = 1 / Gy
    
    # triplet
    Gy *= (1 + (T * np.exp(-tau / tautrip)) / (1 - T))
    
    # diffusion
    Gy *= F / (1 + tau/tauD1) + alpha**2 * (1-F) / (1 + tau/tauD2)

    # offset
    Gy += offset

    # afterpulsing (see e.g. Buchholz, Biophys J., 2018)
    Gy += A * tau**B

    return Gy

def nanosecond_fcs_analytical(tau, A, c_ab, tau_ab, c_conf, tau_conf, c_rot, tau_rot, c_trip, tau_trip, tauD, SP):
    # source: Galvanetto et al., Nature, 2023
    G = A
    G *= (1 - c_ab * np.exp(-tau / tau_ab)) # antibunching
    G *= (1 + c_conf * np.exp(-tau / tau_conf)) # conformational dynamics
    G *= (1 + c_rot * np.exp(-tau / tau_rot)) # rotational dynamics
    G *= (1 + c_trip * np.exp(-tau / tau_trip)) # triplet
    G /= ((1 + tau/tauD) * np.sqrt(1 + tau / SP**2 / tauD))
    
    return G

def uncoupled_reaction_diffusion(tau, A, tauD, SP, f_eq, k_off):
    """
    Uncoupled reaction and diffusion model
    Assumes that tauD << 1/k_on
    See Mazza et al., ch 12, Monitoring Dynamic Binding of Chromatin Proteins
    In Vivo by Fluorescence Correlation Spectroscopy
    and Temporal Image Correlation Spectroscopy
    """
    G = A
    G /= ((1 + tau/tauD) * np.sqrt(1 + tau / SP**2 / tauD))
    G += (1-f_eq)*np.exp(-k_off * tau)
    
    return G
