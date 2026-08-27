import numpy as np
from scipy.special import erfcx


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
    """
    Calculate the analytical fcs autocorrelation function assuming 3D Gaussian
    diffusion with triplet state, afterpulsing and 2 components anomalous diffusion

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
    alpha1 : scalar
        Anomalous diffusion parameter species 1
    alpha2 : scalar
        Anomalous diffusion parameter species 2
    F : scalar
        Fraction of species 1.
    T : scalar, optional
        Fraction in triplet. The default is 0.
    tautrip : scalar
        Residence time in triplet state [s]. The default is 1e-6.
    SF : scalar
        Shape factor of the PSF. The default is 5.
    offset : scalar
        Offset. The default is 0.
    brightness : scalar
        Relative brightness species2/species1

    Returns
    -------
    Gy : 1D numpy array
        Vector with the autocorrelation G(tau).

    """
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


def fcs_analytical_2c_anomalous_c(tau, c, D1, D2, alpha1, alpha2, F, T, tau_triplet, w, SF, offset, brightness):
    """
    Calculate the analytical fcs autocorrelation function assuming 3D Gaussian
    diffusion with triplet state, afterpulsing and 2 components anomalous diffusion

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    c : scalar
        Particle concentration [/um^3]
        N = w0^2 * z0 * c * pi^(3/2).
    tauD1 : scalar
        Diffusion coefficient species 1 [um^2/s].
    tauD2 : scalar
        Diffusion coefficient species 2 [um^2/s].
    alpha1 : scalar
        Anomalous diffusion parameter species 1
    alpha2 : scalar
        Anomalous diffusion parameter species 2
    F : scalar
        Fraction of species 1.
    T : scalar, optional
        Fraction in triplet.
    tautrip : scalar
        Residence time in triplet state [s].
    w : scalar
        Beam waist [um]
    SF : scalar
        Shape factor of the PSF.
    offset : scalar
        Offset. The default is 0.
    brightness : scalar
        Relative brightness species2/species1

    Returns
    -------
    Gy : 1D numpy array
        Vector with the autocorrelation G(tau).

    """
    # effective volume
    V_eff = np.pi**(3/2) * w**3 * SF
    N = c * V_eff
    
    # diffusion time
    tauD1 = w**2 / 4 / D1
    tauD2 = w**2 / 4 / D2
    
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


def fcs_dualfocus_c(tau, c, D, w, SF, rhox, rhoy, offset, vx=0, vy=0):
    """
    Calculate the analytical fcs crosscorrelation function for dual focus fcs
    assuming 3D Gaussian and diffusion without triplet state
    Equation from Scipioni, Nat. Comm., 2018 and consistent with own Maple
    calculations

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    c : scalar
        Particle concentration [/um^3]
        N = w0^2 * z0 * c * pi^(3/2)
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
    
    V_eff = np.pi**(3/2) * w**3 * SF
    N = c * V_eff
    
    tauD = w**2 / 4 / D
    G = N * (1 + tau/tauD) * np.sqrt(1 + tau/(tauD*SF**2))
    G = 1 / G
    G = G * np.exp(-((rhox - vx*tau)**2 + (rhoy - vy*tau)**2) / w**2 / (1 + tau/tauD))
    G += offset
    
    return G


def fcs_circular_scanning(tau, N, tauD, w, SF, orbit_time, orbit_radius, offset, vx=0, vy=0):
    """
    Orbital scanning correlation formula

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    N : scalar
        Number of particles in the focal volume
    tauD : scalar
        Diffusion time of the fluorophores/particles [s].
    w : scalar
        1/e^2 radius of the effective PSF [m]
    SF : scalar
        Shape factor of the PSF.
    orbit_time : scalar
        Orbit tine [s]
    orbit_radius : scalar
        Orbit radius [um]
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
    
    alpha = 2 * np.pi / orbit_time * tau
    rho = orbit_radius * np.sqrt(2-2*np.cos(alpha)) # = 2 * Rcirc * abs(sin(alpha/2))
    D = w**2 / 4 / tauD
    
    G = fcs_dualfocus(tau, N, D, w, SF, rho, 0, offset, vx=vx, vy=vy)
    
    return G


def fcs_circular_scanning_c(tau, c, D, w, SF, orbit_time, orbit_radius, offset, vx=0, vy=0):
    """
    Orbital scanning correlation formula

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    c : scalar
        Particle concentration [/um^3]
        N = w0^2 * z0 * c * pi^(3/2)
        and w0 the effective focal volume (w0^2 + w1^2) / 2.
    D : scalar
        Diffusion coefficient of the fluorophores/particles [um^2/s].
    w : scalar
        1/e^2 radius of the effective PSF [um]
    SF : scalar
        Shape factor of the PSF.
    orbit_time : scalar
        Orbit tine [s]
    orbit_radius : scalar
        Orbit radius [um]
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
    
    alpha = 2 * np.pi / orbit_time * tau
    rho = orbit_radius * np.sqrt(2-2*np.cos(alpha)) # = 2 * Rcirc * abs(sin(alpha/2))
    
    G = fcs_dualfocus_c(tau, c, D, w, SF, rho, 0, offset, vx=0, vy=0)
    
    return G


def fcs_2c_2d_analytical(tau, N, tauD1, tauD2, F, alpha=1, T=0, tautrip=1e-6, offset=0, A=0, B=0):
    """
    Calculate the analytical fcs autocorrelation function assuming 2D free diffusion
    with triplet state, afterpulsing and 2 components

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    N : scalar
        Number of particles on average in the focal volume [dimensionsless]
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
    Gy *= F / (1 + tau/tauD1) + alpha**2 * (1-F) / (1 + tau/tauD2)

    # offset
    Gy += offset

    # afterpulsing (see e.g. Buchholz, Biophys J., 2018)
    Gy += A * tau**B

    return Gy


def nanosecond_fcs_analytical(tau, A, c_ab, tau_ab, c_conf, tau_conf, c_rot, tau_rot, c_trip, tau_trip, tauD, SP):
    """
    Calculate the analytical fcs autocorrelation function for nanosecond fcs

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    A : scalar
        Amplitude of the autocorrelation function.
    c_ab : scalar
        Amplitude of the antibunching effect.
    tau_ab : scalar
        Characteristic antibunching time [s].
    c_conf : scalar
        Amplitude of the conformational changes effect.
    tau_conf : scalar
        Characteristic time for the conformational changes time [s].
    c_rot : scalar
        Amplitude of the rotational diffusion.
    tau_rot : scalar
        Characteristic time for the rotational diffusion [s].
    c_trip : scalar
        Amplitude of the triplet effect.
    tau_trip : scalar
        Characteristic time for the triplet state [s].
    tauD : scalar
        Amplitude of the translational diffusion.
    SP : scalar
        Shape parameter.

    Returns
    -------
    G : 1D numpy array
        Vector with the autocorrelation G(tau).

    """
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


def fcs_finitelength(tau, N, tauD, SF, brightness, T, Tsampling):
    """
    Fit function finite length, 3D free diffusion
    Based on Kohler et al., Biophys. J., 2023

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    N : scalar
        Number of particles on average in the focal volume [dimensionsless]
        N = w0^2 * z0 * c * pi^(3/2).
        with c the average particle concentration
    tauD : scalar
        Diffusion time species 1 [s].
    SF : scalar
        Shape factor.
    brightness : scalar, optional
        Brightness (photons per molecule per second)
    T : scalar
        Chunk duration [s]
    Tsampling : scalar
        Dwell time [s]

    Returns
    -------
    G : 1D np.array
        Correlation function.

    """
    
    FCStheo = fcs_analytical(tau, N, tauD, SF, 0, 0, 0, 1)
    gamma = compute_gamma(tau, N, tauD, T, Tsampling, gamma_factor=1, SP=SF, brightness=brightness)
    G = FCStheo + gamma
    return G


def fcs_1c_2dgl(tau, N, tauD, SF, offset):
    """
    Fit function 1 component, 3D free diffusion
    2D gaussian in xy plus lorentzian in z
    See Leclerc et al., Physical Rev. Applied 26, 2026

    Parameters
    ----------
    tau : 1D numpy array
        Lag time [s] (vector).
    N : scalar
        Number of particles on average in the focal volume [dimensionsless]
        N = w0^2 * z0 * c * pi^(3/2).
        with c the average particle concentration
    tauD : scalar
        Diffusion time species 1 [s].
    SF : scalar
        Shape factor.
    Offset : scalar
        Offset

    Returns
    -------
    G : 1D np.array
        Correlation function.

    """
    # standard autocorrelation function
    Gy = np.sqrt(np.pi) / N * SF / (1 + (tau / tauD)) / np.sqrt(tau / tauD)
    Gy *= erfcx(np.sqrt(tauD / tau) * SF)
    Gy += offset  # offset

    if type(Gy) == np.float64:
        Garray = np.zeros((1, 2))
    else:
        Garray = np.zeros((np.size(Gy, 0), 2))
    Garray[:, 0] = tau
    Garray[:, 1] = Gy

    return Gy

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
